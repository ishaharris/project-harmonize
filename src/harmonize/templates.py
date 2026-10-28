"""Search experimental PDB polymers and freeze explicit OpenFold template chains."""

import io
import json
import re
import time

import httpx
import numpy as np
from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal
from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile

from .providers import ProviderError
from .storage import atomic_json, sha256
from .targets import AA3, column

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"


def read_public(client, url, *, params=None, sleep=time.sleep):
    """Bounded retries for read-only public data; never send NVIDIA credentials."""
    for attempt in range(3):
        try:
            response = client.get(url, params=params)
        except httpx.TransportError:
            response = None
        if response is not None and response.status_code in (200, 204):
            return response
        if response is not None and response.status_code not in (429, 500, 502, 503, 504):
            raise ProviderError(f"Public template source returned HTTP {response.status_code}")
        if attempt < 2:
            sleep(2**attempt)
    raise ProviderError("Public template source unavailable after bounded retries")


def template_query(sequence, policy):
    return {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": policy["evalue_cutoff"],
                "identity_cutoff": policy["min_identity"],
                "sequence_type": "protein",
                "value": sequence,
            },
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": policy["max_candidates"]},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }


def prepare_template(text, entity, query, policy):
    """Select one polymer chain, preserve sequence numbering, and rename it A.

    OpenFold2's current direct-template path uses label_asym_id A and requires
    >=90% identity relative to the shorter sequence. Local filtering is conservative;
    the service performs its own Kalign validation.
    """
    source = CIFFile.read(io.StringIO(text)).block
    poly = source["entity_poly"]
    rows = [i for i, value in enumerate(column(poly, "entity_id")) if value == entity]
    if len(rows) != 1:
        raise ValueError("Template entity is missing or ambiguous")
    row = rows[0]
    sequence = "".join(column(poly, "pdbx_seq_one_letter_code_can")[row].split())
    if not sequence or set(sequence) - set(AA3.values()):
        raise ValueError("Template polymer contains unsupported residues")
    seq_cat = source["entity_poly_seq"]
    seq_rows = np.array(column(seq_cat, "entity_id")) == entity
    if any(x not in AA3 for x in np.array(column(seq_cat, "mon_id"))[seq_rows]):
        raise ValueError("Template polymer contains modified residues")
    alignment = align_optimal(
        ProteinSequence(query),
        ProteinSequence(sequence),
        SubstitutionMatrix.std_protein_matrix(),
        gap_penalty=(-10, -1),
        max_number=1,
    )[0]
    trace = alignment.trace
    matched = trace[(trace >= 0).all(1)]
    identity = sum(query[i] == sequence[j] for i, j in matched) / min(len(query), len(sequence))
    coverage = len(matched) / len(query)
    if identity < policy["min_identity"] or coverage < policy["min_query_coverage"]:
        raise ValueError("Template sequence identity or query coverage is below threshold")
    atom = source["atom_site"]
    entities = np.array(column(atom, "label_entity_id"))
    chains = np.array(column(atom, "label_asym_id"))
    models = np.array(column(atom, "pdbx_PDB_model_num", "1"))
    names = np.array(column(atom, "label_atom_id"))
    seq_ids = np.array(column(atom, "label_seq_id"))
    eligible = []
    for chain in sorted(set(chains[entities == entity])):
        mask = (entities == entity) & (chains == chain) & (models == "1")
        backbone = set.intersection(
            *[set(seq_ids[mask & (names == name)]) - {".", "?"} for name in ("N", "CA", "C")]
        )
        if len(backbone) / len(sequence) >= policy["min_backbone_coverage"]:
            eligible.append((len(backbone), chain, mask))
    if not eligible:
        raise ValueError("Template has insufficient observed backbone coordinates")
    observed, chain, mask = sorted(eligible, key=lambda x: (-x[0], x[1]))[0]
    block = CIFBlock()
    # Keep required chemical/experimental metadata; omit categories referencing discarded chains.
    for name in (
        "entry",
        "exptl",
        "pdbx_audit_revision_history",
        "pdbx_database_status",
        "chem_comp",
    ):
        if name in source:
            block[name] = CIFCategory(
                {key: value.as_array(str) for key, value in source[name].items()}
            )
    filters = {
        "atom_site": mask,
        "entity_poly": np.arange(poly.row_count) == row,
        "entity_poly_seq": seq_rows,
    }
    if "entity" in source:
        filters["entity"] = np.array(column(source["entity"], "id")) == entity
    if "struct_asym" in source:
        filters["struct_asym"] = np.array(column(source["struct_asym"], "id")) == chain
    if "pdbx_poly_seq_scheme" in source:
        filters["pdbx_poly_seq_scheme"] = (
            np.array(column(source["pdbx_poly_seq_scheme"], "asym_id")) == chain
        )
    rename = {
        "atom_site": ("label_asym_id", "auth_asym_id"),
        "entity_poly": ("pdbx_strand_id",),
        "struct_asym": ("id",),
        "pdbx_poly_seq_scheme": ("asym_id", "pdb_strand_id"),
    }
    for name, selected in filters.items():
        values = {key: value.as_array(str)[selected] for key, value in source[name].items()}
        for key in rename.get(name, ()):
            if key in values:
                values[key] = np.full(len(values[key]), "A")
        block[name] = CIFCategory(values)
    output = CIFFile({"harmonize_template": block})
    stream = io.StringIO()
    output.write(stream)
    return stream.getvalue(), {
        "source_label_chain": chain,
        "submitted_chain": "A",
        "identity_to_shorter": identity,
        "query_coverage": coverage,
        "backbone_coverage": observed / len(sequence),
        "sequence_sha256": sha256(sequence),
    }


def search_templates(target, policy, directory, client, *, sleep=time.sleep):
    directory.mkdir(parents=True, exist_ok=True)
    query = template_query(target.sequence, policy)
    atomic_json(directory / "search_request.json", query)
    response_path = directory / "search_response.json"
    if not response_path.exists():
        response = read_public(client, RCSB_SEARCH, params={"json": json.dumps(query)}, sleep=sleep)
        atomic_json(
            response_path, {"result_set": []} if response.status_code == 204 else response.json()
        )
    payload = json.loads(response_path.read_text())
    hits = payload.get("result_set", [])
    if not isinstance(hits, list):
        raise ProviderError("Malformed RCSB template search results")
    selected, rejected, entries = [], [], set()
    ordered = sorted(hits, key=lambda hit: (-float(hit["score"]), hit["identifier"]))
    for hit in ordered[: policy["max_candidates"]]:
        identifier = hit["identifier"]
        if not re.fullmatch(r"[A-Za-z0-9]{4}_[0-9]+", identifier):
            raise ProviderError("Unexpected RCSB polymer entity identifier")
        pdb_id, entity = identifier.upper().split("_")
        if pdb_id in entries:
            continue
        source = directory / f"{pdb_id}.source.cif"
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        if not source.exists():
            source.write_bytes(read_public(client, url, sleep=sleep).content)
        try:
            structure, info = prepare_template(source.read_text(), entity, target.sequence, policy)
        except (ValueError, KeyError, IndexError) as exc:
            rejected.append({"identifier": identifier, "reason": str(exc)})
            continue
        filename = f"{identifier}.template.cif"
        (directory / filename).write_text(structure)
        selected.append(
            {
                "identifier": identifier,
                "pdb_id": pdb_id,
                "entity_id": entity,
                "search_score": float(hit["score"]),
                "source_url": url,
                "source_file": source.name,
                "source_sha256": sha256(source.read_bytes()),
                "file": filename,
                "sha256": sha256(structure),
                "same_entry_as_reference": pdb_id == target.pdb_id.upper(),
                **info,
            }
        )
        entries.add(pdb_id)
        if len(selected) == policy["max_templates"]:
            break
    atomic_json(
        directory / "selection.json", {"selected": selected, "rejected": rejected, "policy": policy}
    )
    if not selected:
        raise ProviderError(
            "No eligible structural templates found; template-free fallback is disabled"
        )
    return selected
