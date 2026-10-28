"""Freeze deposited polymer sequences and author-to-input residue mappings."""

from pathlib import Path

import httpx
import numpy as np
import yaml
from biotite.structure.io.pdbx import CIFFile

from .storage import atomic_json, read_json, sha256
from .types import Residue, Target

AA3 = dict(
    zip(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
        "ARNDCQEGHILKMFPSTWYV",
        strict=True,
    )
)


def column(category, key: str, default: str = "") -> list[str]:
    if key not in category:
        return [default] * category.row_count
    return category[key].as_array(str).tolist()


def clean(value: str) -> str:
    return "" if value in (".", "?") else value


def parse_target(spec: dict, source: Path, prepared: Path) -> Target:
    block = CIFFile.read(source).block
    poly = block["entity_poly"]
    matches = [
        i
        for i, chains in enumerate(column(poly, "pdbx_strand_id"))
        if spec["chain"] in chains.split(",")
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one polymer for {spec['pdb_id']}:{spec['chain']}")
    row = matches[0]
    entity = column(poly, "entity_id")[row]
    sequence = "".join(column(poly, "pdbx_seq_one_letter_code_can")[row].split())
    if len(sequence) != spec["expected_length"] or set(sequence) - set(AA3.values()):
        raise ValueError(f"Unexpected canonical sequence for {spec['id']}")
    scheme = block["pdbx_poly_seq_scheme"]
    scheme_rows = []
    keys = [
        "seq_id",
        "entity_id",
        "asym_id",
        "pdb_strand_id",
        "mon_id",
        "pdb_seq_num",
        "pdb_ins_code",
    ]
    for values in zip(*(column(scheme, k) for k in keys), strict=True):
        item = dict(zip(keys, values, strict=True))
        if item["entity_id"] == entity and item["pdb_strand_id"] == spec["chain"]:
            scheme_rows.append(item)
    if len(scheme_rows) != len(sequence):
        raise ValueError("Incomplete polymer sequence mapping")
    atom = block["atom_site"]
    atom_keys = [
        "label_atom_id",
        "label_asym_id",
        "label_seq_id",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "label_alt_id",
        "pdbx_PDB_model_num",
    ]
    coordinates = {}
    asym = scheme_rows[0]["asym_id"]
    for values in zip(*(column(atom, k) for k in atom_keys), strict=True):
        item = dict(zip(atom_keys, values, strict=True))
        if (
            item["label_asym_id"] != asym
            or item["label_atom_id"] != "CA"
            or item["pdbx_PDB_model_num"] != "1"
        ):
            continue
        position = int(item["label_seq_id"])
        # Highest occupancy; blank then A then lexical alternate wins ties.
        priority = (-float(item["occupancy"]), clean(item["label_alt_id"]))
        xyz = [float(item[k]) for k in ("Cartn_x", "Cartn_y", "Cartn_z")]
        if not np.isfinite(xyz).all():
            raise ValueError("Non-finite experimental coordinates")
        if position not in coordinates or priority < coordinates[position][0]:
            coordinates[position] = (priority, xyz)
    overrides = spec.get("overrides", [])
    sequence_list = list(sequence)
    for override in overrides:
        index = override["position"] - 1
        mapped = next(r for r in scheme_rows if int(r["seq_id"]) == index + 1)
        if (
            sequence_list[index] != override["from_residue"]
            or mapped["pdb_seq_num"] != override["author_residue"]
        ):
            raise ValueError("Sequence override no longer matches deposited source")
        sequence_list[index] = override["to_residue"]
    sequence = "".join(sequence_list)
    residues = []
    for item in sorted(scheme_rows, key=lambda r: int(r["seq_id"])):
        position = int(item["seq_id"])
        modified = item["mon_id"] not in AA3
        ca = coordinates.get(position, (None, None))[1]
        residues.append(
            Residue(
                position=position,
                amino_acid=sequence[position - 1],
                author_number=item["pdb_seq_num"],
                insertion_code=clean(item["pdb_ins_code"]),
                deposited_name=item["mon_id"],
                observed=ca is not None,
                modified=modified,
                reference_eligible=ca is not None and not modified,
                ca=ca,
            )
        )
    ligands = []
    if "pdbx_entity_nonpoly" in block:
        ligands = sorted(set(column(block["pdbx_entity_nonpoly"], "comp_id")) - {"HOH"})
    return Target(
        id=spec["id"],
        name=spec["name"],
        pdb_id=spec["pdb_id"],
        chain=spec["chain"],
        sequence=sequence,
        sequence_sha256=sha256(sequence),
        source_url=f"https://files.rcsb.org/download/{spec['pdb_id']}.cif",
        source_sha256=sha256(source.read_bytes()),
        source_file=str(source.resolve()),
        prepared_file=str(prepared.resolve()),
        notes=spec["notes"],
        residues=residues,
        overrides=overrides,
        ligands=ligands,
    )


def prepare_targets(manifest="configs/targets.yaml", *, root=None, offline=False) -> list[Target]:
    root = Path(root or Path.cwd()).resolve()
    manifest = Path(manifest)
    if not manifest.is_absolute():
        manifest = root / manifest
    specs = yaml.safe_load(manifest.read_text())["targets"]
    if len({s["id"] for s in specs}) != len(specs):
        raise ValueError("Duplicate target IDs")
    lock_path = root / "inputs/sources.lock.json"
    locked = read_json(lock_path) if lock_path.exists() else {}
    targets = []
    for spec in specs:
        source = root / "data/raw" / f"{spec['pdb_id']}.cif"
        prepared = root / "data/prepared" / f"{spec['id']}.json"
        if not source.exists():
            if offline:
                raise FileNotFoundError(f"Missing cached PDB source: {source}")
            response = httpx.get(
                f"https://files.rcsb.org/download/{spec['pdb_id']}.cif",
                timeout=60,
                follow_redirects=True,
            )
            response.raise_for_status()
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(response.content)
        target = parse_target(spec, source, prepared)
        fingerprint = {
            "source_url": target.source_url,
            "source_sha256": target.source_sha256,
            "sequence_sha256": target.sequence_sha256,
        }
        if spec["id"] in locked and locked[spec["id"]] != fingerprint:
            raise ValueError(
                f"Frozen source changed for {target.id}; review before refreshing the lock"
            )
        locked[spec["id"]] = fingerprint
        atomic_json(prepared, target.model_dump(mode="json"))
        targets.append(target)
    fasta = "".join(
        f">{t.id} {t.pdb_id}:{t.chain} sha256={t.sequence_sha256}\n{t.sequence}\n" for t in targets
    )
    fasta_path = root / "inputs/targets.fasta"
    if fasta_path.exists() and fasta_path.read_text() != fasta:
        raise ValueError("Frozen FASTA differs from resolved targets")
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text(fasta)
    atomic_json(lock_path, locked)
    snapshot = []
    for target in targets:
        item = target.model_dump(mode="json")
        item["source_file"] = str(Path(target.source_file).relative_to(root))
        item["prepared_file"] = str(Path(target.prepared_file).relative_to(root))
        snapshot.append(item)
    atomic_json(
        root / "inputs/references.json",
        {"kind": "experimental_reference_coordinates", "targets": snapshot},
    )
    return targets


def load_reference_snapshot(root=None) -> list[Target]:
    """Load frozen experimental CA coordinates for fully offline notebook validation."""
    root = Path(root or Path.cwd()).resolve()
    data = read_json(root / "inputs/references.json")
    if data.get("kind") != "experimental_reference_coordinates":
        raise ValueError("Unexpected reference snapshot kind")
    targets = [Target.model_validate(item) for item in data["targets"]]
    locked = read_json(root / "inputs/sources.lock.json")
    for target in targets:
        if (
            sha256(target.sequence) != locked[target.id]["sequence_sha256"]
            or target.source_sha256 != locked[target.id]["source_sha256"]
        ):
            raise ValueError("Frozen reference snapshot does not match input lock")
        target.source_file = str(root / target.source_file)
        target.prepared_file = str(root / target.prepared_file)
    return targets
