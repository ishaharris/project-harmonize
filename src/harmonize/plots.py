"""Notebook figures and an interactive viewer backed only by cached artifacts."""

import io
import math

import matplotlib.pyplot as plt
import numpy as np
from biotite.structure.io.pdb import PDBFile

from .structures import ca_atoms, reference_arrays, superpose
from .workflow import artifact_arrays

COLORS = ["#2166ac", "#d6604d", "#1b9e77", "#7b3294"]


def _title(analysis, title):
    return (
        f"VALIDATION FIXTURES — NOT MODEL PREDICTIONS\n{title}"
        if analysis.run.kind == "validation_fixture"
        else title
    )


def export_figure(figure, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(directory / f"{name}.{extension}", dpi=170, bbox_inches="tight")


def plot_overview(analysis, metric="tm_score"):
    nrows = math.ceil(len(analysis.run.targets) / 3)
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 4 * nrows), layout="constrained", squeeze=False)
    models = list(analysis.run.protocol.models)
    labels = [m.replace("fixture_", "").replace("_", " ") for m in models]
    for ax in list(axes.flat)[len(analysis.run.targets) :]:
        ax.set_visible(False)
    for ax, target in zip(axes.flat, analysis.run.targets):
        frame = analysis.pairs[
            (analysis.pairs.target_id == target.id)
            & (analysis.pairs.comparison_type != "experimental_reference")
            & (analysis.pairs.status == "complete")
        ]
        matrix = np.full((len(models), len(models)), np.nan)
        for i, left in enumerate(models):
            for j, right in enumerate(models):
                values = frame[
                    ((frame.left_model == left) & (frame.right_model == right))
                    | ((frame.left_model == right) & (frame.right_model == left))
                ][metric]
                if len(values):
                    matrix[i, j] = values.median()
        cmap = plt.get_cmap("viridis" if metric != "rmsd" else "magma_r").copy()
        cmap.set_bad("#e9ecef")
        ax.imshow(matrix, vmin=0, vmax=1 if metric in ("tm_score", "lddt") else None, cmap=cmap)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                "N/A" if np.isnan(value) else f"{value:.3f}",
                ha="center",
                va="center",
                color="#202020" if np.isnan(value) or value > 0.7 else "white",
                fontsize=11,
            )
        ax.set_xticks(range(len(models)), labels, rotation=20, ha="right")
        ax.set_yticks(range(len(models)), labels)
        ax.set_title(f"{target.name} · {target.pdb_id}:{target.chain}")
    fig.suptitle(
        _title(analysis, f"Median {metric.replace('_', ' ')} · diagonal = within-group agreement"),
        fontsize=15,
    )
    export_figure(fig, analysis.directory / "figures", f"overview_{metric}")
    return fig


def plot_residues(analysis, target_id):
    target = next(t for t in analysis.run.targets if t.id == target_id)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    frame = analysis.residues[
        (analysis.residues.target_id == target_id)
        & (analysis.residues.comparison_type == "cross_model")
    ]
    for i, ((left, right), group) in enumerate(frame.groupby(["left_model", "right_model"])):
        label = f"{left} / {right}".replace("fixture_", "")
        for ax, metric in zip(axes[:2], ("displacement", "lddt"), strict=True):
            grouped = group.groupby("position")[metric]
            mid, lo, hi = grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)
            ax.plot(mid.index, mid, label=label, color=COLORS[i % len(COLORS)])
            ax.fill_between(mid.index, lo, hi, color=COLORS[i % len(COLORS)], alpha=0.15)
    for i, (model, label) in enumerate(analysis.medoids.get(target_id, {}).items()):
        artifact = next(
            a for a in analysis.run.artifacts if a.target_id == target_id and a.label == label
        )
        values = artifact_arrays(analysis.run, artifact)["confidence"]
        axes[2].plot(
            np.arange(1, len(values) + 1),
            values,
            label=model.replace("fixture_", ""),
            color=COLORS[i % len(COLORS)],
        )
    axes[0].set_ylabel("Cα displacement (Å)")
    axes[1].set_ylabel("Symmetric Cα lDDT")
    axes[1].set_ylim(0, 1.02)
    axes[2].set_ylabel("pLDDT (0–100)\nwhere scale is known")
    axes[2].set_ylim(0, 102)
    axes[2].set_xlabel(
        "Input sequence position (1-based); author numbering is retained in residues.csv"
    )
    for ax in axes[:2]:
        if not ax.lines:
            ax.text(
                0.5,
                0.5,
                "No completed cross-model comparisons yet",
                ha="center",
                transform=ax.transAxes,
            )
    if not axes[2].lines:
        axes[2].text(
            0.5, 0.5, "No prediction confidence available", ha="center", transform=axes[2].transAxes
        )
    for ax in axes:
        ax.grid(alpha=0.15)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        _title(
            analysis, f"{target.name}: local agreement · bands show pairwise interquartile ranges"
        ),
        fontsize=13,
    )
    export_figure(fig, analysis.directory / "figures", f"residues_{target_id}")
    return fig


def plot_distributions(analysis, target_id, *, reference=False):
    frame = analysis.pairs[
        (analysis.pairs.target_id == target_id) & (analysis.pairs.status == "complete")
    ]
    frame = frame[
        (frame.comparison_type == "experimental_reference")
        if reference
        else (frame.comparison_type != "experimental_reference")
    ]
    groups = list(frame.groupby(["comparison_type", "left_model", "right_model"]))
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    if groups:
        values = [g.tm_score.to_numpy() for _, g in groups]
        labels = [f"{key[1]}\n↔ {key[2]}".replace("fixture_", "") for key, _ in groups]
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        for i, data in enumerate(values):
            ax.scatter(i + 1 + np.linspace(-0.15, 0.15, len(data)), data, s=14, alpha=0.65)
        ax.tick_params(axis="x", labelsize=8)
    else:
        ax.text(0.5, 0.5, "No completed comparisons", ha="center", transform=ax.transAxes)
    ax.set_ylabel("TM-score")
    ax.set_ylim(0, 1.03)
    context = (
        "experimental reference comparisons"
        if reference
        else "within-group and cross-group comparisons"
    )
    ax.set_title(_title(analysis, f"{target_id}: {context}"))
    export_figure(
        fig,
        analysis.directory / "figures",
        f"{'reference' if reference else 'distributions'}_{target_id}",
    )
    return fig


def structure_view(analysis, target_id, labels=None, include_reference=True):
    import py3Dmol

    target = next(t for t in analysis.run.targets if t.id == target_id)
    labels = list(analysis.medoids.get(target_id, {}).values()) if labels is None else labels
    structures = []
    for label in labels:
        artifact = next(
            a
            for a in analysis.run.artifacts
            if a.target_id == target_id and a.label == label and a.status == "complete"
        )
        structures.append((label, artifact_arrays(analysis.run, artifact)["xyz"]))
    if include_reference:
        structures.append(("experimental reference", reference_arrays(target)["xyz"]))
    if not structures:
        raise ValueError("No structures available for this target")
    view = py3Dmol.view(width=900, height=550)
    reference = structures[0][1]
    for index, (label, xyz) in enumerate(structures):
        present = np.isfinite(xyz).all(1)
        common = present & np.isfinite(reference).all(1)
        moved = xyz.copy()
        if index:
            _, transform, _ = superpose(reference[common], xyz[common])
            moved[present] = transform.apply(xyz[present])
        positions = np.flatnonzero(present) + 1
        color = COLORS[index % len(COLORS)]
        atoms = ca_atoms(moved[present], positions, target.sequence)
        pdb_file = PDBFile()
        pdb_file.set_structure(atoms)
        stream = io.StringIO()
        pdb_file.write(stream)
        view.addModel(stream.getvalue(), "pdb")
        view.setStyle({"model": index}, {"sphere": {"radius": 0.28, "color": color}})
        # Explicit CA traces preserve chain gaps rather than adding artificial bonds.
        breaks = np.flatnonzero(np.diff(positions) != 1) + 1
        for segment in np.split(moved[present], breaks):
            if len(segment) >= 2:
                view.addCurve(
                    {
                        "points": [
                            dict(zip(("x", "y", "z"), point.tolist(), strict=True))
                            for point in segment
                        ],
                        "radius": 0.13,
                        "color": color,
                        "smooth": 5,
                    }
                )
        view.addLabel(
            label.replace("fixture_", ""),
            {
                "position": {"x": 8, "y": 25 + 24 * index},
                "useScreen": True,
                "fontColor": color,
                "backgroundColor": "white",
                "showBackground": True,
                "fontSize": 13,
            },
        )
    view.setBackgroundColor("#f8fafc")
    view.zoomTo()
    return view


def interactive_viewer(analysis):
    import ipywidgets as widgets
    from IPython.display import display

    target_select = widgets.Dropdown(
        options=[(t.name, t.id) for t in analysis.run.targets], description="Protein"
    )
    sample_select = widgets.SelectMultiple(
        description="Samples", rows=6, layout=widgets.Layout(width="480px")
    )
    reference_toggle = widgets.Checkbox(value=True, description="Show experimental reference")
    output = widgets.Output()

    def refresh_samples(change=None):
        available = [
            a.label
            for a in analysis.run.artifacts
            if a.target_id == target_select.value and a.status == "complete"
        ]
        sample_select.options = available
        sample_select.value = tuple(analysis.medoids.get(target_select.value, {}).values())

    def render(change=None):
        with output:
            output.clear_output(wait=True)
            if analysis.run.kind == "validation_fixture":
                print(
                    "VALIDATION FIXTURES — perturbed experimental coordinates, not model predictions"
                )
            try:
                display(
                    structure_view(
                        analysis,
                        target_select.value,
                        list(sample_select.value),
                        reference_toggle.value,
                    )
                )
            except ValueError as exc:
                print(str(exc))

    target_select.observe(refresh_samples, names="value")
    for widget in (target_select, sample_select, reference_toggle):
        widget.observe(render, names="value")
    refresh_samples()
    render()
    return widgets.VBox([target_select, sample_select, reference_toggle, output])
