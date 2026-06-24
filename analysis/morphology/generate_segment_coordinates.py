"""Export soma-centered segment coordinates and render the L5PC morphology.

The JSON output is newline-delimited JSON (one segment per line).  Segment
indices follow the same soma -> basal -> apical traversal used by
L5PNModel.all_segments_noaxon and the global segment recordings.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MORPHOLOGY = REPO_ROOT / "model" / "cell1.asc"
DEFAULT_BIOPHYSICS = REPO_ROOT / "model" / "L5PCbiophys3.hoc"
DEFAULT_TEMPLATE = REPO_ROOT / "model" / "L5PCtemplate.hoc"
DEFAULT_JSON_OUT = REPO_ROOT / "results" / "morphology" / "segment_center_coordinates_soma_origin.json"
DEFAULT_PNG_OUT = REPO_ROOT / "results" / "morphology" / "segment_center_coordinates_soma_origin.png"


def _mechanisms_available(h) -> bool:
    """Return whether the custom Hay-model mechanisms are already loaded."""
    probe = h.Section(name="segment_coordinate_mechanism_probe")
    try:
        probe.insert("CaDynamics_E2")
        probe.insert("NaTa_t")
    except Exception:
        return False
    finally:
        h.delete_section(sec=probe)
    return True


def _load_local_mechanisms(h) -> Path | None:
    """Load the repository's compiled mechanisms when NEURON has not done so."""
    if _mechanisms_available(h):
        return None

    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        candidates = [
            REPO_ROOT / "arm64" / "libnrnmech.dylib",
            REPO_ROOT / "arm64" / ".libs" / "libnrnmech.dylib",
            REPO_ROOT / "arm64" / ".libs" / "libnrnmech.so",
        ]
    elif machine in {"x86_64", "amd64"}:
        candidates = [
            REPO_ROOT / "x86_64" / "libnrnmech.so",
            REPO_ROOT / "x86_64" / ".libs" / "libnrnmech.so",
            REPO_ROOT / "mod" / "x86_64" / ".libs" / "libnrnmech.so",
        ]
    else:
        candidates = []

    candidates.extend(
        [
            REPO_ROOT / "mod" / "nrnmech.dll",
            REPO_ROOT / "nrnmech.dll",
        ]
    )

    for candidate in candidates:
        if not candidate.exists():
            continue
        h.nrn_load_dll(str(candidate))
        if _mechanisms_available(h):
            return candidate

    raise FileNotFoundError(
        "Could not load the compiled NEURON mechanisms required by L5PCbiophys3.hoc. "
        "Run nrnivmodl or check the arm64/x86_64 mechanism output."
    )


def load_l5pc(morphology: Path, biophysics: Path, template: Path):
    """Instantiate the same L5PC morphology and discretization used in simulations."""
    from neuron import h

    _load_local_mechanisms(h)
    h.load_file("import3d.hoc")
    h.load_file(str(biophysics))
    h.load_file(str(template))
    cell = h.L5PCtemplate(str(morphology))
    h.define_shape()
    return h, cell


def xyz_at_section_location(h, section, location: float) -> np.ndarray:
    """Interpolate a section location in [0, 1] onto its pt3d centerline."""
    location = float(location)
    if not 0.0 <= location <= 1.0:
        raise ValueError(f"section location must be in [0, 1], got {location}")

    n3d = int(section.n3d())
    if n3d < 2:
        raise ValueError(f"section {section.name()} has fewer than two pt3d points")

    arc = np.fromiter((section.arc3d(i) for i in range(n3d)), dtype=float, count=n3d)
    x3d = np.fromiter((section.x3d(i) for i in range(n3d)), dtype=float, count=n3d)
    y3d = np.fromiter((section.y3d(i) for i in range(n3d)), dtype=float, count=n3d)
    z3d = np.fromiter((section.z3d(i) for i in range(n3d)), dtype=float, count=n3d)

    # The first pt3d point is at the end connected to the parent.  This is
    # normally section(0), but section_orientation makes the mapping robust to
    # sections whose section(1) end is connected instead.
    if int(h.section_orientation(sec=section)) == 0:
        target_arc = location * float(section.L)
    else:
        target_arc = (1.0 - location) * float(section.L)

    return np.array(
        [
            np.interp(target_arc, arc, x3d),
            np.interp(target_arc, arc, y3d),
            np.interp(target_arc, arc, z3d),
        ],
        dtype=float,
    )


def collect_segment_centers(h, cell):
    """Collect soma-centered non-axon segment centers in recording order."""
    soma_sections = list(cell.soma)
    basal_sections = list(cell.basal)
    apical_sections = list(cell.apical)
    section_groups = (
        ("soma", soma_sections),
        ("basal", basal_sections),
        ("apical", apical_sections),
    )

    if not soma_sections:
        raise ValueError("the imported morphology contains no soma section")

    soma_origin = xyz_at_section_location(h, soma_sections[0], 0.5)
    records = []
    regions = []

    for region, sections in section_groups:
        for section in sections:
            for segment in section:
                xyz = xyz_at_section_location(h, section, segment.x) - soma_origin
                records.append(
                    {
                        "segment_idx": len(records),
                        "x": float(xyz[0]),
                        "y": float(xyz[1]),
                        "z": float(xyz[2]),
                    }
                )
                regions.append(region)

    return records, np.asarray(regions), soma_origin


def write_json_lines(records, output_path: Path) -> None:
    """Write one compact JSON object per line, as requested for segment data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")))
            handle.write("\n")


def _set_equal_3d_limits(axis, coordinates: np.ndarray) -> None:
    mins = coordinates.min(axis=0)
    maxs = coordinates.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def render_segment_centers(records, regions: np.ndarray, output_path: Path, dpi: int) -> None:
    """Render the segment-center point cloud with morphology-region colors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coordinates = np.asarray([[row["x"], row["y"], row["z"]] for row in records], dtype=float)
    colors = {"soma": "black", "basal": "#2774AE", "apical": "#D1495B"}
    sizes = {"soma": 22.0, "basal": 5.0, "apical": 5.0}

    figure = plt.figure(figsize=(10, 9), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")

    for region in ("basal", "apical", "soma"):
        mask = regions == region
        if not np.any(mask):
            continue
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            coordinates[mask, 2],
            s=sizes[region],
            c=colors[region],
            alpha=0.9,
            linewidths=0,
            label=region,
            depthshade=False,
        )

    axis.scatter([0], [0], [0], marker="*", s=130, c="#F2B134", edgecolors="black", linewidths=0.6, label="origin")
    axis.set_title(f"L5PC segment centers (soma-centered, n={len(records)})")
    axis.set_xlabel("x (µm)")
    axis.set_ylabel("y (µm)")
    axis.set_zlabel("z (µm)")
    axis.view_init(elev=18, azim=-62)
    _set_equal_3d_limits(axis, coordinates)
    axis.legend(loc="upper right", frameon=True)
    axis.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export soma-centered L5PC segment-center coordinates and a 3D PNG visualization."
    )
    parser.add_argument("--morphology", type=Path, default=DEFAULT_MORPHOLOGY)
    parser.add_argument("--biophysics", type=Path, default=DEFAULT_BIOPHYSICS)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--json_out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--png_out", type=Path, default=DEFAULT_PNG_OUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h, cell = load_l5pc(args.morphology.resolve(), args.biophysics.resolve(), args.template.resolve())
    records, regions, soma_origin = collect_segment_centers(h, cell)
    write_json_lines(records, args.json_out.resolve())
    render_segment_centers(records, regions, args.png_out.resolve(), args.dpi)

    print(f"segments: {len(records)}")
    print(f"original soma(0.5): {soma_origin.tolist()} µm")
    print(f"json: {args.json_out.resolve()}")
    print(f"png: {args.png_out.resolve()}")


if __name__ == "__main__":
    main()
