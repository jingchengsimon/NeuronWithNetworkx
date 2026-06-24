from pathlib import Path
import argparse
import pickle
import sys

import pandas as pd
from neuron import h

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.morphology_graph import create_directed_graph


def _section_lists(cell):
    sections_soma = [section for section in map(list, list(cell.soma))]
    sections_basal = [section for section in map(list, list(cell.basal))]
    sections_apical = [section for section in map(list, list(cell.apical))]
    return sections_soma + sections_basal + sections_apical


def build_segment_graph(swc_file, biophys_file, template_file):
    h.load_file("import3d.hoc")
    h.load_file(str(biophys_file))
    h.load_file(str(template_file))

    cell = h.L5PCtemplate(str(swc_file))
    all_sections = _section_lists(cell)
    all_segments_noaxon = [segment for section in all_sections for segment in section]

    section_df = pd.DataFrame(columns=[
        "parent_id",
        "section_id",
        "parent_name",
        "section_name",
        "length",
        "branch_idx",
        "section_type",
    ])

    return create_directed_graph(
        all_sections,
        all_segments_noaxon,
        section_df,
        return_segment_graph=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate section- and segment-level morphology graphs.")
    parser.add_argument("--swc", default="model/cell1.asc")
    parser.add_argument("--biophys", default="model/L5PCbiophys3.hoc")
    parser.add_argument("--template", default="model/L5PCtemplate.hoc")
    parser.add_argument("--out-dir", default="results/morphology/segment_graph")
    args, _ = parser.parse_known_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    section_df, section_dig, segment_df, segment_dig = build_segment_graph(
        Path(args.swc),
        Path(args.biophys),
        Path(args.template),
    )

    section_df.to_csv(out_dir / "section_df.csv", index=False)
    segment_df.to_csv(out_dir / "segment_df.csv", index=False)

    with (out_dir / "section_DiG.pkl").open("wb") as handle:
        pickle.dump(section_dig, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with (out_dir / "segment_DiG.pkl").open("wb") as handle:
        pickle.dump(segment_dig, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"sections: {section_dig.number_of_nodes()} nodes, {section_dig.number_of_edges()} edges")
    print(f"segments: {segment_dig.number_of_nodes()} nodes, {segment_dig.number_of_edges()} edges")
    print(f"output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
