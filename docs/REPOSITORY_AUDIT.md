# Repository organization record

This document records the cleaned repository structure used for simulation, analysis, and writing.
Historical prototypes remain available through Git history but are not present in the current tree.

## Production simulation path

```text
L5b_simulation.py
├── model/
│   ├── cell1.asc
│   ├── L5PCtemplate.hoc
│   ├── L5PCbiophys3.hoc
│   ├── L5PCbiophys3withNaCa.hoc
│   └── AMPANMDA.json
├── mod/*.mod
├── analysis/nmda_spike_detection.py
└── utils/
    ├── l5pn_model.py
    ├── morphology_graph.py
    ├── cable_distance.py
    ├── synaptic_inputs.py
    ├── synapse_models.py
    ├── pink_noise.py
    ├── cluster_protocol.py
    └── random_streams.py
```

`L5b_simulation.py` is the production CLI. It resolves parameter combinations, builds
`L5PNModel`, runs the requested simulations, and writes run data below `--results_root`.

## Maintained directories

| Path | Responsibility | Git policy |
| --- | --- | --- |
| `model/` | Morphology, HOC templates, and model parameter files | Tracked |
| `mod/` | Hand-maintained NMODL mechanism sources | Tracked |
| `utils/` | Production simulation implementation | Tracked |
| `analysis/` | Post-simulation analysis, figures, diagnostics, and morphology tools | Tracked except notebooks |
| `scripts/` | Operational validation entry points | Tracked |
| `docs/` | Architecture and repository documentation | Tracked |
| `results/` | Generated simulation and analysis output | Local/remote only; ignored |
| `arm64/`, `x86_64/` | Locally compiled NEURON mechanisms | Ignored; rebuild from `mod/*.mod` |

Generated C/C++, object files, shared libraries, `special`, and Numba/Python caches are build
products rather than source. They are ignored even when generated inside `mod/`.

## Analysis layout

```text
analysis/
├── nmda_spike_detection.py
├── ap_ca_spike_analysis.py
├── trace_analysis.py
├── variability_analysis.py
├── single_cluster_nonlinearity.py
├── figures/
│   ├── plot_figure1_traces.py
│   ├── plot_nmda_rate_tertiles.py
│   ├── plot_soma_trials.py
│   └── plot_vitro_na_area.py
├── diagnostics/
│   └── check_ap_ca_rates.py
├── morphology/
│   ├── generate_segment_graph.py
│   ├── visualize_segment_graph.py
│   └── generate_segment_coordinates.py
└── notebooks/                 # local exploratory state; ignored by Git
```

`single_cluster_nonlinearity.py` is retained because it produces the EPSP summary consumed by
`plot_vitro_na_area.py`. It should eventually replace its hard-coded paths and global-variable
pickle contract with explicit CLI inputs and compact summary tables.

## Removed material

- `archive/`: obsolete prototypes and earlier model variants; retained in Git history only.
- `utils_viz/results_compression.py`: hard-coded one-off compression script.
- `utils_viz/simulate_L5PC_and_create_dataset.py`: unrelated legacy dataset workflow.
- `utils_viz/subtree_reductor_func.py`: unused morphology-reduction implementation.
- Separate `outputs/`: merged into the ignored `results/` tree.

## Public-result policy

Full simulation arrays, intermediate tables, caches, and regenerated figures remain under the
ignored `results/` directory. A small set of final README/manuscript figures may later be copied
to `docs/figures/` and intentionally tracked. This keeps Git history readable without losing
local or remote scientific outputs.
