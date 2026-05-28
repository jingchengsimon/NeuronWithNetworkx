# NeuronWithNetworkx — L5 Pyramidal Neuron Dendritic Clustering Simulation

Biophysically detailed compartmental simulation of a **Layer 5 pyramidal neuron (L5PN)**, studying how **dendritic synaptic clustering** drives nonlinear integration at the soma and dendrite. Built on the NEURON simulator with networkx-based morphological analysis.

---

## Quick Start

```bash
# 1. Compile NEURON mechanisms (once)
cd mod && nrnivmodl && cd ..

# 2. Run a single-cluster simulation (default parameters)
python L5b_simulation.py

# 3. Run with action potentials enabled
python L5b_simulation.py --with_ap

# 4. Run with full-segment recording (large memory)
python L5b_simulation.py --with_ap --with_global_rec
```

Key CLI arguments (see `python L5b_simulation.py --help` for full list):

| Argument | Default | Description |
|----------|---------|-------------|
| `--sec_type` | `basal` | Target dendritic region (`basal` / `apical`) |
| `--spat_condition` | `clus` | Spatial pattern (`clus` = clustered, `distr` = distributed control) |
| `--distance_to_root` | `0` | Distance zone for cluster placement (0 / 1 / 2) |
| `--num_clusters` | `1` | Number of synaptic clusters |
| `--cluster_radius` | `5.0` | Cluster spatial extent (µm) |
| `--bg_exc_freq` | `1.0` | Background excitatory frequency (Hz) |
| `--with_ap` | `False` | Enable Na/Ca channels (action potentials) |
| `--with_global_rec` | `False` | Record voltage/current at every segment |
| `--use_replay_bg` | `False` | Replay background spikes from a reference run |
| `--num_epochs` | `1` | Number of epochs (`start_epoch` .. `start_epoch + num_epochs - 1`) |
| `--start_epoch` | `1` | First epoch index |
| `--max_workers_epoch` | `20` | Max parallel `build_cell` processes per `spat_cond` |
| `--max_workers_synapse` | `30` | Max threads per process for synapse/input prep |

---

## Project Structure

```
.
├── L5b_simulation.py              # Main entry: CLI and parallel scheduling
│
├── modelFile/
│   ├── cell1.asc / cell1.swc      # Reconstructed L5PN morphology
│   ├── L5PCtemplate.hoc            # NEURON cell template (soma, basal, apical)
│   ├── L5PCbiophys3.hoc            # Passive biophysics (no AP)
│   ├── L5PCbiophys3withNaCa.hoc    # Active biophysics (with AP + Ca²⁺)
│   ├── AMPANMDA.json               # AMPA+NMDA synapse parameters
│   └── PN2PN.json                  # (Reserved) pyramidal-to-pyramidal params
│
├── mod/                            # NEURON mechanism source files (.mod)
│   └── x86_64/ or arm64/          # Compiled mechanisms (.so / .dll)
│
├── utils/                          # Core utility modules (see §Utility Modules below)
│   ├── cell_with_networkx.py       # CellWithNetworkx implementation
├── utils_anal/                     # Post-hoc analysis scripts
├── utils_viz/                      # Visualization & figure-generation scripts
│
├── docs/                           # Technical documentation
│   ├── CONVENTIONS.md              # Magic constants, abbreviations, schema reference
│   └── ARCHITECTURE.md             # Module dependency graph, data flow diagrams
│
├── AGENT.md                        # AI agent constraint specification
└── .cursor/rules/                  # Cursor IDE auto-injected rules
```

---

## Simulation Pipeline

`utils/cell_with_networkx.py` contains the `CellWithNetworkx` class, which executes the following pipeline. `L5b_simulation.py` is the CLI and parallel-scheduling entry point. **The order of steps 4a → 4b → 4c is immutable** — inhibitory inputs track excitatory activity and must be generated last.

```
Step 1: __init__()
   Load morphology (cell1.asc) → build networkx DiG → compute branch orders

Step 2: add_synapses()
   Distribute ~26k excitatory + ~2.8k inhibitory synapses across the dendritic tree
   (length-weighted random placement)

Step 3: assign_clustered_synapses()
   Select cluster center(s) within a distance zone → recruit surround synapses
   via exponential distance distribution → assign preunit-to-cluster mapping

Step 4: add_inputs()
   4a. add_background_exc_inputs()
       Generate pink noise → modulate Poisson spike trains → 50% dropout
       Create AMPA+NMDA synapses with log-normal weights

   4b. add_clustered_inputs()  [loop over activation levels]
       Inject cluster stimulus (VecStim) → merge with background spikes

   4c. add_background_inh_inputs()  [loop]
       Track total excitatory activity → generate coupled inhibitory spikes

   4d. run_simulation()  [loop]
       h.run() → record soma/dendrite/tuft voltage and synaptic currents

Step 5: Save outputs
   → *.npy arrays, section_synapse_df.csv, simulation_params.json
```

### Parallel Execution

`run_combination()` expands all tasks for each `spat_cond` **serially** (`clus` before `distr`):

```
epoch × simu_cond × sec_type × dis_to_root  →  ProcessPoolExecutor(max_workers_epoch)
```

- **Process level** (`--max_workers_epoch`, default 20): one `build_cell` per process; tasks beyond the limit queue in the pool. Always use `list(executor.map(...))` so worker failures are not swallowed.
- **Thread level** (`--max_workers_synapse`, default 30): within each process, `add_synapses()` and background input prep use `ThreadPoolExecutor`; `h.run()` always runs on the main thread (NEURON is not thread-safe).
- **No batch / epoch_mode CLI** — epoch range is only `--num_epochs` + `--start_epoch`.
- **Do not** mix `clus` and `distr` in the same process pool; distr replay depends on clus `section_synapse_df.csv`.

Batch var experiments: see `run_var_exp.sh` (single invocation with `--spat_cond clus distr --num_epochs 100`).

---

## Utility Modules (`utils/`)

### `cell_with_networkx.py` — Cell Class and Simulation Runner

Defines `CellWithNetworkx`: morphology loading, graph construction, synapse placement, cluster assignment, input orchestration, `h.run()`, and output serialization.

### `add_inputs_utils.py` — Synapse Creation & Spike Train Injection

The largest utility module. Three public functions implement the three input pathways:

- **`add_background_exc_inputs()`**: For each excitatory synapse, selects a random pink noise trace, generates a Poisson spike train modulated by the noise envelope, applies 50% presynaptic failure, and creates an AMPA+NMDA synapse object with a log-normal weight. Each synapse gets a deterministic per-index RNG seed to ensure reproducibility under multithreading.

- **`add_background_inh_inputs()`**: Computes the total excitatory spike histogram (background + clustered), then generates inhibitory Poisson spikes whose rate tracks the excitatory activity (λ_inh ∝ total_exc_spikes / mean). This E/I tracking produces balanced, in vivo–like dynamics.

- **`add_clustered_inputs()`**: For each cluster, looks up which preunits are active, retrieves their stimulus spike times, merges with existing background spikes (unique time points only), and attaches a VecStim. Old NetCon weights are zeroed before creating new connections.

Also contains replay variants (`_add_background_exc_inputs_replay`, `_add_background_inh_inputs_replay`) that read spike trains from a reference CSV instead of generating new ones.

### `generate_pink_noise.py` — 1/f Noise Generator

Produces temporally correlated (pink / 1/f) noise traces using an IIR filter applied to white Gaussian noise. The output is z-scored and the first 2000 samples are discarded as burn-in.

- **`make_noise(num_traces, num_samples, spike_gen_seed, scale)`**: Returns an array of shape `(num_traces, num_samples)`. The `scale` parameter controls the white noise standard deviation (default 0.5 for background activity).

The pink noise is **not replaceable with white noise** — its temporal autocorrelation structure is essential for producing realistic in vivo–like firing patterns.

### `generate_stim_utils.py` — Stimulus Generation & Preunit Mapping

Two key functions:

- **`generate_indices(rnd, num_clusters, num_conn_per_preunit, num_preunit)`**: Assigns preunits to clusters using a round-robin strategy that equalizes connection counts across clusters. Returns `indices[i]` = list of preunit IDs connected to cluster `i`.

- **`generate_vecstim(rnd, pre_unit_ids, num_stim, stim_time, stim_time_var=5)`**: For each preunit, generates `num_stim` spike times drawn from `Normal(stim_time, stim_time_var)` and floored to integer ms. The 5 ms jitter corresponds to typical cortical synchrony timescales.

### `generate_init_firing_utils.py` — Segment-Level Pink Noise (Experimental)

An alternative background generation approach that maps pink noise traces to individual dendritic segments (rather than random assignment per synapse). Uses a segment lookup table (`all_segments_dend.csv`) to match each synapse to its closest segment and assign a region-specific (basal vs. apical) noise trace.

> **Status**: Experimental. The main simulation uses `add_inputs_utils.py` instead.

### `graph_utils.py` — Morphology → Directed Graph

*(Not uploaded — description from imports and usage)*

- **`create_directed_graph(all_sections, all_segments, section_df)`**: Builds a networkx `DiGraph` where each node is a section and edges point from parent to child. Also populates `section_df` with section metadata (parent, length, type, branch index).

- **`set_graph_order(DiG, root_tuft_idx)`**: Computes branch order from two roots: soma (for the full tree) and tuft root (for the apical tuft subtree). Returns classification dictionaries mapping order → section indices.

### `distance_utils.py` — Cable Distance Calculation

*(Not uploaded — description from imports and usage)*

- **`recur_dist_to_soma(section, loc)`**: Recursively computes cable distance (µm) from a point on a section to the soma.
- **`recur_dist_to_root(section, loc, root_sec)`**: Cable distance to an arbitrary root section (used for tuft root).
- **`distance_synapse_mark_compare(distances, marks)`**: Given actual synapse-to-center distances and target distance marks (from exponential distribution), returns indices of synapses that fall within each mark — used for cluster member selection.

### `synapses_models.py` — AMPA+NMDA Synapse Wrapper

*(Not uploaded — description from usage)*

- **`AMPANMDA(syn_params, loc, section, channel_type)`**: Creates a dual-component synapse (AMPA + NMDA with Mg²⁺ block) from parameters in `AMPANMDA.json`. The `channel_type` argument (`'AMPANMDA'` or `'AMPA'`) selects whether NMDA is included.

### `replay_background_spikes.py` — Replay System: Spike Loading

*(Not uploaded — description from imports and usage)*

- **`resolve_replay_section_synapse_csv(path)`**: Resolves a CLI path to the actual `section_synapse_df.csv` file.
- **`load_replay_csv_and_maps(csv_path)`**: Reads the reference CSV and builds dictionaries mapping synapse identity keys to their recorded spike trains.
- **`row_syn_key(section_row)`**: Generates a unique identity key for a synapse row (based on section name + location + type) for replay matching.

### `replay_layout_from_csv.py` — Replay System: Layout Reconstruction

*(Not uploaded — description from imports and usage)*

- **`populate_section_synapse_df_from_csv(cell, ..., ref_df)`**: Rebuilds the full `section_synapse_df` from a reference CSV, recreating synapse positions without re-running the random placement.
- **`replay_assign_cluster_metadata(cell, ...)`**: Restores cluster assignments (center/surround, cluster_id, pre_unit_id) from the reference run.

### `nmda_detection_utils.py` — NMDA Spike Rate Detection

*(Not uploaded — description from imports and usage)*

- **`batch_nmda_spike_rates_from_seg_v_array(seg_v_array, dt_s, duration, ...)`**: Post-hoc analysis of segment voltage traces to detect NMDA spikes (based on voltage threshold + minimum duration criteria). Returns spike rate in Hz per segment.

### `visualize_utils.py` — Visualization Helpers

*(Not uploaded — description from imports)*

- **`visualize_synapses(section_synapse_df, output_path)`**: Renders synapse locations on the morphology, color-coded by type and cluster membership.

---

## Random Seed System

Three independent seeds control three orthogonal aspects of the simulation:

| Seed | Controls | What changes if you vary it |
|------|----------|---------------------------|
| `syn_pos_seed` | **Spatial structure**: synapse positions, cluster centers, synaptic weights | Different morphological configuration |
| `bg_spike_gen_seed` | **Background dynamics**: pink noise, Poisson spike trains, inhibitory tracking | Different background activity realization |
| `clus_spike_gen_seed` | **Cluster stimulus**: spike time jitter in `generate_vecstim`, preunit activation order (`perm`) | Different stimulus timing |

All three default to the `epoch` value if not explicitly set. They must never be mixed — each seed feeds a dedicated RNG instance with a specific type (see `AGENT.md` §4 for details).

---

## Output Files

Each simulation run produces the following in its output directory:

| File | Content |
|------|---------|
| `simulation_params.json` | Complete parameter record (reproducibility) |
| `section_synapse_df.csv` | Full synapse table with positions, weights, spike trains |
| `preunit assignment.txt` | Cluster → preunit ID mapping |
| `soma_v_array.npy` | Somatic voltage, shape `(T, num_stim, num_aff, num_trials)` |
| `apic_v_array.npy` | Apical nexus voltage |
| `dend_v_array.npy` | Cluster center voltages, shape `(num_clus, T, ...)` |
| `dend_nmda_i_array.npy` | Summed NMDA current at each cluster |
| `dend_ampa_i_array.npy` | Summed AMPA current at each cluster |
| ... | Additional voltage/current arrays (trunk, tuft, background currents) |
| `seg_v_array.npy` | *(if `--with_global_rec`)* All-segment voltages |
| `segment_nmda_spike_rate.npz` | *(if `--with_global_rec`)* Per-segment NMDA spike rate |

---

## Missing Scripts (Not Uploaded)

The following utility modules are referenced by `L5b_simulation.py` but were **not provided** for this documentation. Their descriptions above are inferred from import signatures and usage patterns. To complete the documentation, these files would be needed:

| File | Functions used |
|------|---------------|
| `utils/graph_utils.py` | `create_directed_graph`, `set_graph_order` |
| `utils/distance_utils.py` | `distance_synapse_mark_compare`, `recur_dist_to_soma`, `recur_dist_to_root` |
| `utils/synapses_models.py` | `AMPANMDA` class |
| `utils/replay_background_spikes.py` | `resolve_replay_section_synapse_csv`, `load_replay_csv_and_maps`, `load_replay_spike_maps`, `row_syn_key` |
| `utils/replay_layout_from_csv.py` | `populate_section_synapse_df_from_csv`, `replay_assign_cluster_metadata` |
| `utils/nmda_detection_utils.py` | `batch_nmda_spike_rates_from_seg_v_array`, `DEFAULT_V_THRESH_MV`, `DEFAULT_MIN_DURATION_MS` |
| `utils/visualize_utils.py` | `visualize_synapses` |
| `utils/count_spikes.py` | (analysis utility) |
| `utils/add_single_synapse.py` | (may be legacy — synapse addition is now inline in `L5b_simulation.py`) |
| `utils/tuning_curve_utils.py` | (analysis utility) |

---

## Dependencies

- **NEURON** (with compiled mod mechanisms)
- numpy, scipy, pandas, networkx, matplotlib, tqdm
- Python ≥ 3.9

---

## References

The L5PN model is based on the Hay et al. (2011) detailed biophysical model. Synapse densities (10042 basal exc / 16070 apical exc / ~2800 inh) follow experimental literature estimates for cortical L5 pyramidal neurons.
