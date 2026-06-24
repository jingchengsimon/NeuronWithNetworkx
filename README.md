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
├── model/
│   ├── cell1.asc / cell1.swc      # Reconstructed L5PN morphology
│   ├── L5PCtemplate.hoc            # NEURON cell template (soma, basal, apical)
│   ├── L5PCbiophys3.hoc            # Passive biophysics (no AP)
│   ├── L5PCbiophys3withNaCa.hoc    # Active biophysics (with AP + Ca²⁺)
│   ├── AMPANMDA.json               # AMPA+NMDA synapse parameters
│   └── PN2PN.json                  # (Reserved) pyramidal-to-pyramidal params
│
├── mod/                            # Tracked NEURON mechanism sources (.mod)
├── arm64/ and x86_64/              # Local compiled mechanisms (Git-ignored)
│
├── utils/                          # Core simulation modules
│   └── l5pn_model.py               # L5PNModel implementation
├── analysis/                       # Post-hoc analysis and figure generation
│   ├── nmda_spike_detection.py
│   ├── ap_ca_spike_analysis.py
│   ├── trace_analysis.py
│   ├── variability_analysis.py
│   ├── figures/
│   ├── diagnostics/
│   ├── morphology/
│   └── notebooks/                  # Local notebooks; ignored by Git
├── scripts/
│   └── test_seed_reproducibility.py
├── results/                        # Local generated output; ignored by Git
│
├── docs/                           # Technical documentation
│   ├── CONVENTIONS.md              # Magic constants, abbreviations, schema reference
│   └── ARCHITECTURE.md             # Module dependency graph, data flow diagrams
│
├── AGENT.md                        # AI agent constraint specification
└── pyproject.toml                  # Project and tooling configuration
```

---

## Simulation Pipeline

`utils/l5pn_model.py` contains the `L5PNModel` class, which executes the following pipeline. `L5b_simulation.py` is the CLI and parallel-scheduling entry point. **The order of steps 4a → 4b → 4c is immutable** — inhibitory inputs track excitatory activity and must be generated last.

```
Step 1: __init__()
   Load morphology (cell1.asc) → build networkx DiG → compute branch orders

Step 2: initialize_synapse_layout()
   Distribute ~26k excitatory + ~2.8k inhibitory synapses across the dendritic tree
   (length-weighted random placement)

Step 3: assign_synapse_clusters()
   Select cluster center(s) within a distance zone → recruit surround synapses
   via exponential distance distribution → assign preunit-to-cluster mapping

Step 4: run_stimulation_protocol()
   4a. add_background_exc_inputs()
       Generate pink noise → modulate Poisson spike trains → 50% dropout
       Create AMPA+NMDA synapses with log-normal weights

   4b. add_clustered_inputs()  [loop over activation levels]
       Inject cluster stimulus (VecStim) → merge with background spikes

   4c. add_background_inh_inputs()  [loop]
       Track total excitatory activity → generate coupled inhibitory spikes

   4d. _run_single_trial()  [loop]
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
- **Thread level** (`--max_workers_synapse`, default 30): within each process, `initialize_synapse_layout()` and background input prep use `ThreadPoolExecutor`; `h.run()` always runs on the main thread (NEURON is not thread-safe).
- **No batch / epoch_mode CLI** — epoch range is only `--num_epochs` + `--start_epoch`.
- **Do not** mix `clus` and `distr` in the same process pool; distr replay depends on clus `section_synapse_df.csv`.

Batch var experiments: see `run_var_exp.sh` (single invocation with `--spat_cond clus distr --num_epochs 100`).

---

## Utility Modules (`utils/`)

### `l5pn_model.py` — Cell Class and Simulation Runner

Defines `L5PNModel`: morphology loading, graph construction, synapse placement, cluster assignment, input orchestration, `h.run()`, and output serialization.

### `synaptic_inputs.py` — Synapse Creation & Spike Train Injection

The largest utility module. Three public functions implement the three input pathways:

- **`add_background_exc_inputs()`**: For each excitatory synapse, selects a random pink noise trace, generates a Poisson spike train modulated by the noise envelope, applies 50% presynaptic failure, and creates an AMPA+NMDA synapse object with a log-normal weight. Each synapse gets a deterministic per-index RNG seed to ensure reproducibility under multithreading.

- **`add_background_inh_inputs()`**: Computes the total excitatory spike histogram (background + clustered), then generates inhibitory Poisson spikes whose rate tracks the excitatory activity (λ_inh ∝ total_exc_spikes / mean). This E/I tracking produces balanced, in vivo–like dynamics.

- **`add_clustered_inputs()`**: For each cluster, looks up which preunits are active, retrieves their stimulus spike times, merges with existing background spikes (unique time points only), and attaches a VecStim. Old NetCon weights are zeroed before creating new connections.

Also contains replay variants (`_add_background_exc_inputs_replay`, `_add_background_inh_inputs_replay`) that read spike trains from a reference CSV instead of generating new ones.

### `pink_noise.py` — 1/f Noise Generator

Produces temporally correlated (pink / 1/f) noise traces using an IIR filter applied to white Gaussian noise. The output is z-scored and the first 2000 samples are discarded as burn-in.

- **`make_noise(num_traces, num_samples, spike_gen_seed, scale)`**: Returns an array of shape `(num_traces, num_samples)`. The `scale` parameter controls the white noise standard deviation (default 0.5 for background activity).

The pink noise is **not replaceable with white noise** — its temporal autocorrelation structure is essential for producing realistic in vivo–like firing patterns.

### `cluster_protocol.py` — Cluster Protocol & Preunit Mapping

Two key functions:

- **`generate_indices(rnd, num_clusters, num_conn_per_preunit, num_preunit)`**: Assigns preunits to clusters using a round-robin strategy that equalizes connection counts across clusters. Returns `indices[i]` = list of preunit IDs connected to cluster `i`.

- **`generate_vecstim(rnd, pre_unit_ids, num_stim, stim_time, stim_time_var=5)`**: For each preunit, generates `num_stim` spike times drawn from `Normal(stim_time, stim_time_var)` and floored to integer ms. The 5 ms jitter corresponds to typical cortical synchrony timescales.

### `random_streams.py` — Random-Seed Ownership

Centralizes CLI seed fallback, RNG-engine selection, deterministic per-synapse seed derivation, and named streams for placement, cluster assignment, activation order, spike timing, weights, and pink noise. Simulation modules do not construct production RNGs directly.

### `morphology_graph.py` — Morphology → Directed Graph

- **`create_directed_graph(all_sections, all_segments, section_df, return_segment_graph=False)`**: Builds a networkx `DiGraph` where each node is a section and edges point from parent to child. Also populates `section_df` with section metadata (parent, length, type, branch index). When `return_segment_graph=True`, it also returns `segment_df` and `segment_DiG`, where segment nodes are connected in series inside each section and across parent/child section boundaries.

- **`set_graph_order(DiG, root_tuft_idx)`**: Computes branch order from two roots: soma (for the full tree) and tuft root (for the apical tuft subtree). Returns classification dictionaries mapping order → section indices.

### `cable_distance.py` — Cable Distance Calculation

- **`recur_dist_to_soma(section, loc)`**: Recursively computes cable distance (µm) from a point on a section to the soma.
- **`recur_dist_to_root(section, loc, root_sec)`**: Cable distance to an arbitrary root section (used for tuft root).
- **`distance_synapse_mark_compare(distances, marks)`**: Given actual synapse-to-center distances and target distance marks (from exponential distribution), returns indices of synapses that fall within each mark — used for cluster member selection.

### `synapse_models.py` — AMPA+NMDA Synapse Wrapper

- **`AMPANMDA(syn_params, loc, section, channel_type)`**: Creates a dual-component synapse (AMPA + NMDA with Mg²⁺ block) from parameters in `AMPANMDA.json`. The `channel_type` argument (`'AMPANMDA'` or `'AMPA'`) selects whether NMDA is included.

### `nmda_spike_detection.py` — NMDA Spike Rate Detection

- **`batch_nmda_spike_rates_from_seg_v_array(seg_v_array, dt_s, duration, ...)`**: Post-hoc analysis of segment voltage traces to detect NMDA spikes (based on voltage threshold + minimum duration criteria). Returns spike rate in Hz per segment.

---

## Random Seed System

Four independent seeds control separate spatial and temporal aspects of the simulation:

| Seed | Controls | What changes if you vary it |
|------|----------|---------------------------|
| `bg_syn_pos_seed` | All synapse locations and invivo excitatory weights | Different base synapse realization |
| `clus_syn_pos_seed` | Cluster assignment, preunit mapping/order, and invitro clustered weights | Different clustered connectivity realization |
| `bg_spike_gen_seed` | **Background dynamics**: pink noise, Poisson spike trains, inhibitory tracking | Different background activity realization |
| `clus_spike_gen_seed` | Clustered presynaptic spike-time jitter in `generate_vecstim` | Different clustered stimulus timing |

All four default to the `epoch` value if not explicitly set. The deprecated `syn_pos_seed` is only a fallback for the two position seeds. RNG construction and deterministic child-seed derivation are centralized in `utils/random_streams.py`.

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

## Dependencies

- **NEURON** (with compiled mod mechanisms)
- numpy, scipy, pandas, networkx, matplotlib, tqdm
- Python ≥ 3.9

---

## References

The L5PN model is based on the Hay et al. (2011) detailed biophysical model. Synapse densities (10042 basal exc / 16070 apical exc / ~2800 inh) follow experimental literature estimates for cortical L5 pyramidal neurons.
