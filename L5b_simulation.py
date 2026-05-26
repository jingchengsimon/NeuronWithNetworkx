import argparse
import itertools
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from utils.cell_with_networkx import CellWithNetworkx
from utils.replay_background_spikes import resolve_replay_section_synapse_csv

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "30"))
MAX_PROCESS_COMBINATIONS = 64

# distr add_inputs reads section_synapse_df.csv from the matching clus run (same epoch/path)
SPAT_CONDITION_ORDER = ("clus", "distr")


def ordered_spat_conditions(spat_list: list[str]) -> list[str]:
    """Return spat conditions in dependency order: clus before distr."""
    seen = set(spat_list)
    ordered = [s for s in SPAT_CONDITION_ORDER if s in seen]
    unknown = [s for s in spat_list if s not in SPAT_CONDITION_ORDER]
    if unknown:
        raise ValueError(f"Unknown spat_condition values: {unknown}")
    return ordered


# main function
swc_file_path = './modelFile/cell1.asc'

def create_parser():
    """Create and configure argument parser with default values from utils"""
    parser = argparse.ArgumentParser(description='Neuron simulation parameters')
    
    # Synapse numbers
    parser.add_argument('--num_syn_basal_exc', type=int, default=10042,
                        help='Number of basal excitatory synapses (default: 10042)')
    parser.add_argument('--num_syn_apic_exc', type=int, default=16070,
                        help='Number of apical excitatory synapses (default: 16070)')
    parser.add_argument('--num_syn_basal_inh', type=int, default=1023,
                        help='Number of basal inhibitory synapses (default: 1023)')
    parser.add_argument('--num_syn_apic_inh', type=int, default=1637,
                        help='Number of apical inhibitory synapses (default: 1637)')
    parser.add_argument('--num_syn_soma_inh', type=int, default=150,
                        help='Number of soma inhibitory synapses (default: 150)')
    
    # Simulation duration
    parser.add_argument('--simu_duration', type=int, default=1000,
                        help='Simulation duration in ms (default: 1000)')
    parser.add_argument('--stim_duration', type=int, default=1000,
                        help='Stimulation duration in ms (default: 1000)')
    parser.add_argument('--stim_time', type=int, default=500,
                        help='Time point of stimulation in ms (default: 500)')
    parser.add_argument('--num_stim', type=int, default=1,
                        help='Number of stimuli (default: 1)')
    
    # Channel types
    parser.add_argument('--basal_channel_type', type=str, default='AMPANMDA',
                        choices=['AMPANMDA', 'AMPA'],
                        help='Basal channel type (default: AMPANMDA)')
    parser.add_argument('--bg_exc_channel_type', type=str, default='AMPANMDA',
                        choices=['AMPANMDA', 'AMPA'],
                        help='Background excitatory channel type (default: AMPANMDA)')
    parser.add_argument('--channel_suffix', type=str, default='singclus',
                        help='Base channel suffix for simulation folder name (e.g. singclus, singclus_AMPA). '
                             'With --with_ap / --with_global_rec, _ap / _globrec are appended (default: singclus)')
    
    # Cluster parameters
    parser.add_argument('--cluster_radius', type=float, default=5.0,
                        help='Cluster radius in um (default: 5.0)')
    parser.add_argument('--num_clusters', type=int, default=1,
                        help='Number of clusters (default: 1)')
    parser.add_argument('--num_syn_per_clus', type=int, default=72,
                        help='Number of synapses per cluster (default: 72)')
    parser.add_argument('--num_conn_per_preunit', type=int, default=3,
                        help='Number of connections per preunit (default: 3)')
    
    # Simulation parameters
    parser.add_argument('--simu_condition', type=str, nargs='+', default=['invivo'],
                        choices=['invivo', 'invitro'],
                        help='Simulation condition (default: invivo)')
    parser.add_argument('--spat_condition', type=str, nargs='+', default=['clus'],
                        choices=['clus', 'distr'],
                        help='Spatial condition: clus (clustered) or distr (distributed). '
                             'When both are given, clus always runs before distr (distr replays clus synapse layout). '
                             '(default: clus)')
    parser.add_argument('--sec_type', type=str, nargs='+', default=['basal'],
                        choices=['basal', 'apical'],
                        help='Section type (default: basal)')
    parser.add_argument('--distance_to_root', type=int, nargs='+', default=[0],
                        help='Distance from clusters to root (default: 0)')

    # Simulation modes
    parser.add_argument('--expected', action='store_true', default=False,
                        help='Use expected/linear-sum cluster stimulus logic. '
                             'If set, iter_step is forced to 1 in add_inputs (default: False)')
    parser.add_argument('--aff_mode', type=str, default='linear',
                        choices=['linear', 'curve', 'full', 'custom'],
                        help='Activation mode across preunits: linear uses range(0, N+1, iter_step); '
                             'curve uses dense first then sparse increments; full runs 0 and N preunits; '
                             'custom uses aff_list as the within-run aff axis and auto-adds 0 if omitted '
                             '(default: linear)')
    parser.add_argument('--aff_list', type=int, nargs='+', default=None,
                        help='Custom activated-preunit counts for aff_mode=custom, e.g. --aff_list 4 24 48 72. '
                             'This list is used within each run and does not expand parameter combinations.')
    parser.add_argument('--iter_step', type=int, default=2,
                        help='Step size for aff_mode linear/curve. Ignored by full; forced to 1 when --expected is set (default: 2)')
    parser.add_argument('--epoch_mode', type=str, default='sing',
                        choices=['sing', 'multi'],
                        help='Epoch execution mode: sing runs num_epochs epochs directly; '
                             'multi runs num_batches batches with epochs_per_batch (default: sing)')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs to run when epoch_mode is sing (default: 1)')
    parser.add_argument('--num_batches', type=int, default=1,
                        help='Number of batches when epoch_mode is multi (default: 1)')
    parser.add_argument('--epochs_per_batch', type=int, default=1,
                        help='Epochs per batch when epoch_mode is multi (default: 1)')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='Starting epoch index for epoch execution (default: 1)')
    
    # Background input parameters
    parser.add_argument('--bg_exc_freq', type=float, default=1.0,
                        help='Background excitatory frequency in Hz (default: 1.0)')
    parser.add_argument('--bg_inh_freq', type=float, default=4.0,
                        help='Background inhibitory frequency in Hz (default: 4.0)')
    parser.add_argument('--input_ratio_basal_apic', type=float, default=1.0,
                        help='Input ratio of basal to apical (default: 1.0)')
    
    # Synaptic weight parameters
    parser.add_argument(
        '--initW',
        type=float,
        default=0.0004,
        help='When use_fixedW is off: mean of log-normal excitatory weights (sigma=1, '
             'mu=log(initW)-0.5*sigma^2 so E[weight]=initW). When use_fixedW is on: not used for '
             'those weights (they are fixedW). Units: uS (default: 0.0004).',
    )
    parser.add_argument(
        '--use_fixedW',
        action='store_true',
        default=False,
        help='If set, all excitatory synapse weights that would be log-normal draws use fixedW '
             'instead; initW does not affect those weights. Default: off (log-normal from initW).',
    )
    parser.add_argument(
        '--fixedW',
        type=float,
        default=0.0004,
        help='Excitatory synapse weight (uS) when --use_fixedW is set. Does not change the '
             'log-normal distribution when use_fixedW is off (default: 0.0004).',
    )
    parser.add_argument('--num_func_group', type=int, default=10,
                        help='Number of functional groups (default: 10)')
    parser.add_argument('--inh_delay', type=float, default=4.0,
                        help='Delay of inhibitory inputs in ms (default: 4.0)')
    
    # Other parameters
    parser.add_argument('--num_trials', type=int, default=1,
                        help='Number of trials (default: 1)')
    parser.add_argument('--folder_tag', type=str, default='1',
                        help='Folder tag for output (default: 1)')
    # Random seeds - default to epoch value
    parser.add_argument('--bg_syn_pos_seed', type=int, default=None,
                        help='Random seed for background synapse positioning and background synapse weights. '
                             'If None, uses syn_pos_seed when provided, otherwise epoch value (default: None)')
    parser.add_argument('--clus_syn_pos_seed', type=int, default=None,
                        help='Random seed for cluster assignment and clustered synapse weight overwrite. '
                             'If None, uses syn_pos_seed when provided, otherwise epoch value (default: None)')
    parser.add_argument('--syn_pos_seed', type=int, default=None,
                        help='Deprecated shared synapse-position seed. Used as fallback for bg_syn_pos_seed '
                             'and clus_syn_pos_seed when those are not set (default: None)')
    parser.add_argument('--bg_spike_gen_seed', type=int, default=None,
                        help='Random seed for background (bg) spike generation (bg spike trains, pink noise). If None, uses epoch value (default: None)')
    parser.add_argument('--clus_spike_gen_seed', type=int, default=None,
                        help='Random seed for cluster stimulus spike generation (stim times in generate_vecstim, '
                             'preunit permutation). If None, uses epoch value. Distinct from bg_spike_gen_seed (bg only) (default: None)')
    parser.add_argument(
        '--use_replay_bg',
        action='store_true',
        default=False,
        help='If set, load synapse layout, cluster metadata, and background spikes from replay_bg_csv '
             '(reference run). If not set, normal simulation: random synapses and generated background.',
    )
    parser.add_argument(
        '--replay_bg_csv',
        type=str,
        default='/G/results/simulation_singclus_Oct25/basal_range0_clus_invivo_REAL/1/8/section_synapse_df.csv',
        help='Reference path used only when --use_replay_bg is set: section_synapse_df.csv '
             'or a run directory (uses section_synapse_df.csv inside). Spikes matched by syn identity.',
    )
    
    # Biophysics model selection
    # Default is False (use L5PCbiophys3.hoc without AP and Ca)
    # Use --with_ap to enable L5PCbiophys3withNaCa.hoc (with AP and Ca dynamics)
    parser.add_argument('--with_ap', action='store_true', default=False,
                        help='Use L5PCbiophys3withNaCa.hoc (with AP and Ca dynamics). '
                             'Default is False (uses L5PCbiophys3.hoc without AP and Ca)')
    # Global segment recording: seg_ina and seg_inmda for all segments, saved as npy
    parser.add_argument('--with_global_rec', action='store_true', default=False,
                        help='Record seg_ina and seg_inmda for all segments and save to npy. Default is False')
    
    return parser

def build_cell(args):
    """Build and simulate cell with parameters from argparse"""
    
    # Extract parameters from args using getattr (more concise than individual assignments)
    def get_param(name): return getattr(args, name)
    
    # Get epoch first for separator, then extract all parameters
    epoch = get_param('epoch')
    print('\n' + '='*80)
    print(f'EPOCH {epoch}')
    print('='*80 + '\n')
    
    # Core parameters
    NUM_SYN_BASAL_EXC = get_param('num_syn_basal_exc')
    NUM_SYN_APIC_EXC = get_param('num_syn_apic_exc')
    NUM_SYN_BASAL_INH = get_param('num_syn_basal_inh')
    NUM_SYN_APIC_INH = get_param('num_syn_apic_inh')
    NUM_SYN_SOMA_INH = get_param('num_syn_soma_inh')
    SIMU_DURATION = get_param('simu_duration')
    STIM_DURATION = get_param('stim_duration')

    simu_condition = get_param('simu_condition')
    spat_condtion = get_param('spat_condition')
    basal_channel_type = get_param('basal_channel_type')
    bg_exc_channel_type = get_param('bg_exc_channel_type')
    sec_type = get_param('sec_type')
    distance_to_root = get_param('distance_to_root')

    num_clusters = get_param('num_clusters')
    cluster_radius = get_param('cluster_radius')
    num_conn_per_preunit = get_param('num_conn_per_preunit')
    num_syn_per_clus = get_param('num_syn_per_clus')

    bg_exc_freq = get_param('bg_exc_freq')
    bg_inh_freq = get_param('bg_inh_freq')
    input_ratio_basal_apic = get_param('input_ratio_basal_apic')

    initW = get_param('initW')
    use_fixedW = get_param('use_fixedW')
    fixedW = get_param('fixedW')
    num_func_group = get_param('num_func_group')
    inh_delay = get_param('inh_delay')

    num_stim = get_param('num_stim')
    stim_time = get_param('stim_time')
    num_trials = get_param('num_trials')
    folder_tag = get_param('folder_tag')

    channel_suffix_arg = get_param('channel_suffix')
    with_ap = get_param('with_ap')
    with_global_rec = get_param('with_global_rec')
    expected = get_param('expected')
    aff_mode = get_param('aff_mode')
    aff_list = get_param('aff_list')
    iter_step = get_param('iter_step')
    epoch_mode = get_param('epoch_mode')
    num_epochs = get_param('num_epochs')
    num_batches = get_param('num_batches')
    epochs_per_batch = get_param('epochs_per_batch')
    start_epoch = get_param('start_epoch')
    use_replay_bg = get_param('use_replay_bg')
    replay_bg_csv_arg = get_param('replay_bg_csv')
    
    # Random seeds: default to epoch if not set
    # bg_syn_pos_seed: synapse locations and background synapse weights
    # clus_syn_pos_seed: cluster assignment and clustered synapse weight overwrite
    # bg_spike_gen_seed: background (bg) spike generation (bg spike trains, pink noise)
    # clus_spike_gen_seed: cluster stimulus (stim times in generate_vecstim, preunit permutation)
    bg_syn_pos_seed_arg = get_param('bg_syn_pos_seed')
    clus_syn_pos_seed_arg = get_param('clus_syn_pos_seed')
    syn_pos_seed_arg = get_param('syn_pos_seed')
    bg_spike_gen_seed_arg = get_param('bg_spike_gen_seed')
    clus_spike_gen_seed_arg = get_param('clus_spike_gen_seed')

    bg_syn_pos_seed = bg_syn_pos_seed_arg if bg_syn_pos_seed_arg is not None else (
        syn_pos_seed_arg if syn_pos_seed_arg is not None else epoch
    )
    clus_syn_pos_seed = clus_syn_pos_seed_arg if clus_syn_pos_seed_arg is not None else (
        syn_pos_seed_arg if syn_pos_seed_arg is not None else epoch
    )
    bg_spike_gen_seed = bg_spike_gen_seed_arg if bg_spike_gen_seed_arg is not None else epoch
    clus_spike_gen_seed = clus_spike_gen_seed_arg if clus_spike_gen_seed_arg is not None else epoch

    if use_replay_bg:
        replay_bg_csv = resolve_replay_section_synapse_csv(replay_bg_csv_arg)
    else:
        replay_bg_csv = None
           
    # Build channel_suffix: ensure leading underscore, then append conditional suffixes
    channel_suffix = channel_suffix_arg.strip()
    channel_suffix = ('_' + channel_suffix) if channel_suffix and not channel_suffix.startswith('_') else channel_suffix
    channel_suffix += ''.join(['_ap' if with_ap else '', '_globrec' if with_global_rec else ''])
    simu_folder = f'{sec_type}_range{distance_to_root}_{spat_condtion}_{simu_condition}{channel_suffix}'
    if use_fixedW:
        w_tag = format(fixedW, '.10g').replace('-', 'neg')
        simu_folder = f'{simu_folder}_fixedW{w_tag}'

    # Normalize folder tag
    folder_tag = str(int(folder_tag) % 100) if int(folder_tag) % 100 != 0 else '100'
    expected_suffix = '_expected' if expected else ''
    folder_path = f'/G/results/simulation_singclus_supple_May26/{simu_folder}{expected_suffix}/{folder_tag}/{epoch}'
    # folder_path = Path('/G/results/simulation_multiclus_Oct25') / simu_folder / folder_tag / str(epoch)

    simulation_params = {
        'cell model': 'L5PN',
        'NUM_SYN_BASAL_EXC': NUM_SYN_BASAL_EXC, 'NUM_SYN_APIC_EXC': NUM_SYN_APIC_EXC,
        'NUM_SYN_BASAL_INH': NUM_SYN_BASAL_INH, 'NUM_SYN_APIC_INH': NUM_SYN_APIC_INH,
        'NUM_SYN_SOMA_INH': NUM_SYN_SOMA_INH, 'SIMU DURATION': SIMU_DURATION,
        'STIM DURATION': STIM_DURATION, 'simulation condition': simu_condition,
        'synaptic spatial condition': spat_condtion, 'basal channel type': basal_channel_type,
        'channel_suffix': channel_suffix_arg, 'section type': sec_type,
        'distance from clusters to root': distance_to_root, 'number of clusters': num_clusters,
        'cluster radius': cluster_radius, 'background excitatory frequency': bg_exc_freq,
        'background inhibitory frequency': bg_inh_freq, 'input ratio of basal to apical': input_ratio_basal_apic,
        'background excitatory channel type': bg_exc_channel_type, 'initial weight of AMPANMDA synapses': initW,
        'use_fixedW': use_fixedW, 'fixedW': fixedW,
        'number of functional groups': num_func_group, 'delay of inhibitory inputs': inh_delay,
        'number of stimuli': num_stim, 'time point of stimulation': stim_time,
        'number of connection per preunit': num_conn_per_preunit, 'number of synapses per cluster': num_syn_per_clus,
        'number of trials': num_trials, 'syn_pos_seed': bg_syn_pos_seed,
        'bg_syn_pos_seed': bg_syn_pos_seed, 'clus_syn_pos_seed': clus_syn_pos_seed,
        'bg_spike_gen_seed': bg_spike_gen_seed, 'clus_spike_gen_seed': clus_spike_gen_seed,
        'expected': expected, 'aff_mode': aff_mode, 'aff_list': aff_list, 'iter_step': iter_step,
        'effective_iter_step': 1 if expected else iter_step,
        'epoch_mode': epoch_mode,
        'num_epochs': num_epochs,
        'num_batches': num_batches,
        'epochs_per_batch': epochs_per_batch,
        'start_epoch': start_epoch,
        'with_ap': with_ap, 'with_global_rec': with_global_rec,
        'use_replay_bg': use_replay_bg,
        'replay_bg_csv': replay_bg_csv,
        'segment_nmda_spike_rate_npz': 'segment_nmda_spike_rate.npz' if with_global_rec else None,
    }

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    json_filename = os.path.join(folder_path, "simulation_params.json")
    with open(json_filename, 'w') as json_file:
        json.dump(simulation_params, json_file, indent=4)

    cell1 = CellWithNetworkx(swc_file_path, bg_exc_freq, bg_inh_freq, SIMU_DURATION, STIM_DURATION, 
                            bg_syn_pos_seed, bg_spike_gen_seed, clus_spike_gen_seed, with_ap, with_global_rec,
                            replay_bg_csv=replay_bg_csv, clus_syn_pos_seed=clus_syn_pos_seed)
    cell1.add_synapses(NUM_SYN_BASAL_EXC, NUM_SYN_APIC_EXC, NUM_SYN_BASAL_INH, NUM_SYN_APIC_INH, NUM_SYN_SOMA_INH)
    
    cell1.assign_clustered_synapses(basal_channel_type, sec_type, distance_to_root, 
                                    num_clusters, cluster_radius, num_stim, stim_time, 
                                    spat_condtion, num_conn_per_preunit, num_syn_per_clus, folder_path) 

    cell1.add_inputs(folder_path, simu_condition, input_ratio_basal_apic, 
                     bg_exc_channel_type, initW, num_func_group, inh_delay, num_trials,
                     use_fixedW=use_fixedW, fixedW=fixedW,
                     expected=expected, aff_mode=aff_mode, aff_list=aff_list, iter_step=iter_step)

def run_processes(args_list, epoch):
    """Run multiple processes with different parameter sets"""
    processes = []  # Create a new process list for each set of parameters
    for args in args_list:
        # Create a copy of args and set epoch
        args_copy = argparse.Namespace(**vars(args))
        args_copy.epoch = epoch
        process = multiprocessing.Process(target=build_cell, args=(args_copy,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()  # Join each batch of processes before moving to the next parameter set

def run_combination(combination_args):
    """
    Run combination of parameters.
    """
    simu_condition, spat_cond, sec_type, dis_to_root, epoch, base_args = combination_args

    # Create a copy of base_args (which contains command-line arguments)
    args = argparse.Namespace(**vars(base_args))

    # Override with scalar values from the current combination.
    args.simu_condition = simu_condition
    args.spat_condition = spat_cond
    args.sec_type = sec_type
    args.distance_to_root = dis_to_root

    run_processes([args], epoch)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()  # Parse command-line arguments once

    if args.epoch_mode == 'sing':
        if args.num_epochs <= 0:
            raise ValueError('num_epochs must be positive when epoch_mode is sing')
        epoch_ranges = [
            range(args.start_epoch, args.start_epoch + args.num_epochs)
        ]
    else:
        if args.num_batches <= 0:
            raise ValueError('num_batches must be positive when epoch_mode is multi')
        if args.epochs_per_batch <= 0:
            raise ValueError('epochs_per_batch must be positive when epoch_mode is multi')

        epoch_ranges = []
        for batch_idx in range(args.num_batches):
            start_epoch = args.start_epoch + batch_idx * args.epochs_per_batch
            end_epoch = start_epoch + args.epochs_per_batch
            epoch_ranges.append(range(start_epoch, end_epoch))

    spat_conditions = ordered_spat_conditions(args.spat_condition)
    combos_per_epoch = (
        len(args.simu_condition)
        * len(args.sec_type)
        * len(args.distance_to_root)
    )
    if combos_per_epoch > MAX_PROCESS_COMBINATIONS:
        raise ValueError(
            f'Parameter combinations per epoch/spat_condition exceed CPU core limit: '
            f'{combos_per_epoch} > {MAX_PROCESS_COMBINATIONS}. '
            f'Reduce simu_condition/sec_type/distance_to_root values.'
        )

    for epoch_range in epoch_ranges:
        tasks_per_spat = len(epoch_range) * combos_per_epoch
        if tasks_per_spat > MAX_PROCESS_COMBINATIONS:
            raise ValueError(
                f'Parameter combinations per epoch_range/spat_condition exceed CPU core limit: '
                f'{tasks_per_spat} = {len(epoch_range)} epochs * {combos_per_epoch} combinations '
                f'> {MAX_PROCESS_COMBINATIONS}. '
                f'Reduce epochs_per_batch/num_epochs or parameter combinations.'
            )

        for spat_cond in spat_conditions:
            combinations = [
                (simu_condition, spat_cond, sec_type, dis_to_root, epoch, args)
                for epoch in epoch_range
                for simu_condition, sec_type, dis_to_root in itertools.product(
                    args.simu_condition,
                    args.sec_type,
                    args.distance_to_root,
                )
            ]
            print(
                f'Running epochs={epoch_range.start}..{epoch_range.stop - 1} '
                f'spat_condition={spat_cond} '
                f'({len(combinations)} combinations in parallel)'
            )
            with ProcessPoolExecutor(max_workers=len(combinations)) as executor:
                executor.map(run_combination, combinations)


    # multiprocessing.set_start_method('spawn', force=True) # Use spawn will initiate too many NEURON instances 

    # # Running for multi-cluster analysis
    # parser = create_parser()
    # for sec_type in ['basal']: # ['basal', 'apical']
    #     for spat_cond in ['clus', 'distr']: # ['clus', 'distr']
    #         for dis_to_root in [0, 2]: # [0, 1, 2]
    #             args_list = []
    #             base_args = parser.parse_args([])
    #             base_args.sec_type = sec_type
    #             base_args.spat_condition = spat_cond
    #             base_args.distance_to_root = dis_to_root
    #             args_list.append(base_args)
    #             for epoch in range(1, 6):
    #                 run_processes(args_list, epoch)

    #   python L5b_simulation.py \
    #     --with_ap \
    #     --aff_mode custom \
    #     --aff_list 4 8 12 24 48 72 \
    #     --simu_condition invivo \
    #     --sec_type basal apical \
    #     --distance_to_root 0 1 2 \
    #     --spat_condition clus distr \
    #     --epoch_mode multi \
    #     --num_batches 10 \
    #     --epochs_per_batch 10 \
    #     --start_epoch 1 \
    #     --bg_exc_freq 1.3

