"""Central random-seed and RNG policy for the simulation workflow.

Each function represents one owned random process.  Keep the RNG engines and
seed-derivation formulas stable: changing ``RandomState`` to ``default_rng``
or sharing a mutable RNG between processes can change published simulations.
"""

from dataclasses import dataclass

import numpy as np


SEED_FIELDS = (
    ('bg_syn_pos_seed', 'bpos'),
    ('clus_syn_pos_seed', 'cpos'),
    ('bg_spike_gen_seed', 'bspk'),
    ('clus_spike_gen_seed', 'cspk'),
)


@dataclass(frozen=True)
class WorkflowSeeds:
    """Resolved seeds used by one simulation run."""

    bg_syn_pos: int
    clus_syn_pos: int
    bg_spike_gen: int
    clus_spike_gen: int


def resolve_workflow_seeds(
    epoch,
    bg_syn_pos_seed=None,
    clus_syn_pos_seed=None,
    bg_spike_gen_seed=None,
    clus_spike_gen_seed=None,
    legacy_syn_pos_seed=None,
):
    """Resolve CLI seed fallbacks without coupling spatial and temporal seeds."""
    return WorkflowSeeds(
        bg_syn_pos=(
            bg_syn_pos_seed
            if bg_syn_pos_seed is not None
            else legacy_syn_pos_seed if legacy_syn_pos_seed is not None else epoch
        ),
        clus_syn_pos=(
            clus_syn_pos_seed
            if clus_syn_pos_seed is not None
            else legacy_syn_pos_seed if legacy_syn_pos_seed is not None else epoch
        ),
        bg_spike_gen=bg_spike_gen_seed if bg_spike_gen_seed is not None else epoch,
        clus_spike_gen=clus_spike_gen_seed if clus_spike_gen_seed is not None else epoch,
    )


def synapse_placement_seed(bg_syn_pos_seed, region, synapse_type, index):
    """Derive a deterministic placement seed independent of thread scheduling."""
    region_code = {"basal": 11, "apical": 22, "soma": 33}[region]
    type_code = 1 if synapse_type == "exc" else 2
    return (
        bg_syn_pos_seed * 1_000_003
        + region_code * 10_000
        + type_code * 1_000
        + index * 100_003
    ) % (2**31)


def synapse_placement_rng(bg_syn_pos_seed, region, synapse_type, index):
    return np.random.default_rng(
        synapse_placement_seed(bg_syn_pos_seed, region, synapse_type, index)
    )


def cluster_assignment_rng(clus_syn_pos_seed):
    """Cluster/preunit assignment and physical clustered-synapse selection."""
    return np.random.RandomState(clus_syn_pos_seed)


def preunit_activation_rng(clus_syn_pos_seed):
    """Preunit activation order; intentionally restarts from the position seed."""
    return np.random.RandomState(clus_syn_pos_seed)


def cluster_spike_rng(clus_spike_gen_seed):
    """Presynaptic clustered-stimulus timing."""
    return np.random.RandomState(clus_spike_gen_seed)


def excitatory_weight_rng(syn_pos_seed):
    """Excitatory weight draws for invivo background or invitro clusters."""
    return np.random.default_rng(syn_pos_seed)


def background_exc_synapse_seed(bg_spike_gen_seed, index):
    """Derive one background-spike seed per excitatory synapse worker."""
    return (bg_spike_gen_seed * 1_000_003 + index * 100_003) % (2**31)


def background_exc_synapse_rng(bg_spike_gen_seed, index):
    return np.random.default_rng(
        background_exc_synapse_seed(bg_spike_gen_seed, index)
    )


def background_inhibitory_spike_rng(bg_spike_gen_seed):
    return np.random.default_rng(bg_spike_gen_seed)


def pink_noise_rng(bg_spike_gen_seed):
    """Legacy MT19937 stream matching ``np.random.seed`` + module draws."""
    return np.random.RandomState(bg_spike_gen_seed)
