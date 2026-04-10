"""
NMDA spike detection on 1D membrane potential traces.

Criterion: V must exceed a threshold continuously for at least a minimum duration.
Each contiguous supra-threshold run (same as scipy.ndimage.label on 1D binary mask)
with length >= min_samples counts as one event.

Accelerated path: Numba (parallel batch). Fallback: pure Python (same algorithm).
"""
import numpy as np

DEFAULT_V_THRESH_MV = -40.0
DEFAULT_MIN_DURATION_MS = 26.0

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def _count_nmda_spikes_runlen(v, v_thresh, min_samples):
    """Pure Python; identical rule to 1D label + length filter."""
    n = v.shape[0]
    count = 0
    i = 0
    while i < n:
        if v[i] <= v_thresh:
            i += 1
            continue
        j = i + 1
        while j < n and v[j] > v_thresh:
            j += 1
        run_len = j - i
        if run_len >= min_samples:
            count += 1
        i = j
    return count


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _count_nmda_spikes_numba(v, v_thresh, min_samples):
        n = v.shape[0]
        count = 0
        i = 0
        while i < n:
            if v[i] <= v_thresh:
                i += 1
                continue
            j = i + 1
            while j < n and v[j] > v_thresh:
                j += 1
            run_len = j - i
            if run_len >= min_samples:
                count += 1
            i = j
        return count

    @njit(parallel=True, cache=True)
    def _batch_nmda_spike_rates_numba(v_2d, v_thresh, min_samples, sim_duration_ms):
        n_traces, n_t = v_2d.shape
        out = np.empty(n_traces, dtype=np.float64)
        dur_s = sim_duration_ms / 1000.0
        inv_dur = 1.0 / dur_s if dur_s > 0 else 0.0
        for k in prange(n_traces):
            c = 0
            i = 0
            v = v_2d[k]
            while i < n_t:
                if v[i] <= v_thresh:
                    i += 1
                    continue
                j = i + 1
                while j < n_t and v[j] > v_thresh:
                    j += 1
                run_len = j - i
                if run_len >= min_samples:
                    c += 1
                i = j
            out[k] = c * inv_dur
        return out


def count_nmda_spikes(v_mV, dt_s, v_thresh_mV=DEFAULT_V_THRESH_MV, min_duration_ms=DEFAULT_MIN_DURATION_MS):
    """
    Count NMDA spike events: each event is one connected run of V > v_thresh_mV
    with duration >= min_duration_ms.
    """
    v = np.asarray(v_mV, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return 0
    min_samples = max(1, int(np.ceil((min_duration_ms / 1000.0) / dt_s)))
    if NUMBA_AVAILABLE:
        return int(_count_nmda_spikes_numba(v, v_thresh_mV, min_samples))
    return _count_nmda_spikes_runlen(v, v_thresh_mV, min_samples)


def compute_nmda_spike_rate_hz(
    v_mV,
    dt_s,
    sim_duration_ms,
    v_thresh_mV=DEFAULT_V_THRESH_MV,
    min_duration_ms=DEFAULT_MIN_DURATION_MS,
):
    """
    NMDA spike rate = (number of detected events) / (simulation duration in seconds).
    """
    dur_s = sim_duration_ms / 1000.0
    if dur_s <= 0:
        return 0.0
    n = count_nmda_spikes(v_mV, dt_s, v_thresh_mV, min_duration_ms)
    return float(n) / dur_s


def batch_nmda_spike_rates_from_seg_v_array(
    seg_v_array,
    dt_s,
    sim_duration_ms,
    v_thresh_mV=DEFAULT_V_THRESH_MV,
    min_duration_ms=DEFAULT_MIN_DURATION_MS,
):
    """
    Vectorized NMDA spike rate (Hz) for all traces in seg_v_array
    shape (n_seg, n_t, n_stim, n_aff, n_trials). Returns same shape without time axis.
    Uses Numba parallel over traces when available.
    """
    seg = np.asarray(seg_v_array, dtype=np.float64)
    if seg.ndim != 5:
        raise ValueError('seg_v_array must have shape (n_seg, n_t, n_stim, n_aff, n_trials)')
    n_seg, n_t, n_stim, n_aff, n_trials = seg.shape
    if n_t < 2:
        return np.zeros((n_seg, n_stim, n_aff, n_trials), dtype=np.float64)
    min_samples = max(1, int(np.ceil((min_duration_ms / 1000.0) / dt_s)))
    v_2d = np.ascontiguousarray(seg.reshape(n_seg * n_stim * n_aff * n_trials, n_t))
    if NUMBA_AVAILABLE:
        flat = _batch_nmda_spike_rates_numba(v_2d, v_thresh_mV, min_samples, sim_duration_ms)
    else:
        flat = np.empty(v_2d.shape[0], dtype=np.float64)
        dur_s = sim_duration_ms / 1000.0
        inv = 1.0 / dur_s if dur_s > 0 else 0.0
        for k in range(v_2d.shape[0]):
            flat[k] = _count_nmda_spikes_runlen(v_2d[k], v_thresh_mV, min_samples) * inv
    return flat.reshape(n_seg, n_stim, n_aff, n_trials)
