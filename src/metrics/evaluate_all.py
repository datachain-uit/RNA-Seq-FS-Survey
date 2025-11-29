from .classification import compute_classification_metrics
from .energy_carbon import compute_energy, compute_edp, compute_carbon
from .uncertainty import calc_sent
from .drift import calc_sdrift
from .leak import calc_sleak
from .snan import calc_snan

def evaluate_all_metrics(
    y_true, y_pred, y_prob=None,
    cpu_util=0.0, wall_time=0.0,
    peak_mem_mb=0.0, base_mem_mb=0.0,
    num_evals=0, missing_flags=None, labels=None,
    cache_hit=None
):
    cls = compute_classification_metrics(y_true, y_pred, y_prob)

    energy = compute_energy(cpu_util, wall_time)
    edp = compute_edp(energy, wall_time)
    carbon = compute_carbon(energy)

    mem_overhead = peak_mem_mb / base_mem_mb if base_mem_mb > 0 else None

    return {
        **cls,
        "WallTime(s)": wall_time,
        "CPUUtil(%)": cpu_util,
        "PeakMem(MB)": peak_mem_mb,
        "MemoryOverhead": mem_overhead,
        "CacheHit": cache_hit,
        "NumEval": num_evals,
        "Energy(J)": energy,
        "EDP": edp,
        "Carbon(gCO2e)": carbon,
        "sent": calc_sent(y_prob),
        "sdrift": calc_sdrift(y_prob, y_true),
        "sleak": calc_sleak(missing_flags, labels),
        "snan": calc_snan(y_prob),
    }
