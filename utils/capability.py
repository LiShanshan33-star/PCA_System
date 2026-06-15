import numpy as np
import math
from scipy.stats import norm


def get_c4(df_val):
    """计算无偏修正因子 c4"""
    n = df_val + 1
    if n <= 1:
        return 1.0
    return np.sqrt(2 / (n - 1)) * (math.gamma(n / 2) / math.gamma((n - 1) / 2))


def calculate_within_std(data, subgroup_size):
    """计算组内标准差（合并方差 + c4 修正）"""
    n_samples = len(data)

    if subgroup_size > 1:
        k = n_samples // subgroup_size
        truncated = data[:k * subgroup_size]
        matrix = truncated.reshape(k, subgroup_size)
        variances = np.var(matrix, axis=1, ddof=1)
        sp = np.sqrt(np.mean(variances))
        df_pool = k * (subgroup_size - 1)
        return sp / get_c4(df_pool)

    mr_bar = np.mean(np.abs(np.diff(data)))
    return mr_bar / 1.128


def calculate_capability(data, lsl, usl, subgroup_size, target=None):
    """计算正态过程能力指标，支持单边规格（lsl/usl 可为 None）"""
    mean = np.mean(data)
    std_overall = np.std(data, ddof=1)
    std_within = calculate_within_std(data, subgroup_size)

    has_lsl = lsl is not None
    has_usl = usl is not None

    # 短期能力（组内）
    cp = None
    if has_lsl and has_usl and std_within > 0:
        cp = (usl - lsl) / (6 * std_within)
    cpu = (usl - mean) / (3 * std_within) if has_usl and std_within > 0 else None
    cpl = (mean - lsl) / (3 * std_within) if has_lsl and std_within > 0 else None
    if cpu is not None and cpl is not None:
        cpk = min(cpu, cpl)
    elif cpu is not None:
        cpk = cpu
    elif cpl is not None:
        cpk = cpl
    else:
        cpk = None

    # 长期能力（整体）
    pp = None
    if has_lsl and has_usl and std_overall > 0:
        pp = (usl - lsl) / (6 * std_overall)
    ppu = (usl - mean) / (3 * std_overall) if has_usl and std_overall > 0 else None
    ppl = (mean - lsl) / (3 * std_overall) if has_lsl and std_overall > 0 else None
    if ppu is not None and ppl is not None:
        ppk = min(ppu, ppl)
    elif ppu is not None:
        ppk = ppu
    elif ppl is not None:
        ppk = ppl
    else:
        ppk = None

    # Cpm
    cpm = None
    if has_lsl and has_usl and target is not None and std_overall > 0:
        cpm = (usl - lsl) / (6 * np.sqrt(std_overall**2 + (mean - target)**2))

    # PPM
    ppm = 0.0
    if has_lsl:
        ppm += norm.cdf(lsl, mean, std_overall)
    if has_usl:
        ppm += (1 - norm.cdf(usl, mean, std_overall))
    ppm *= 1e6

    result = {
        "mean": mean,
        "std_overall": std_overall,
        "std_within": std_within,
        "cp": cp if cp is not None else 0,
        "cpk": cpk if cpk is not None else 0,
        "pp": pp if pp is not None else 0,
        "ppk": ppk if ppk is not None else 0,
        "ppm": ppm,
        "cpu": cpu if cpu is not None else 0,
        "cpl": cpl if cpl is not None else 0,
        "ppu": ppu if ppu is not None else 0,
        "ppl": ppl if ppl is not None else 0,
    }
    if cpm is not None:
        result["cpm"] = cpm
    return result
