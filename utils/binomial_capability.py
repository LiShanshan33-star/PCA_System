import numpy as np
from scipy.stats import norm, chi2
from scipy.special import comb


def calculate_binomial_capability(samples, defectives):
    """
    二项过程能力分析 — 核心统计计算

    Parameters
    ----------
    samples : np.ndarray
        各批次样本数
    defectives : np.ndarray
        各批次不合格品数

    Returns
    -------
    dict
    """
    total_samples = int(np.sum(samples))
    total_defects = int(np.sum(defectives))
    n_batches = len(samples)

    # 平均不合格率
    p_bar = total_defects / total_samples if total_samples > 0 else 0.0

    # PPM / DPMO（二项过程两者等价）
    ppm = p_bar * 1_000_000
    dpmo = ppm

    # --------------------------------------------------
    # Sigma Level (长期，1.5σ偏移)
    # --------------------------------------------------
    if dpmo <= 0:
        sigma_level = 6.0
    else:
        sigma_level = 0.8406 + np.sqrt(29.37 - 2.221 * np.log(max(dpmo, 1e-10)))

    # --------------------------------------------------
    # Zbench（长期过程 Z 值 = norm.ppf(1 - p_bar)）
    # --------------------------------------------------
    if p_bar <= 0:
        z_bench = 6.0
    elif p_bar >= 1:
        z_bench = -6.0
    else:
        z_bench = norm.ppf(1.0 - p_bar)

    # --------------------------------------------------
    # 95% 置信区间（Wilson Score Method）
    # --------------------------------------------------
    z = norm.ppf(0.975)  # 1.96
    n_total = total_samples
    p = p_bar

    denominator = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt((p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denominator

    ci_lower = max(0.0, centre - margin)
    ci_upper = min(1.0, centre + margin)

    ci_ppm_lower = ci_lower * 1_000_000
    ci_ppm_upper = ci_upper * 1_000_000

    # --------------------------------------------------
    # 各批次统计
    # --------------------------------------------------
    batch_rates = defectives / np.maximum(samples, 1)

    # --------------------------------------------------
    # 二项分布拟合优度检验 (Chi-Square)
    # --------------------------------------------------
    expected = samples * p_bar
    # 合并期望值过小的批次
    observed_list = []
    expected_list = []
    pooled_obs = 0
    pooled_exp = 0

    for obs, exp in zip(defectives.tolist(), expected.tolist()):
        if exp < 5:
            pooled_obs += obs
            pooled_exp += exp
        else:
            if pooled_exp > 0:
                observed_list.append(pooled_obs)
                expected_list.append(pooled_exp)
                pooled_obs = 0
                pooled_exp = 0
            observed_list.append(obs)
            expected_list.append(exp)

    if pooled_exp > 0:
        observed_list.append(pooled_obs)
        expected_list.append(pooled_exp)

    observed_arr = np.array(observed_list)
    expected_arr = np.array(expected_list)

    if len(observed_arr) > 1:
        chi_sq = np.sum((observed_arr - expected_arr)**2 / expected_arr)
        df = len(observed_arr) - 1
        p_fit = 1.0 - chi2.cdf(chi_sq, df) if df > 0 else 1.0
    else:
        chi_sq = 0.0
        df = 0
        p_fit = 1.0

    return {
        # 基础统计
        "total_samples": total_samples,
        "total_defects": total_defects,
        "n_batches": n_batches,
        "p_bar": p_bar,

        # 能力指标
        "ppm": ppm,
        "dpmo": dpmo,
        "sigma": sigma_level,
        "z_bench": z_bench,

        # 置信区间
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_ppm_lower": ci_ppm_lower,
        "ci_ppm_upper": ci_ppm_upper,

        # 批次统计
        "batch_rates": batch_rates,

        # 拟合检验
        "chi_sq": chi_sq,
        "chi_df": df,
        "p_fit": p_fit,
    }
