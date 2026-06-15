import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import norm, gaussian_kde
from scipy import stats

# ==================================================
# 中文字体配置
# ==================================================

def _setup_chinese_font():
    preferred_fonts = [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi",
        "FangSong", "Arial Unicode MS",
    ]
    available = {f.name: f for f in font_manager.fontManager.ttflist}
    for name in preferred_fonts:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    fallback_paths = [
        "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/simkai.ttf",
    ]
    for path in fallback_paths:
        try:
            font_prop = font_manager.FontProperties(fname=path)
            plt.rcParams["font.sans-serif"] = [font_prop.get_name(), "DejaVu Sans", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False
            return font_prop.get_name()
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return None

_CHINESE_FONT = _setup_chinese_font()

# ==================================================
# 企业级图表主题
# ==================================================

CHART_COLORS = {
    "primary": "#1a56db",
    "secondary": "#64748b",
    "accent": "#f59e0b",
    "success": "#059669",
    "danger": "#dc2626",
    "info": "#0891b2",
    "grid": "#e2e8f0",
    "bg": "#f8fafc",
}

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": CHART_COLORS["bg"],
    "axes.edgecolor": "#cbd5e1",
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.color": CHART_COLORS["grid"],
    "axes.labelcolor": "#475569",
    "text.color": "#334155",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#e2e8f0",
    "legend.framealpha": 0.95,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
})


# ==================================================
# 正态过程能力直方图
# ==================================================

def plot_capability(data, mean, std_overall, std_within, lsl, usl, target):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(data, bins="auto", density=True, alpha=0.6, edgecolor="black")

    vals = [np.min(data), np.max(data)]
    if lsl is not None: vals.append(lsl)
    if usl is not None: vals.append(usl)
    xmin, xmax = min(vals), max(vals)
    pad = (xmax - xmin) * 0.1
    x = np.linspace(xmin - pad, xmax + pad, 500)

    ax.plot(x, norm.pdf(x, mean, std_overall), linewidth=2, label="整体分布")
    ax.plot(x, norm.pdf(x, mean, std_within), "--", linewidth=2, label="组内分布")

    if lsl is not None:
        ax.axvline(lsl, color="red", linestyle=":", label=f"LSL={lsl:.3f}")
    ax.axvline(usl, color="red", linestyle=":", label=f"USL={usl:.3f}")
    if target is not None:
        ax.axvline(target, color="green", linestyle="-.", label=f"目标={target:.3f}")

    ax.set_title("过程能力直方图", fontsize=14, fontweight="bold")
    ax.set_xlabel("测量值", fontsize=12)
    ax.set_ylabel("密度", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    return fig


# ==================================================
# 箱线图
# ==================================================

def plot_boxplot(data):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.boxplot(data, vert=True)
    ax.set_title("箱线图", fontsize=14, fontweight="bold")
    ax.set_ylabel("测量值", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    return fig


# ==================================================
# Q-Q 图
# ==================================================

def plot_qq(data):
    fig, ax = plt.subplots(figsize=(6, 6))

    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title("Q-Q 图", fontsize=14, fontweight="bold")
    ax.set_xlabel("理论分位数", fontsize=12)
    ax.set_ylabel("样本分位数", fontsize=12)
    ax.grid(True, alpha=0.3)

    return fig


# ==================================================
# 正态概率图
# ==================================================

def plot_probability(data):
    fig, ax = plt.subplots(figsize=(6, 6))

    sorted_data = np.sort(data)
    p = np.arange(1, len(data) + 1) / (len(data) + 1)
    z = norm.ppf(p)

    ax.scatter(sorted_data, z, alpha=0.6)
    ax.set_title("正态概率图", fontsize=14, fontweight="bold")
    ax.set_xlabel("观测值", fontsize=12)
    ax.set_ylabel("期望 Z 值", fontsize=12)
    ax.grid(True, alpha=0.3)

    return fig


# ==================================================
# 子组均值控制图
# ==================================================

def plot_subgroup_means(data, subgroup_size):
    k = len(data) // subgroup_size
    data = data[:k * subgroup_size]
    matrix = data.reshape(k, subgroup_size)
    means = matrix.mean(axis=1)

    grand_mean = np.mean(means)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(range(1, k + 1), means, marker="o", linewidth=1.5, color="#1565C0")
    ax.axhline(grand_mean, color="red", linestyle="--", label=f"总均值={grand_mean:.4f}")

    ax.set_title("子组均值控制图", fontsize=14, fontweight="bold")
    ax.set_xlabel("子组编号", fontsize=12)
    ax.set_ylabel("子组均值", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    return fig


# ==================================================
# P 控制图
# ==================================================

def plot_p_chart(samples, defectives):
    p = defectives / samples
    p_bar = defectives.sum() / samples.sum()

    ucl, lcl = [], []
    for n in samples:
        sigma_p = np.sqrt(p_bar * (1 - p_bar) / n) if n > 0 else 0
        ucl.append(p_bar + 3 * sigma_p)
        lcl.append(max(0, p_bar - 3 * sigma_p))

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(1, len(samples) + 1)

    ax.plot(x, p, marker="o", linewidth=1.5, color="#1565C0", markersize=5, label="不合格率")
    ax.axhline(p_bar, color="#2E7D32", linestyle="--", linewidth=1.5, label=f"中心线 CL={p_bar:.4f}")
    ax.plot(x, ucl, "r--", linewidth=1.5, label=f"上控制限 UCL")
    ax.plot(x, lcl, "r--", linewidth=1.5, label=f"下控制限 LCL")

    ax.set_title("P 控制图", fontsize=14, fontweight="bold")
    ax.set_xlabel("批次编号", fontsize=12)
    ax.set_ylabel("不合格率", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    return fig


# ==================================================
# 累计不合格率趋势图
# ==================================================

def plot_cumulative_rate(samples, defectives):
    import matplotlib.ticker as mticker

    cum_defects = np.cumsum(defectives)
    cum_samples = np.cumsum(samples)
    cum_rate = cum_defects / np.maximum(cum_samples, 1)
    p_bar = defectives.sum() / samples.sum()

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(1, len(samples) + 1)

    ax.plot(x, cum_rate, marker="o", linewidth=2, color="#1565C0", markersize=4, label="累计不合格率")
    ax.axhline(p_bar, color="#E53935", linestyle="--", linewidth=1.5, label=f"总体平均 p̄={p_bar:.4%}")

    ax.fill_between(x, cum_rate, p_bar, alpha=0.1, color="#1565C0")

    ax.set_title("累计不合格率变化趋势", fontsize=14, fontweight="bold")
    ax.set_xlabel("批次编号", fontsize=12)
    ax.set_ylabel("累计不合格率", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig


# ==================================================
# 缺陷率分布图
# ==================================================

def plot_defect_distribution(samples, defectives, nbins=None):
    rates = defectives / np.maximum(samples, 1)
    p_bar = defectives.sum() / samples.sum()

    if nbins is None:
        nbins = max(6, min(20, int(np.sqrt(len(rates)))))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        rates * 100, bins=nbins, density=True,
        alpha=0.6, edgecolor="white", color="#42A5F5",
        label="各批次不合格率分布",
    )

    if len(rates) >= 4:
        try:
            kde = gaussian_kde(rates)
            x_kde = np.linspace(max(0, rates.min() * 0.8), min(1, rates.max() * 1.2), 200)
            ax.plot(x_kde * 100, kde(x_kde) / 100, color="#E53935", linewidth=2, label="密度曲线")
        except Exception:
            pass

    ax.axvline(p_bar * 100, color="#2E7D32", linestyle="--", linewidth=1.5, label=f"平均 p̄={p_bar:.3%}")

    ax.set_title("缺陷率分布分析", fontsize=14, fontweight="bold")
    ax.set_xlabel("不合格率 (%)", fontsize=12)
    ax.set_ylabel("密度", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # 偏态标注
    if np.std(rates, ddof=1) > 0:
        skewness = float(np.mean(((rates - np.mean(rates)) / np.std(rates, ddof=1))**3))
    else:
        skewness = 0

    if skewness > 0.5:
        skew_text = "右偏（存在偏高异常批次）"
    elif skewness < -0.5:
        skew_text = "左偏（存在偏低异常批次）"
    else:
        skew_text = "近似对称"

    cv_val = np.std(rates, ddof=1) / max(np.mean(rates), 1e-10)

    ax.text(
        0.98, 0.95, f"偏态：{skew_text}\n波动 CV：{cv_val:.2f}",
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    return fig


# ==================================================
# 二项分布拟合检验图
# ==================================================

def plot_binomial_fit(samples, defectives, p_bar):
    rates = defectives / np.maximum(samples, 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    bins_edges = np.linspace(0, max(rates) * 1.1, 12)
    observed, _ = np.histogram(rates, bins=bins_edges)

    avg_n = int(np.mean(samples))
    expected_normal = np.array([
        len(rates) * np.diff(norm.cdf(
            [bins_edges[i], bins_edges[i+1]],
            loc=p_bar,
            scale=np.sqrt(p_bar * (1 - p_bar) / max(avg_n, 1)),
        ))[0]
        for i in range(len(bins_edges) - 1)
    ])
    expected_normal = expected_normal * (observed.sum() / max(expected_normal.sum(), 1e-10))

    x_pos = np.arange(len(observed))
    width = 0.35

    ax.bar(x_pos - width/2, observed, width, label="观测频数", color="#42A5F5", edgecolor="white")
    ax.bar(x_pos + width/2, expected_normal, width, label="二项期望频数", color="#FFA726", edgecolor="white")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{bins_edges[i]:.2%}~{bins_edges[i+1]:.2%}" for i in range(len(bins_edges) - 1)],
        rotation=45, ha="right", fontsize=8,
    )

    ax.set_title("二项分布拟合检验", fontsize=14, fontweight="bold")
    ax.set_xlabel("不合格率区间", fontsize=12)
    ax.set_ylabel("频数", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    return fig

