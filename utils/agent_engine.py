"""
SPC 全流程分析智能体引擎
Auto-detect → Analyze → AI Interpret → PDF
"""
import numpy as np
import pandas as pd
from io import BytesIO

from utils.data_loader import load_file
from utils.normality_test import shapiro_test
from utils.capability import calculate_capability
from utils.boxcox_utils import boxcox_transform
from utils.binomial_capability import calculate_binomial_capability
from utils.plotting import (
    plot_capability, plot_qq, plot_boxplot, plot_probability,
    plot_p_chart, plot_cumulative_rate, plot_defect_distribution, plot_binomial_fit,
)
from utils.ai_assistant import ask_ai
from utils.report_generator import (
    generate_normal_report, generate_binomial_report, generate_boxcox_report,
    generate_metrology_report, generate_attributes_report,
    generate_small_batch_report, generate_small_shift_report,
)


# ==================================================
# Step 1: 数据读取与结构分析
# ==================================================
def load_and_profile(uploaded_file):
    df = load_file(uploaded_file)
    profile = {
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "missing_pct": (df.isna().sum() / len(df) * 100).round(2).to_dict(),
        "num_cols": [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
        "cat_cols": [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])],
    }
    stats = {}
    for c in profile["num_cols"]:
        s = df[c].dropna()
        stats[c] = {
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std(ddof=1)), 4),
            "unique": int(s.nunique()),
            "n": len(s),
        }
    profile["num_stats"] = stats
    return df, profile


# ==================================================
# Step 2: 数据结构智能识别
# ==================================================
def _is_group_column(name, series):
    name_lower = str(name).lower()
    keywords = ["批次", "batch", "组", "group", "序号", "id", "no", "编号", "样本", "子组", "subgroup", "日期", "date", "天", "day"]
    if any(kw in name_lower for kw in keywords):
        return True
    if pd.api.types.is_numeric_dtype(series):
        vals = series.dropna()
        if len(vals) == 0:
            return False
        if pd.api.types.is_integer_dtype(series) and vals.nunique() == len(vals):
            return True
        if vals.nunique() == len(vals):
            int_vals = np.round(vals).astype(np.int64)
            if np.allclose(vals, int_vals.astype(float)):
                return True
    return False


def detect_structure(df, profile):
    """智能识别数据结构，返回结构描述和推荐方法列表"""
    num_cols = profile["num_cols"]
    nc = len(num_cols)
    if nc == 0:
        return {"error": "未检测到数值列"}

    group_cols = [c for c in num_cols if _is_group_column(c, df[c])]
    measure_cols = [c for c in num_cols if c not in group_cols]

    # 如果没有分组列，所有数值列都是测量列
    if not measure_cols:
        measure_cols = num_cols
        group_cols = []

    result = {
        "num_cols": num_cols,
        "group_cols": group_cols,
        "measure_cols": measure_cols,
        "nc": nc,
        "methods": [],
        "recommended": None,
        "format": None,
    }

    # --- Case 1: 宽格式 (一行一个批次，多列测量值) ---
    is_wide = False
    if len(measure_cols) >= 2:
        means = [df[c].mean() for c in measure_cols]
        stds = [df[c].std() for c in measure_cols]
        mean_cv = np.std(means) / (abs(np.mean(means)) + 1e-10)
        # 如果多列均值和标准差相近，很可能是宽格式测量数据
        if mean_cv < 0.3 and max(stds) / (min(stds) + 1e-10) < 15:
            is_wide = True

    # --- Case 2: 二项/计数数据检测 ---
    is_binomial = False
    if len(measure_cols) >= 2 and not is_wide:
        col_means = {c: df[c].mean() for c in measure_cols}
        sorted_cols = sorted(measure_cols, key=lambda c: col_means[c], reverse=True)
        big_col, small_col = sorted_cols[0], sorted_cols[-1]
        big_vals = df[big_col].dropna()
        small_vals = df[small_col].dropna()
        if len(big_vals) >= 3 and np.all(big_vals > 0) and big_vals.mean() > small_vals.mean() * 2:
            rates = small_vals.values / big_vals.values
            if np.all((rates >= 0) & (rates <= 1)):
                is_binomial = True

    # --- Case 3: 缺陷数检测 (U/C chart) ---
    is_defect = False
    if len(measure_cols) >= 2 and not is_wide and not is_binomial:
        col_means = {c: df[c].mean() for c in measure_cols}
        sorted_cols = sorted(measure_cols, key=lambda c: col_means[c], reverse=True)
        a, b = sorted_cols[0], sorted_cols[-1]
        # 如果一大一小但有比例关系，但不是0-1范围 → 可能是缺陷数
        if df[a].mean() > df[b].mean() * 1.5:
            is_defect = True

    # --- 构建推荐方法列表 ---
    methods = []

    if is_wide:
        result["format"] = "wide"
        result["subgroup_size"] = len(measure_cols)
        methods.append({"id": "xbar_r", "name": "均值-极差控制图 (Xbar-R)",
                        "desc": f"检测到宽格式数据：{len(measure_cols)}列测量值，适合子组控制图"})
        methods.append({"id": "normal_cap", "name": "正态过程能力分析",
                        "desc": "基于展平后的测量值进行过程能力评估"})
        result["recommended"] = "xbar_r"
    elif is_binomial:
        result["format"] = "long_count"
        col_means = {c: df[c].mean() for c in measure_cols}
        sorted_cols = sorted(measure_cols, key=lambda c: col_means[c], reverse=True)
        result["sample_col"] = sorted_cols[0]
        result["defect_col"] = sorted_cols[-1]
        methods.append({"id": "binomial", "name": "二项过程能力分析 + P/NP控制图",
                        "desc": f"检测到计数型数据（样本量列={result['sample_col']}，不合格列={result['defect_col']}）"})
        result["recommended"] = "binomial"
    elif is_defect:
        result["format"] = "long_defect"
        col_means = {c: df[c].mean() for c in measure_cols}
        sorted_cols = sorted(measure_cols, key=lambda c: col_means[c], reverse=True)
        result["sample_col"] = sorted_cols[0]
        result["defect_col"] = sorted_cols[-1]
        methods.append({"id": "u_chart", "name": "U/C 缺陷数控制图",
                        "desc": f"检测到缺陷计数数据"})
        result["recommended"] = "u_chart"
    else:
        # 单个测量列
        result["format"] = "single_column"
        result["value_col"] = measure_cols[0] if measure_cols else num_cols[0]
        methods.append({"id": "imr", "name": "单值-移动极差控制图 (I-MR)",
                        "desc": "单列连续数据，适合单值控制图"})
        methods.append({"id": "normal_cap", "name": "正态过程能力分析",
                        "desc": "评估过程能力指标 Cp/Cpk/Pp/Ppk"})
        # 检测是否非正态
        data = df[result["value_col"]].dropna().values
        if len(data) >= 3:
            _, p = shapiro_test(data)
            result["normality_p"] = p
            if p <= 0.05:
                methods.append({"id": "boxcox", "name": "Box-Cox 变换分析",
                                "desc": f"数据可能非正态（P={p:.4f}），建议尝试Box-Cox变换"})
                result["is_non_normal"] = True
            else:
                result["is_non_normal"] = False

        # 单列也支持CUSUM和EWMA
        methods.append({"id": "cusum", "name": "累积和控制图 (CUSUM)",
                        "desc": "检测过程均值微小偏移"})
        methods.append({"id": "ewma", "name": "指数加权移动平均 (EWMA)",
                        "desc": "平滑监控过程变化趋势"})
        result["recommended"] = "normal_cap"

    result["methods"] = methods
    return result


# ==================================================
# Step 3: 自动执行分析
# ==================================================
def execute_analysis(df, profile, structure, lsl=None, usl=None, target=None, subgroup_size=1):
    method = structure.get("recommended", "")
    result = {"method": method, "capability": {}, "figures": {}, "structure": structure}

    if structure.get("format") == "wide":
        measure_cols = structure["measure_cols"]
        data_matrix = df[measure_cols].values
        n_sub = len(data_matrix)
        sg_size = len(measure_cols)

        xbar_vals = np.mean(data_matrix, axis=1)
        r_vals = np.max(data_matrix, axis=1) - np.min(data_matrix, axis=1)
        xbarbar = np.mean(xbar_vals)
        rbar = np.mean(r_vals)

        A2_dict = {2:1.880,3:1.023,4:0.729,5:0.577,6:0.483,7:0.419,8:0.373,9:0.337,10:0.308}
        D3_dict = {2:0,3:0,4:0,5:0,6:0,7:0.076,8:0.136,9:0.184,10:0.223}
        D4_dict = {2:3.267,3:2.574,4:2.282,5:2.114,6:2.004,7:1.924,8:1.864,9:1.816,10:1.777}
        A2 = A2_dict.get(sg_size, 0.577)
        D3 = D3_dict.get(sg_size, 0)
        D4 = D4_dict.get(sg_size, 2.114)

        xr = {}
        xr["xbar_values"] = xbar_vals
        xr["r_values"] = r_vals
        xr["xbarbar"] = xbarbar
        xr["rbar"] = rbar
        xr["xbar_ucl"] = xbarbar + A2 * rbar
        xr["xbar_lcl"] = xbarbar - A2 * rbar
        xr["r_ucl"] = D4 * rbar
        xr["r_lcl"] = D3 * rbar
        xr["n_subgroups"] = n_sub
        xr["subgroup_size"] = sg_size
        result["xbar_r"] = xr
        result["capability"] = {"xbarbar": xbarbar, "rbar": rbar, "n_subgroups": n_sub}

        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
        x = np.arange(1, n_sub + 1)
        ax1.plot(x, xbar_vals, "bo-", markersize=5)
        ax1.axhline(xbarbar, color="green")
        ax1.axhline(xr["xbar_ucl"], color="red", linestyle="--")
        ax1.axhline(xr["xbar_lcl"], color="red", linestyle="--")
        ax1.set_title("Xbar Chart")
        ax1.grid(True, alpha=0.3)
        ax2.plot(x, r_vals, "bo-", markersize=5)
        ax2.axhline(rbar, color="green")
        ax2.axhline(xr["r_ucl"], color="red", linestyle="--")
        ax2.set_title("R Chart")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        result["figures"]["main"] = fig

        flat_data = data_matrix.flatten()
        flat_data = flat_data[~np.isnan(flat_data)]
        _lsl = lsl if (lsl is not None) else np.percentile(flat_data, 0.135)
        _usl = usl if (usl is not None) else np.percentile(flat_data, 99.865)
        _tgt = target if (target is not None) else float(np.mean(flat_data))
        cap = calculate_capability(flat_data, _lsl, _usl, sg_size, _tgt)
        result["capability"].update(cap)
        result["figures"]["capability"] = plot_capability(flat_data, cap["mean"], cap["std_overall"], cap["std_within"], _lsl, _usl, _tgt)
        result["figures"]["qq"] = plot_qq(flat_data)

    elif method == "binomial":
        sample_col = structure["sample_col"]
        defect_col = structure["defect_col"]
        samples = df[sample_col].values.astype(float)
        defects = df[defect_col].values.astype(float)
        cap = calculate_binomial_capability(samples, defects)
        result["capability"] = cap
        result["figures"]["p_chart"] = plot_p_chart(samples, defects)
        result["figures"]["cumulative"] = plot_cumulative_rate(samples, defects)
        result["figures"]["distribution"] = plot_defect_distribution(samples, defects)
        result["figures"]["fit"] = plot_binomial_fit(samples, defects, cap["p_bar"])

    elif method == "u_chart":
        sample_col = structure["sample_col"]
        defect_col = structure["defect_col"]
        samples = df[sample_col].values.astype(float)
        defects = df[defect_col].values.astype(float)
        u_i = defects / samples
        u_bar = np.sum(defects) / np.sum(samples)
        ucl = u_bar + 3 * np.sqrt(u_bar / samples)
        lcl = np.maximum(0, u_bar - 3 * np.sqrt(u_bar / samples))
        result["capability"] = {"u_bar": u_bar, "total_defects": int(np.sum(defects)), "n": len(samples)}
        result["u_chart"] = {"u_i": u_i, "u_bar": u_bar, "ucl": ucl, "lcl": lcl}
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(1, len(samples) + 1)
        ax.plot(x, u_i, "bo-", markersize=5)
        ax.axhline(u_bar, color="green")
        ax.step(x, ucl, "r--", where="mid")
        ax.step(x, lcl, "r--", where="mid")
        ax.set_title("U Chart")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        result["figures"]["main"] = fig

    else:
        value_col = structure.get("value_col", profile["num_cols"][0])
        data = df[value_col].dropna().values
        mr = np.abs(np.diff(data))
        imr = {}
        imr["values"] = data
        imr["mr"] = mr
        imr["mean"] = np.mean(data)
        imr["mrbar"] = np.mean(mr)
        imr["i_ucl"] = np.mean(data) + 2.66 * np.mean(mr)
        imr["i_lcl"] = np.mean(data) - 2.66 * np.mean(mr)
        imr["mr_ucl"] = 3.267 * np.mean(mr)
        imr["n"] = len(data)
        result["imr"] = imr

        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
        ax1.plot(range(1, len(data)+1), data, "bo-", markersize=5)
        ax1.axhline(imr["mean"], color="green")
        ax1.axhline(imr["i_ucl"], color="red", linestyle="--")
        ax1.axhline(imr["i_lcl"], color="red", linestyle="--")
        ax1.set_title("I-MR Chart - Individuals")
        ax1.grid(True, alpha=0.3)
        ax2.plot(range(2, len(data)+1), mr, "bo-", markersize=5)
        ax2.axhline(imr["mrbar"], color="green")
        ax2.axhline(imr["mr_ucl"], color="red", linestyle="--")
        ax2.set_title("Moving Range")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        result["figures"]["main"] = fig

        _, p = shapiro_test(data)
        _lsl = lsl if (lsl is not None) else np.percentile(data, 0.135)
        _usl = usl if (usl is not None) else np.percentile(data, 99.865)
        _tgt = target if (target is not None) else float(np.mean(data))
        cap = calculate_capability(data, _lsl, _usl, subgroup_size, _tgt)
        result["capability"] = cap
        result["p_value"] = p
        result["figures"]["capability"] = plot_capability(data, cap["mean"], cap["std_overall"], cap["std_within"], _lsl, _usl, _tgt)
        result["figures"]["qq"] = plot_qq(data)
        result["figures"]["boxplot"] = plot_boxplot(data)

    return result


# ==================================================
# Step 4: AI ++++++
# ==================================================
def ai_interpret(structure, result):
    method = result.get("method", "")
    cap = result.get("capability", {})
    ctx_parts = []
    fmt = structure.get("format", "")
    if fmt == "wide":
        ctx_parts.append("Xbar-R: Xbarbar=%.4f Rbar=%.4f" % (cap.get('xbarbar',0), cap.get('rbar',0)))
    elif method == "binomial":
        ctx_parts.append("Binomial: p=%.4f%% Sigma=%.2f" % (cap.get('p_bar',0)*100, cap.get('sigma',0)))
    elif method == "u_chart":
        ctx_parts.append("U-chart: u=%.4f" % cap.get('u_bar',0))
    else:
        ctx_parts.append("Normal Cap: Cpk=%.2f PPM=%.0f" % (cap.get('cpk',0), cap.get('ppm',0)))
    context = " / ".join(ctx_parts)
    prompt = "SPC Result: " + context + ". Give: (1) 3-sentence summary for management (2) 3 improvement suggestions for quality engineer."
    try:
        return ask_ai(prompt, "")
    except Exception as e:
        return "AI not available: " + str(e)


def generate_agent_pdf(method, result, ai_text="", lsl=None, usl=None, target=None):
    cap = result.get("capability", {})
    figs = result.get("figures", {})

    if method == "binomial":
        return generate_binomial_report(
            result=cap, ai_analysis=ai_text,
            fig_p_chart=figs.get("p_chart"), fig_cumulative=figs.get("cumulative"),
            fig_distribution=figs.get("distribution"), fig_fit=figs.get("fit"),
        )
    elif method == "u_chart":
        return generate_attributes_report(
            result=result.get("u_chart", {}), chart_type="u",
            auto_eval="", ai_analysis=ai_text, fig_chart=figs.get("main"),
        )
    elif result.get("xbar_r"):
        return generate_metrology_report(
            result=result["xbar_r"], chart_type="xbar_r",
            auto_eval="", ai_analysis=ai_text, fig_chart=figs.get("main"),
        )
    else:
        return generate_normal_report(
            result=cap, lsl=lsl or 0, usl=usl or 0, target=target or 0,
            p_value=result.get("p_value", 1),
            fig_capability=figs.get("capability"), fig_qq=figs.get("qq"),
            fig_boxplot=figs.get("boxplot"), fig_probability=figs.get("probability"),
            ai_analysis=ai_text,
        )
