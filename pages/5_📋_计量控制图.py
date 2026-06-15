import streamlit as st
st.set_page_config(page_title="常规计量控制图", page_icon="📋", layout="wide")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils.data_loader import load_file
from utils.ai_assistant import ask_ai
from utils.theme import apply_theme, Colors, kpi_card, render_copilot_sidebar
from utils.report_generator import generate_metrology_report, generate_attributes_report, generate_small_batch_report, generate_small_shift_report

apply_theme()

# ==================================================
# 中文字体
# ==================================================
from matplotlib import font_manager
def _setup_chinese_font():
    preferred = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
    available = {f.name: f for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    fallback = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"]
    for path in fallback:
        try:
            font_prop = font_manager.FontProperties(fname=path)
            plt.rcParams["font.sans-serif"] = [font_prop.get_name(), "DejaVu Sans", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

_setup_chinese_font()

# ==================================================
# 控制图常数 (Minitab标准)
# ==================================================
# A2, D3, D4 for Xbar-R (n=2..25)
_A2_TABLE = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308, 11: 0.285, 12: 0.266, 13: 0.249, 14: 0.235, 15: 0.223, 16: 0.212, 17: 0.203, 18: 0.194, 19: 0.187, 20: 0.180, 21: 0.173, 22: 0.167, 23: 0.162, 24: 0.157, 25: 0.153}
_D3_TABLE = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223, 11: 0.256, 12: 0.284, 13: 0.308, 14: 0.329, 15: 0.348, 16: 0.364, 17: 0.379, 18: 0.392, 19: 0.404, 20: 0.414, 21: 0.425, 22: 0.434, 23: 0.443, 24: 0.452, 25: 0.459}
_D4_TABLE = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777, 11: 1.744, 12: 1.716, 13: 1.692, 14: 1.671, 15: 1.652, 16: 1.636, 17: 1.621, 18: 1.608, 19: 1.596, 20: 1.586, 21: 1.575, 22: 1.566, 23: 1.557, 24: 1.548, 25: 1.541}

# ==================================================
# 示例数据
# ==================================================
def load_example_data(exp_type):
    if exp_type == "xbar_r":
        # 实验二 例1：活塞环直径数据 (Table 3, 25 subgroups × 5)
        subgroups = [
            [74.030, 74.002, 74.019, 73.992, 74.008],
            [73.995, 73.992, 74.001, 74.001, 74.011],
            [73.988, 74.024, 74.021, 74.005, 74.002],
            [74.002, 73.996, 73.993, 74.015, 74.009],
            [73.992, 74.007, 74.015, 73.989, 74.014],
            [74.009, 73.994, 73.997, 73.985, 73.993],
            [73.995, 74.006, 73.994, 74.000, 74.005],
            [73.985, 74.003, 73.993, 74.015, 73.998],
            [74.008, 73.995, 74.009, 74.005, 74.004],
            [73.998, 74.000, 73.990, 74.007, 73.995],
            [73.994, 73.998, 73.994, 73.995, 73.990],
            [74.004, 74.000, 74.007, 74.000, 73.996],
            [73.983, 74.002, 73.998, 73.997, 74.012],
            [74.006, 73.967, 73.994, 74.000, 73.984],
            [74.012, 74.014, 73.998, 73.999, 74.007],
            [74.000, 73.984, 74.005, 73.998, 73.996],
            [73.994, 74.012, 73.986, 74.005, 74.007],
            [74.006, 74.010, 74.018, 74.003, 74.000],
            [73.984, 74.002, 74.003, 74.005, 73.997],
            [73.990, 74.012, 74.010, 74.006, 73.998],
            [74.001, 73.989, 73.995, 74.008, 74.002],
            [73.996, 74.005, 73.999, 74.010, 74.001],
            [74.003, 73.993, 74.008, 73.997, 74.005],
            [73.999, 74.007, 73.995, 74.002, 74.008],
            [74.005, 73.998, 74.002, 74.003, 73.997],
        ]
        df = pd.DataFrame(subgroups, columns=[f"X{i+1}" for i in range(5)])
        df.index = range(1, len(df) + 1)
        return df, 5
    elif exp_type == "imr":
        # 实验二 例2：焊接强度数据 (Table 4)
        data = [
            12.1, 12.1, 12.4, 13.2, 13.3, 12.4, 13.0, 13.5,
            12.5, 12.8, 13.1, 12.8, 13.4, 13.0, 12.5,
            12.2, 13.0, 12.8, 12.5, 12.6, 12.4, 12.8,
            12.7, 12.6, 13.0,
        ]
        return pd.DataFrame({"测量值": data}), 1
    return None, 1


# ==================================================
# Xbar-R 计算 (Minitab算法)
# ==================================================
def compute_xbar_r(data_matrix, subgroup_size):
    """data_matrix: 2D array (n_subgroups × subgroup_size)"""
    n_subgroups = len(data_matrix)
    xbar_values = np.mean(data_matrix, axis=1)
    r_values = np.max(data_matrix, axis=1) - np.min(data_matrix, axis=1)
    xbarbar = np.mean(xbar_values)
    rbar = np.mean(r_values)
    A2 = _A2_TABLE.get(subgroup_size, 0.577)
    D3 = _D3_TABLE.get(subgroup_size, 0)
    D4 = _D4_TABLE.get(subgroup_size, 2.114)

    xbar_ucl = xbarbar + A2 * rbar
    xbar_lcl = xbarbar - A2 * rbar
    r_ucl = D4 * rbar
    r_lcl = D3 * rbar

    return {
        "xbar_values": xbar_values, "r_values": r_values,
        "xbarbar": xbarbar, "rbar": rbar,
        "xbar_ucl": xbar_ucl, "xbar_lcl": xbar_lcl,
        "r_ucl": r_ucl, "r_lcl": r_lcl,
        "n_subgroups": n_subgroups, "subgroup_size": subgroup_size,
    }


# ==================================================
# I-MR 计算
# ==================================================
def compute_imr(data):
    """data: 1D array of individual measurements"""
    n = len(data)
    mr = np.abs(np.diff(data))
    mrbar = np.mean(mr)
    mean_val = np.mean(data)

    i_ucl = mean_val + 2.66 * mrbar
    i_lcl = mean_val - 2.66 * mrbar
    mr_ucl = 3.267 * mrbar

    return {
        "values": data, "mr": mr,
        "mean": mean_val, "mrbar": mrbar,
        "i_ucl": i_ucl, "i_lcl": i_lcl,
        "mr_ucl": mr_ucl, "n": n,
    }


# ==================================================
# 判异准则检测
# ==================================================
def check_control_violations(values, ucl, lcl, cl, n_subgroups):
    """返回违反控制限的批次列表和异常描述"""
    violations = []
    desc = []
    for i in range(len(values)):
        if values[i] > ucl or values[i] < lcl:
            violations.append(i + 1)
    # 判异准则: 连续9点在中心线同一侧
    above_cl = np.array(values) > cl
    below_cl = np.array(values) < cl
    run = 1
    for i in range(1, len(values)):
        if above_cl[i] == above_cl[i-1]:
            run += 1
        else:
            run = 1
        if run >= 9:
            if len(desc) == 0 or desc[-1] != f"连续9点在中心线同一侧（第{i-8+1}~{i+1}批）":
                desc.append(f"连续9点在中心线同一侧（第{i-8+1}~{i+1}批）")
    # 连续6点递增或递减
    for i in range(len(values) - 6 + 1):
        segment = values[i:i+6]
        if np.all(np.diff(segment) > 0):
            desc.append(f"连续6点递增（第{i+1}~{i+6}批）")
        if np.all(np.diff(segment) < 0):
            desc.append(f"连续6点递减（第{i+1}~{i+6}批）")
    return violations, list(set(desc))


# ==================================================
# 绘图
# ==================================================
def plot_xbar_r(result, ylabel="测量值"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 0.7]})

    x = np.arange(1, result["n_subgroups"] + 1)

    # Xbar chart
    ax1.plot(x, result["xbar_values"], "bo-", markersize=5, linewidth=1.5, label="样本均值")
    ax1.axhline(result["xbarbar"], color="green", linestyle="-", linewidth=1.5, label=f"CL={result['xbarbar']:.4f}")
    ax1.axhline(result["xbar_ucl"], color="red", linestyle="--", linewidth=1.5, label=f"UCL={result['xbar_ucl']:.4f}")
    ax1.axhline(result["xbar_lcl"], color="red", linestyle="--", linewidth=1.5, label=f"LCL={result['xbar_lcl']:.4f}")
    ax1.set_title(f"均值控制图 (n={result['subgroup_size']})", fontsize=14, fontweight="bold")
    ax1.set_ylabel(f"样本均值 ({ylabel})", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, result["n_subgroups"] + 0.5)

    # R chart
    ax2.plot(x, result["r_values"], "bo-", markersize=5, linewidth=1.5, label="极差 R")
    ax2.axhline(result["rbar"], color="green", linestyle="-", linewidth=1.5, label=f"CL={result['rbar']:.4f}")
    ax2.axhline(result["r_ucl"], color="red", linestyle="--", linewidth=1.5, label=f"UCL={result['r_ucl']:.4f}")
    if result["r_lcl"] > 0:
        ax2.axhline(result["r_lcl"], color="red", linestyle="--", linewidth=1.5, label=f"LCL={result['r_lcl']:.4f}")
    ax2.set_title("极差控制图", fontsize=14, fontweight="bold")
    ax2.set_xlabel("批次编号", fontsize=11)
    ax2.set_ylabel("样本极差", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, result["n_subgroups"] + 0.5)

    plt.tight_layout()
    return fig


def plot_imr(result, ylabel="测量值"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 0.7]})

    x = np.arange(1, result["n"] + 1)

    # I chart
    ax1.plot(x, result["values"], "bo-", markersize=5, linewidth=1.5, label="单值")
    ax1.axhline(result["mean"], color="green", linestyle="-", linewidth=1.5, label=f"CL={result['mean']:.4f}")
    ax1.axhline(result["i_ucl"], color="red", linestyle="--", linewidth=1.5, label=f"UCL={result['i_ucl']:.4f}")
    ax1.axhline(result["i_lcl"], color="red", linestyle="--", linewidth=1.5, label=f"LCL={result['i_lcl']:.4f}")
    ax1.set_title(f"单值控制图 (I-MR)", fontsize=14, fontweight="bold")
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, result["n"] + 0.5)

    # MR chart
    mr_x = np.arange(2, result["n"] + 1)
    ax2.plot(mr_x, result["mr"], "bo-", markersize=5, linewidth=1.5, label="移动极差")
    ax2.axhline(result["mrbar"], color="green", linestyle="-", linewidth=1.5, label=f"CL={result['mrbar']:.4f}")
    ax2.axhline(result["mr_ucl"], color="red", linestyle="--", linewidth=1.5, label=f"UCL={result['mr_ucl']:.4f}")
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.5, label="LCL=0")
    ax2.set_title("移动极差控制图", fontsize=14, fontweight="bold")
    ax2.set_xlabel("序号", fontsize=11)
    ax2.set_ylabel("移动极差", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1.5, result["n"] + 0.5)

    plt.tight_layout()
    return fig


# ==================================================
# session_state
# ==================================================
def _init_state():
    defaults = {
        "met_result": None, "met_chart_type": None,
        "met_auto_eval": None, "met_ai_answer": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v


# ==================================================
# 主页面
# ==================================================
def metrology_page():
    _init_state()
    render_copilot_sidebar()

    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">
        <div style="width:40px;height:40px;background:var(--pca-primary);border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-size:1.2rem;">M</div>
        <div>
            <div style="font-weight:700;font-size:1.2rem;color:var(--pca-text-primary);">常规计量控制图</div>
            <div style="font-size:0.8rem;color:var(--pca-text-secondary);">Variables Control Charts · Xbar-R · I-MR</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 图表类型选择
    chart_type = st.radio("图表类型", ["均值-极差图 (Xbar-R)", "单值-移动极差图 (I-MR)"], horizontal=True)

    # 数据源
    with st.container(border=True):
        data_source = st.radio("数据来源", ["上传文件", "使用示例数据"], horizontal=True, label_visibility="collapsed")
        df = None; subgroup_size = 1
        is_example = (data_source == "使用示例数据")

        if is_example:
            if "Xbar-R" in chart_type:
                df, subgroup_size = load_example_data("xbar_r")
                st.caption("已加载示例数据：活塞环直径，25 批次 × 5 个样本")
            else:
                df, subgroup_size = load_example_data("imr")
                st.caption("已加载示例数据：焊接强度，25 个单值测量")
        else:
            uploaded_file = st.file_uploader("上传 Excel 或 CSV 文件", type=["xlsx", "csv"], label_visibility="collapsed")
            if uploaded_file is None:
                st.info("请先上传数据文件，或切换到「使用示例数据」")
                return
            try:
                df = load_file(uploaded_file)
            except Exception as e:
                st.error(f"文件加载失败：{e}")
                return

    if df is None:
        return

    # 数据预览与参数设置
    st.markdown("##### 数据预览")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    st.caption(f"共 {len(df)} 行数据")

    # 分析按钮
    if st.button("执行分析", type="primary", use_container_width=True):
        with st.spinner("分析中..."):
            if "Xbar-R" in chart_type:
                if not is_example:
                    # 自动检测：多列宽格式还是单列长格式
                    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    if len(num_cols) >= 2:
                        data_matrix = df[num_cols].values
                        subgroup_size = len(num_cols)
                    else:
                        data_matrix = df[num_cols[0]].values.reshape(-1, 5)
                        subgroup_size = 5
                else:
                    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    data_matrix = df[num_cols].values

                result = compute_xbar_r(data_matrix, subgroup_size)
                st.session_state["met_result"] = result
                st.session_state["met_chart_type"] = "xbar_r"

                # 判异
                v, desc = check_control_violations(result["xbar_values"], result["xbar_ucl"], result["xbar_lcl"], result["xbarbar"], result["n_subgroups"])
                rv, rdesc = check_control_violations(result["r_values"], result["r_ucl"], result["r_lcl"], result["rbar"], result["n_subgroups"])
                all_violations = list(set(v + rv))
                all_desc = list(set(desc + rdesc))

                eval_lines = []
                if not all_violations and not all_desc:
                    eval_lines.append("##### ✅ 过程受控\n均值图和极差图均无异常点，过程处于统计受控状态。")
                else:
                    if all_violations:
                        eval_lines.append(f"##### ⚠️ 过程存在异常\n批次 {all_violations} 超出控制限，建议排查特殊原因。")
                    if all_desc:
                        for d in all_desc:
                            eval_lines.append(f"- {d}")
                eval_lines.append(f"**均值图**：CL={result['xbarbar']:.4f}, UCL={result['xbar_ucl']:.4f}, LCL={result['xbar_lcl']:.4f}")
                eval_lines.append(f"**极差图**：CL={result['rbar']:.4f}, UCL={result['r_ucl']:.4f}, LCL={result['r_lcl']:.4f}")
                st.session_state["met_auto_eval"] = "\n\n".join(eval_lines)

            else:  # I-MR
                if not is_example:
                    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    data = df[num_cols[0]].dropna().values
                else:
                    data = df["测量值"].values

                result = compute_imr(data)
                st.session_state["met_result"] = result
                st.session_state["met_chart_type"] = "imr"

                v, desc = check_control_violations(result["values"], result["i_ucl"], result["i_lcl"], result["mean"], result["n"])
                mv, mdesc = check_control_violations(np.insert(result["mr"], 0, 0), result["mr_ucl"], 0, result["mrbar"], result["n"])
                all_violations = list(set(v + [x-1 for x in mv if x > 1]))
                all_desc = list(set(desc + mdesc))

                eval_lines = []
                if not all_violations and not all_desc:
                    eval_lines.append("##### ✅ 过程受控\n单值图和移动极差图均无异常点，过程处于统计受控状态。")
                else:
                    if all_violations:
                        eval_lines.append(f"##### ⚠️ 过程存在异常\n点 {all_violations} 超出控制限，建议排查特殊原因。")
                    if all_desc:
                        for d in all_desc:
                            eval_lines.append(f"- {d}")
                eval_lines.append(f"**单值图**：CL={result['mean']:.4f}, UCL={result['i_ucl']:.4f}, LCL={result['i_lcl']:.4f}")
                eval_lines.append(f"**MR图**：CL={result['mrbar']:.4f}, UCL={result['mr_ucl']:.4f}")
                st.session_state["met_auto_eval"] = "\n\n".join(eval_lines)

        st.rerun()

    # 显示结果
    result = st.session_state.get("met_result")
    if result is None:
        return

    # KPI
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 关键指标概览")

    if st.session_state["met_chart_type"] == "xbar_r":
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(kpi_card("Xbarbar", f"{result['xbarbar']:.4f}", "总均值"), unsafe_allow_html=True)
        k2.markdown(kpi_card("Rbar", f"{result['rbar']:.4f}", "平均极差"), unsafe_allow_html=True)
        k3.markdown(kpi_card("Xbar UCL", f"{result['xbar_ucl']:.4f}", ""), unsafe_allow_html=True)
        k4.markdown(kpi_card("Xbar LCL", f"{result['xbar_lcl']:.4f}", ""), unsafe_allow_html=True)
        k5.markdown(kpi_card("R UCL", f"{result['r_ucl']:.4f}", ""), unsafe_allow_html=True)
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(kpi_card("均值", f"{result['mean']:.4f}", ""), unsafe_allow_html=True)
        k2.markdown(kpi_card("MRbar", f"{result['mrbar']:.4f}", "平均移动极差"), unsafe_allow_html=True)
        k3.markdown(kpi_card("I UCL", f"{result['i_ucl']:.4f}", ""), unsafe_allow_html=True)
        k4.markdown(kpi_card("I LCL", f"{result['i_lcl']:.4f}", ""), unsafe_allow_html=True)
        k5.markdown(kpi_card("MR UCL", f"{result['mr_ucl']:.4f}", ""), unsafe_allow_html=True)

    # 控制图
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 控制图")
    if st.session_state["met_chart_type"] == "xbar_r":
        st.pyplot(plot_xbar_r(result, ylabel="直径 (mm)" if is_example else "测量值"))
    else:
        st.pyplot(plot_imr(result, ylabel="焊接强度" if is_example else "测量值"))

    # 自动评价
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    auto_eval = st.session_state.get("met_auto_eval", "")
    if auto_eval:
        st.markdown(auto_eval)

    # 数据明细
    if st.session_state["met_chart_type"] == "xbar_r":
        detail_df = pd.DataFrame({
            "批次": range(1, result["n_subgroups"] + 1),
            "样本均值": result["xbar_values"].round(4),
            "样本极差": result["r_values"].round(4),
        })
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # AI 分析
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### AI 智能分析")
    ai_answer = st.session_state.get("met_ai_answer")
    if ai_answer:
        st.markdown(ai_answer)

    if st.button("生成 AI 分析报告", key="met_ai_btn"):
        if st.session_state["met_chart_type"] == "xbar_r":
            ctx = f"Xbar-R控制图结果：Xbarbar={result['xbarbar']:.4f}, Rbar={result['rbar']:.4f}, 批次={result['n_subgroups']}, 子组大小={result['subgroup_size']}"
        else:
            ctx = f"I-MR控制图结果：均值={result['mean']:.4f}, MRbar={result['mrbar']:.4f}, 观测数={result['n']}"
        try:
            answer = ask_ai("请评价该控制图并判断过程是否受控，给出改进建议。", ctx)
            st.session_state["met_ai_answer"] = answer
            st.rerun()
        except Exception as e:
            st.error(f"AI 调用失败：{e}")



    # PDF
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    try:
        chart_type = st.session_state["met_chart_type"]
        result = st.session_state["met_result"]
        if chart_type == "xbar_r":
            fig = plot_xbar_r(result, ylabel="测量值")
        else:
            fig = plot_imr(result, ylabel="测量值")
        pdf_bytes = generate_metrology_report(
            result=result, chart_type=chart_type,
            auto_eval=st.session_state.get("met_auto_eval", ""),
            ai_analysis=st.session_state.get("met_ai_answer", ""),
            fig_chart=fig,
        )
        st.download_button(
            label="导出 PDF 报告", data=pdf_bytes,
            file_name="计量控制图分析报告.pdf", mime="application/pdf",
            key="met_download_btn",
        )
    except Exception:
        st.warning("PDF 生成失败，请确保已安装 reportlab")


# --- 入口 ---
metrology_page()
