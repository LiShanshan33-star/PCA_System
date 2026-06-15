import streamlit as st
st.set_page_config(page_title="小批量控制图", page_icon="📐", layout="wide")
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
# 示例数据
# ==================================================
def load_example_data(exp_type):
    if exp_type == "target":
        # 实验四 例1：目标控制图 (Table 9) - 4种型号零件
        records = [
            ("A", 8.00, 7.96), ("A", 8.00, 7.99), ("A", 8.00, 8.01),
            ("A", 8.00, 8.00), ("A", 8.00, 8.02), ("A", 8.00, 8.01),
            ("B", 3.00, 3.01), ("B", 3.00, 2.99), ("B", 3.00, 2.98),
            ("B", 3.00, 3.03), ("B", 3.00, 2.97),
            ("C", 15.00, 15.01), ("C", 15.00, 15.02), ("C", 15.00, 14.99),
            ("C", 15.00, 14.98),
            ("D", 21.00, 20.99), ("D", 21.00, 21.01), ("D", 21.00, 21.02),
        ]
        df = pd.DataFrame(records, columns=["型号", "目标值", "测量值"])
        df["变换值"] = df["测量值"] - df["目标值"]
        return df, "target"
    elif exp_type == "ratio":
        # 实验四 例2：比例控制图 (Table 10) - 喷漆厚度
        records = [
            ("A", 7.50, [6.50, 8.00, 6.60]),
            ("A", 7.50, [7.00, 7.80, 8.40]),
            ("A", 7.50, [7.70, 7.80, 8.30]),
            ("A", 7.50, [7.30, 7.70, 8.40]),
            ("A", 7.50, [7.90, 8.00, 8.30]),
            ("B", 0.55, [0.52, 0.51, 0.53]),
            ("B", 0.55, [0.50, 0.52, 0.52]),
            ("B", 0.55, [0.51, 0.53, 0.54]),
            ("B", 0.55, [0.52, 0.52, 0.56]),
            ("B", 0.55, [0.53, 0.55, 0.56]),
            ("B", 0.55, [0.54, 0.55, 0.56]),
            ("B", 0.55, [0.55, 0.56, 0.55]),
            ("B", 0.55, [0.52, 0.54, 0.56]),
            ("C", 3.25, [3.20, 3.40, 3.30]),
            ("C", 3.25, [3.30, 3.20, 3.20]),
            ("C", 3.25, [3.50, 3.40, 3.30]),
            ("C", 3.25, [3.20, 3.50, 3.30]),
            ("C", 3.25, [3.40, 3.30, 3.40]),
        ]
        # Note: Table 10 has target values multiplied by 10: 75.00, 5.50, 32.50
        records_scaled = [
            ("A", 75.00, [65.00, 80.00, 66.00]),
            ("A", 75.00, [70.00, 78.00, 84.00]),
            ("A", 75.00, [77.00, 78.00, 83.00]),
            ("A", 75.00, [73.00, 77.00, 84.00]),
            ("A", 75.00, [79.00, 80.00, 83.00]),
            ("B", 5.50, [5.20, 5.10, 5.30]),
            ("B", 5.50, [5.00, 5.20, 5.20]),
            ("B", 5.50, [5.10, 5.30, 5.40]),
            ("B", 5.50, [5.20, 5.20, 5.60]),
            ("B", 5.50, [5.30, 5.50, 5.60]),
            ("B", 5.50, [5.40, 5.50, 5.60]),
            ("B", 5.50, [5.50, 5.60, 5.50]),
            ("B", 5.50, [5.20, 5.40, 5.60]),
            ("C", 32.50, [32.00, 34.00, 33.00]),
            ("C", 32.50, [33.00, 32.00, 32.00]),
            ("C", 32.50, [35.00, 34.00, 33.00]),
            ("C", 32.50, [32.00, 35.00, 33.00]),
            ("C", 32.50, [34.00, 33.00, 34.00]),
        ]
        rows = []
        for model, target, vals in records_scaled:
            for v in vals:
                rows.append({"型号": model, "目标值": target, "测量值": v, "比例值": v / target})
        df = pd.DataFrame(rows)
        return df, "ratio"
    elif exp_type == "standardized":
        # 实验四 例3：标准变换控制图 (Table 11) - 舱门间隙
        # Already transformed data
        records = [
            ("1号舱门", 1, -0.367, 1.789),
            ("1号舱门", 2, -0.575, 1.973),
            ("1号舱门", 3, 0.123, 1.876),
            ("1号舱门", 4, 0.551, 1.647),
            ("1号舱门", 5, -0.711, 0.550),
            ("1号舱门", 6, 0.069, 1.697),
            ("1号舱门", 7, 0.505, 4.125),
            ("1号舱门", 8, -0.046, 1.193),
            ("1号舱门", 9, 0.379, 2.618),
            ("1号舱门", 10, 0.723, 2.753),
            ("1号舱门", 11, -0.157, 3.381),
            ("1号舱门", 12, -0.872, 1.192),
            ("2号舱门", 1, 1.739, 1.885),
            ("2号舱门", 2, -0.279, 2.842),
            ("2号舱门", 3, 0.133, 2.331),
            ("2号舱门", 4, 0.141, 3.374),
            ("2号舱门", 5, -0.133, 1.455),
            ("2号舱门", 6, 0.089, 1.338),
            ("2号舱门", 7, -0.069, 1.929),
            ("2号舱门", 8, -0.459, 1.892),
            ("2号舱门", 9, -0.007, 1.274),
            ("2号舱门", 10, -0.208, 10.381),
            ("2号舱门", 11, -0.153, 1.079),
            ("2号舱门", 12, 0.321, 1.237),
        ]
        df = pd.DataFrame(records, columns=["舱门", "间隙序号", "样本均值", "样本极差"])
        return df, "standardized"
    return None, None


# ==================================================
# I-MR 计算
# ==================================================
def compute_imr_simple(data):
    n = len(data)
    mr = np.abs(np.diff(data))
    mrbar = np.mean(mr)
    mean_val = np.mean(data)
    i_ucl = mean_val + 2.66 * mrbar
    i_lcl = mean_val - 2.66 * mrbar
    mr_ucl = 3.267 * mrbar
    return {"values": data, "mr": mr, "mean": mean_val, "mrbar": mrbar,
            "i_ucl": i_ucl, "i_lcl": i_lcl, "mr_ucl": mr_ucl, "n": n}


# ==================================================
# 绘图
# ==================================================
def plot_target_chart(result, title="目标控制图"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 0.7]})

    x = np.arange(1, result["n"] + 1)

    ax1.plot(x, result["values"], "bo-", markersize=5, linewidth=1.5, label="变换值(Y)")
    ax1.axhline(result["mean"], color="green", linestyle="-", linewidth=1.5, label=f"CL={result['mean']:.4f}")
    ax1.axhline(result["i_ucl"], color="red", linestyle="--", linewidth=1.5, label=f"UCL={result['i_ucl']:.4f}")
    ax1.axhline(result["i_lcl"], color="red", linestyle="--", linewidth=1.5, label=f"LCL={result['i_lcl']:.4f}")
    ax1.set_title("偏差单值控制图", fontsize=13, fontweight="bold")
    ax1.set_ylabel("偏差 (测量值-目标值)", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    mr_x = np.arange(2, result["n"] + 1)
    ax2.plot(mr_x, result["mr"], "bo-", markersize=5, linewidth=1.5, label="移动极差")
    ax2.axhline(result["mrbar"], color="green", linestyle="-", linewidth=1.5, label=f"CL={result['mrbar']:.4f}")
    ax2.axhline(result["mr_ucl"], color="red", linestyle="--", linewidth=1.5, label=f"UCL={result['mr_ucl']:.4f}")
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.5, label="LCL=0")
    ax2.set_title("移动极差控制图", fontsize=13, fontweight="bold")
    ax2.set_xlabel("样本序号", fontsize=11)
    ax2.set_ylabel("移动极差", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ratio_chart(df, title="比例控制图"):
    """Xbar-R chart on ratio-transformed data"""
    models = df["型号"].unique()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 0.7]})

    subgroup_means = []
    subgroup_ranges = []
    for model in models:
        sub = df[df["型号"] == model]
        for idx in sub.groupby(sub.index // 3):
            grp = idx[1]
            if len(grp) > 0:
                subgroup_means.append(grp["比例值"].mean())
                subgroup_ranges.append(grp["比例值"].max() - grp["比例值"].min())

    subgroup_means = np.array(subgroup_means)
    subgroup_ranges = np.array(subgroup_ranges)
    n_subgroups = len(subgroup_means)
    xbarbar = np.mean(subgroup_means)
    rbar = np.mean(subgroup_ranges)

    # A2 for n=3
    A2 = 1.023
    D3 = 0
    D4 = 2.574

    xbar_ucl = xbarbar + A2 * rbar
    xbar_lcl = xbarbar - A2 * rbar
    r_ucl = D4 * rbar

    x = np.arange(1, n_subgroups + 1)
    ax1.plot(x, subgroup_means, "bo-", markersize=5, linewidth=1.5, label="比例均值")
    ax1.axhline(xbarbar, color="green", linestyle="-", linewidth=1.5, label=f"CL={xbarbar:.4f}")
    ax1.axhline(xbar_ucl, color="red", linestyle="--", linewidth=1.5, label=f"UCL={xbar_ucl:.4f}")
    ax1.axhline(xbar_lcl, color="red", linestyle="--", linewidth=1.5, label=f"LCL={xbar_lcl:.4f}")
    ax1.set_title("比例值均值控制图", fontsize=13, fontweight="bold")
    ax1.set_ylabel("比例均值", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, subgroup_ranges, "bo-", markersize=5, linewidth=1.5, label="比例极差")
    ax2.axhline(rbar, color="green", linestyle="-", linewidth=1.5, label=f"CL={rbar:.4f}")
    ax2.axhline(r_ucl, color="red", linestyle="--", linewidth=1.5, label=f"UCL={r_ucl:.4f}")
    ax2.set_title("比例值极差控制图", fontsize=13, fontweight="bold")
    ax2.set_xlabel("子组编号", fontsize=11)
    ax2.set_ylabel("比例极差", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_standardized_chart(df, col_name, title):
    """I-MR on standardized data"""
    data = df[col_name].values
    result = compute_imr_simple(data)
    return plot_target_chart(result, title)


# ==================================================
# session_state
# ==================================================
def _init_state():
    defaults = {
        "sb_result": None, "sb_chart_type": None,
        "sb_auto_eval": None, "sb_ai_answer": None,
        "sb_data": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v


# ==================================================
# 主页面
# ==================================================
def small_batch_page():
    _init_state()
    render_copilot_sidebar()

    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">
        <div style="width:40px;height:40px;background:var(--pca-primary);border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-size:1.2rem;">S</div>
        <div>
            <div style="font-weight:700;font-size:1.2rem;color:var(--pca-text-primary);">小批量控制图技术</div>
            <div style="font-size:0.8rem;color:var(--pca-text-secondary);">Small Batch SPC · 目标图 · 比例图 · 标准变换图</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 图表类型选择
    chart_type = st.selectbox(
        "图表类型",
        ["目标控制图 (Target Chart)", "比例控制图 (Ratio Chart)", "标准变换控制图 (Standardized Chart)"]
    )

    example_map = {"目标控制图 (Target Chart)": "target", "比例控制图 (Ratio Chart)": "ratio", "标准变换控制图 (Standardized Chart)": "standardized"}

    with st.container(border=True):
        data_source = st.radio("数据来源", ["上传文件", "使用示例数据"], horizontal=True, label_visibility="collapsed")
        df = None

        labels = {
            "target": "已加载示例数据：4种型号零件，18个测量值（目标值：8.00/3.00/15.00/21.00）",
            "ratio": "已加载示例数据：3种工件喷漆，18个子组（n=3），比例变换",
            "standardized": "已加载示例数据：波音747舱门，24个间隙（1号和2号舱门，已标准变换）",
        }

        if data_source == "使用示例数据":
            exp_key = example_map[chart_type]
            df, _ = load_example_data(exp_key)
            st.session_state["sb_data"] = df
            st.session_state["sb_chart_type"] = exp_key
            st.caption(labels.get(exp_key, ""))
        else:
            uploaded_file = st.file_uploader("上传 Excel 或 CSV 文件", type=["xlsx", "csv"], label_visibility="collapsed")
            if uploaded_file is None:
                st.info("请先上传数据文件，或切换到「使用示例数据」")
                return
            try:
                df = load_file(uploaded_file)
                st.session_state["sb_data"] = df
            except Exception as e:
                st.error(f"文件加载失败：{e}")
                return

    if df is None:
        return

    st.markdown("##### 数据预览")
    st.dataframe(df.head(15), use_container_width=True, hide_index=True)
    st.caption(f"共 {len(df)} 行")

    # 分析
    if st.button("执行分析", type="primary", use_container_width=True):
        ct = st.session_state.get("sb_chart_type", example_map[chart_type])

        if ct == "target":
            if data_source != "使用示例数据":
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if len(num_cols) >= 2:
                    target_vals = df[num_cols[0]].values
                    measure_vals = df[num_cols[1]].values
                else:
                    st.error("需要目标值和测量值两列")
                    return
            else:
                target_vals = df["目标值"].values
                measure_vals = df["测量值"].values
            y_vals = measure_vals - target_vals
            result = compute_imr_simple(y_vals)
            st.session_state["sb_result"] = result
            st.session_state["sb_chart_type"] = "target"

            v = [i+1 for i in range(len(y_vals)) if y_vals[i] > result["i_ucl"] or y_vals[i] < result["i_lcl"]]
            mr_padded = np.insert(result["mr"], 0, 0)
            mv = [i for i in range(1, len(mr_padded)) if mr_padded[i] > result["mr_ucl"]]
            eval_lines = []
            if not v and not mv:
                eval_lines.append("##### ✅ 过程受控\n目标控制图无异常点，加工过程处于统计受控状态。")
            else:
                if v:
                    eval_lines.append(f"##### ⚠️ I图异常\n样本 {v} 超出控制限。")
                if mv:
                    eval_lines.append(f"MR图异常点：{mv}")
            eval_lines.append(f"变换值均值={result['mean']:.4f}, MRbar={result['mrbar']:.4f}")
            st.session_state["sb_auto_eval"] = "\n\n".join(eval_lines)

        elif ct == "ratio":
            if data_source != "使用示例数据":
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                st.info("请确保数据包含型号/目标值/测量值列。上传模式下使用比例图需要预处理数据。")
                return
            st.session_state["sb_chart_type"] = "ratio"
            eval_lines = ["##### 比例控制图\n对测量值除以目标值后，构造Xbar-R图进行分析。"]
            st.session_state["sb_auto_eval"] = "\n\n".join(eval_lines)

        elif ct == "standardized":
            if data_source != "使用示例数据":
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if len(num_cols) >= 2:
                    mean_vals = df[num_cols[0]].values
                    range_vals = df[num_cols[1]].values
                else:
                    st.error("需要样本均值和样本极差两列")
                    return
            else:
                mean_vals = df["样本均值"].values
                range_vals = df["样本极差"].values

            mean_result = compute_imr_simple(mean_vals)
            range_result = compute_imr_simple(range_vals)
            st.session_state["sb_result"] = {"mean": mean_result, "range": range_result}
            st.session_state["sb_chart_type"] = "standardized"

            v = [i+1 for i in range(len(mean_vals)) if mean_vals[i] > mean_result["i_ucl"] or mean_vals[i] < mean_result["i_lcl"]]
            rv = [i+1 for i in range(len(range_vals)) if range_vals[i] > range_result["i_ucl"] or range_vals[i] < range_result["i_lcl"]]
            eval_lines = []
            if not v and not rv:
                eval_lines.append("##### ✅ 过程受控\n样本均值和极差控制图均无异常点，加工过程处于统计受控状态。")
            else:
                if v:
                    eval_lines.append(f"##### ⚠️ 均值图异常\n样本 {v} 超出控制限。")
                if rv:
                    eval_lines.append(f"极差图异常：{rv}")
            eval_lines.append(f"均值图 CL={mean_result['mean']:.4f}, UCL={mean_result['i_ucl']:.4f}, LCL={mean_result['i_lcl']:.4f}")
            eval_lines.append(f"极差图 CL={range_result['mean']:.4f}, UCL={range_result['i_ucl']:.4f}")
            st.session_state["sb_auto_eval"] = "\n\n".join(eval_lines)

        st.rerun()

    # 显示结果
    result = st.session_state.get("sb_result")
    if result is None:
        return

    ct = st.session_state["sb_chart_type"]

    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 控制图")

    if ct == "target":
        st.pyplot(plot_target_chart(result, "目标控制图（偏差 I-MR 图）"))
    elif ct == "ratio":
        df = st.session_state.get("sb_data")
        if df is not None:
            st.pyplot(plot_ratio_chart(df, "比例控制图"))
    elif ct == "standardized":
        st.markdown("**样本均值控制图**")
        st.pyplot(plot_target_chart(result["mean"], "样本均值单值控制图"))
        st.markdown("**样本极差控制图**")
        st.pyplot(plot_target_chart(result["range"], "样本极差单值控制图"))

    # 自动评价
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    auto_eval = st.session_state.get("sb_auto_eval", "")
    if auto_eval:
        st.markdown(auto_eval)

    # AI 分析
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### AI 智能分析")
    ai_answer = st.session_state.get("sb_ai_answer")
    if ai_answer:
        st.markdown(ai_answer)

    if st.button("生成 AI 分析报告", key="sb_ai_btn"):
        ctx = f"小批量控制图({ct})分析结果：" + (str(result)[:500] if result else "")
        try:
            answer = ask_ai("请评价该小批量控制图并判断过程是否受控。", ctx)
            st.session_state["sb_ai_answer"] = answer
            st.rerun()
        except Exception as e:
            st.error(f"AI 调用失败：{e}")



    # PDF
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    try:
        ct = st.session_state.get("sb_chart_type", "")
        result = st.session_state.get("sb_result")
        fig, fig2 = None, None
        if ct == "target" and result is not None:
            fig = plot_target_chart(result, "目标控制图")
        elif ct == "standardized" and isinstance(result, dict):
            if "mean" in result:
                fig = plot_target_chart(result["mean"], "样本均值控制图")
            if "range" in result:
                fig2 = plot_target_chart(result["range"], "样本极差控制图")
        elif ct == "ratio":
            df = st.session_state.get("sb_data")
            if df is not None:
                fig = plot_ratio_chart(df, "比例控制图")
        pdf_bytes = generate_small_batch_report(
            chart_type=ct,
            auto_eval=st.session_state.get("sb_auto_eval", ""),
            ai_analysis=st.session_state.get("sb_ai_answer", ""),
            fig_chart=fig, fig_chart2=fig2,
        )
        st.download_button(
            label="导出 PDF 报告", data=pdf_bytes,
            file_name="小批量控制图分析报告.pdf", mime="application/pdf",
            key="sb_download_btn",
        )
    except Exception:
        st.warning("PDF 生成失败，请确保已安装 reportlab")


# --- 入口 ---
small_batch_page()
