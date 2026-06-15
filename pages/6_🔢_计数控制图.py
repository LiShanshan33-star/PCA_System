import streamlit as st
st.set_page_config(page_title="常规计数控制图", page_icon="🔢", layout="wide")
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
    if exp_type == "np_chart":
        # 实验三 例1：二极管不合格品 NP图 (Table 5, 固定样本量150)
        defectives = [3, 4, 6, 8, 6, 6, 3, 4, 8, 7, 9, 6, 12, 3, 7, 10, 6, 3, 7, 4, 5, 4, 12, 11, 7, 8, 8, 7, 7, 5]
        samples = [150] * 30
        return pd.DataFrame({"样品数量": samples, "不合格品数量": defectives}), "np"
    elif exp_type == "p_chart":
        # 实验三 例2：二极管不合格品 P图 (Table 6, 不等样本量)
        defectives = [3, 4, 6, 8, 6, 6, 3, 4, 8, 7, 9, 6, 12, 3, 7, 10, 6, 3, 7, 4, 5, 4, 12, 11, 7, 8, 8, 7, 7, 5]
        samples = [120, 155, 148, 150, 165, 145, 158, 199, 164, 133, 110, 165, 173, 196, 126, 156, 162, 154, 102, 136, 159, 125, 173, 113, 176, 199, 146, 131, 158, 142]
        return pd.DataFrame({"样品数量": samples, "不合格品数量": defectives}), "p"
    elif exp_type == "u_chart":
        # 实验三 例3：芯片缺陷 U图 (Table 7, 固定样本量10)
        defects = [26, 28, 23, 18, 32, 40, 35, 24, 18, 28, 16, 14, 24, 35, 30]
        samples = [10] * 15
        return pd.DataFrame({"样品数量": samples, "缺陷数": defects}), "u"
    elif exp_type == "c_chart":
        # 实验三 例3同数据：芯片缺陷 C图
        defects = [26, 28, 23, 18, 32, 40, 35, 24, 18, 28, 16, 14, 24, 35, 30]
        samples = [10] * 15
        return pd.DataFrame({"样品数量": samples, "缺陷数": defects}), "c"
    elif exp_type == "u_chart_varying":
        # 实验三 例4：芯片缺陷 U图 (Table 8, 不等样本量)
        defects = [26, 28, 23, 18, 32, 40, 35, 24, 18, 28, 16, 14, 24, 35, 30]
        samples = [8, 10, 12, 10, 11, 15, 10, 9, 6, 13, 12, 9, 14, 9, 10]
        return pd.DataFrame({"样品数量": samples, "缺陷数": defects}), "uv"
    return None, None


# ==================================================
# 控制图计算
# ==================================================
def compute_p_chart(samples, defectives):
    """P 控制图 (样本量可不等)"""
    n = len(samples)
    p_i = defectives / samples
    p_bar = np.sum(defectives) / np.sum(samples)
    ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / samples)
    lcl = np.maximum(0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / samples))
    return {"p_i": p_i, "p_bar": p_bar, "ucl": ucl, "lcl": lcl, "n": n, "samples": samples}


def compute_np_chart(samples, defectives):
    """NP 控制图 (固定样本量)"""
    n = len(samples)
    n_size = samples[0]
    p_bar = np.sum(defectives) / (n * n_size)
    np_bar = n_size * p_bar
    ucl = np_bar + 3 * np.sqrt(np_bar * (1 - p_bar))
    lcl = max(0, np_bar - 3 * np.sqrt(np_bar * (1 - p_bar)))
    return {"np_values": defectives, "np_bar": np_bar, "ucl": np.full(n, ucl), "lcl": np.full(n, lcl), "n": n, "n_size": n_size, "p_bar": p_bar}


def compute_u_chart(samples, defects):
    """U 控制图 (单位缺陷数)"""
    n = len(samples)
    u_i = defects / samples
    u_bar = np.sum(defects) / np.sum(samples)
    ucl = u_bar + 3 * np.sqrt(u_bar / samples)
    lcl = np.maximum(0, u_bar - 3 * np.sqrt(u_bar / samples))
    return {"u_i": u_i, "u_bar": u_bar, "ucl": ucl, "lcl": lcl, "n": n, "samples": samples}


def compute_c_chart(defects):
    """C 控制图 (固定样本量下的缺陷数)"""
    n = len(defects)
    c_bar = np.mean(defects)
    ucl = c_bar + 3 * np.sqrt(c_bar)
    lcl = max(0, c_bar - 3 * np.sqrt(c_bar))
    return {"c_values": defects, "c_bar": c_bar, "ucl": ucl, "lcl": lcl, "n": n}


# ==================================================
# 绘图
# ==================================================
def plot_p_chart_attrs(result, title="P 控制图"):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(1, result["n"] + 1)
    ax.plot(x, result["p_i"], "bo-", markersize=5, linewidth=1.5, label="不合格率")
    ax.axhline(result["p_bar"], color="green", linestyle="-", linewidth=1.5, label=f"CL=p̄={result['p_bar']:.4f}")
    ax.step(x, result["ucl"], "r--", where="mid", linewidth=1.5, label="UCL")
    ax.step(x, result["lcl"], "r--", where="mid", linewidth=1.5, label="LCL")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("批次编号", fontsize=11)
    ax.set_ylabel("不合格率", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, result["n"] + 0.5)
    plt.tight_layout()
    return fig


def plot_np_chart_attrs(result, title="NP 控制图"):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(1, result["n"] + 1)
    ax.plot(x, result["np_values"], "bo-", markersize=5, linewidth=1.5, label="不合格品数")
    ax.axhline(result["np_bar"], color="green", linestyle="-", linewidth=1.5, label=f"CL={result['np_bar']:.2f}")
    ax.plot(x, result["ucl"], "r--", linewidth=1.5, label=f"UCL={result['ucl'][0]:.2f}")
    ax.plot(x, result["lcl"], "r--", linewidth=1.5, label=f"LCL={result['lcl'][0]:.2f}")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("批次编号", fontsize=11)
    ax.set_ylabel("不合格品数", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, result["n"] + 0.5)
    plt.tight_layout()
    return fig


def plot_u_chart_attrs(result, title="U 控制图"):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(1, result["n"] + 1)
    ax.plot(x, result["u_i"], "bo-", markersize=5, linewidth=1.5, label="单位缺陷数")
    ax.axhline(result["u_bar"], color="green", linestyle="-", linewidth=1.5, label=f"CL=ū={result['u_bar']:.4f}")
    ax.step(x, result["ucl"], "r--", where="mid", linewidth=1.5, label="UCL")
    ax.step(x, result["lcl"], "r--", where="mid", linewidth=1.5, label="LCL")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("批次编号", fontsize=11)
    ax.set_ylabel("单位缺陷数", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, result["n"] + 0.5)
    plt.tight_layout()
    return fig


def plot_c_chart_attrs(result, title="C 控制图"):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(1, result["n"] + 1)
    ax.plot(x, result["c_values"], "bo-", markersize=5, linewidth=1.5, label="缺陷数")
    ax.axhline(result["c_bar"], color="green", linestyle="-", linewidth=1.5, label=f"CL=c̄={result['c_bar']:.2f}")
    ax.axhline(result["ucl"], color="red", linestyle="--", linewidth=1.5, label=f"UCL={result['ucl']:.2f}")
    ax.axhline(result["lcl"], color="red", linestyle="--", linewidth=1.5, label=f"LCL={result['lcl']:.2f}")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("批次编号", fontsize=11)
    ax.set_ylabel("缺陷数", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, result["n"] + 0.5)
    plt.tight_layout()
    return fig


# ==================================================
# session_state
# ==================================================
def _init_state():
    defaults = {
        "attr_result": None, "attr_chart_type": None,
        "attr_auto_eval": None, "attr_ai_answer": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v


# ==================================================
# 主页面
# ==================================================
def attributes_page():
    _init_state()
    render_copilot_sidebar()

    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">
        <div style="width:40px;height:40px;background:var(--pca-primary);border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-size:1.2rem;">A</div>
        <div>
            <div style="font-weight:700;font-size:1.2rem;color:var(--pca-text-primary);">常规计数控制图</div>
            <div style="font-size:0.8rem;color:var(--pca-text-secondary);">Attributes Control Charts · P · NP · U · C</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 图表类型选择
    chart_type = st.selectbox(
        "图表类型",
        ["NP 控制图 (固定样本量)", "P 控制图 (不等样本量)", "C 控制图 (固定样本量)", "U 控制图 (固定样本量)", "U 控制图 (不等样本量)"]
    )

    # 示例类型映射
    example_map = {
        "NP 控制图 (固定样本量)": "np_chart",
        "P 控制图 (不等样本量)": "p_chart",
        "C 控制图 (固定样本量)": "c_chart",
        "U 控制图 (固定样本量)": "u_chart",
        "U 控制图 (不等样本量)": "u_chart_varying",
    }

    # 数据源
    with st.container(border=True):
        data_source = st.radio("数据来源", ["上传文件", "使用示例数据"], horizontal=True, label_visibility="collapsed")
        df = None

        labels = {
            "np_chart": "已加载示例数据：二极管不合格品，30天，每日抽检150个",
            "p_chart": "已加载示例数据：二极管不合格品，30天，每日抽检量不等",
            "c_chart": "已加载示例数据：芯片缺陷，15天，每日抽检10片",
            "u_chart": "已加载示例数据：芯片缺陷，15天，每日抽检10片",
            "u_chart_varying": "已加载示例数据：芯片缺陷，15天，每日抽检量不等",
        }

        if data_source == "使用示例数据":
            exp_key = example_map[chart_type]
            df, _ = load_example_data(exp_key)
            st.caption(labels.get(exp_key, ""))
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

    # 数据预览
    st.markdown("##### 数据预览")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    st.caption(f"共 {len(df)} 行")

    # 列选择（上传模式）
    if data_source != "使用示例数据":
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) < 2:
            st.error("需要至少2列数值数据（样品数量和缺陷数/不合格数）")
            return
        sample_col = st.selectbox("样品数量列", num_cols, index=0)
        defect_col = st.selectbox("缺陷/不合格数列", num_cols, index=min(1, len(num_cols)-1))

    # 分析按钮
    if st.button("执行分析", type="primary", use_container_width=True):
        if data_source == "使用示例数据":
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) >= 2:
                sample_col, defect_col = num_cols[0], num_cols[1]
            else:
                st.error("数据格式错误")
                return

        samples = df[sample_col].values.astype(float)
        defects = df[defect_col].values.astype(float)

        result = None
        if "NP" in chart_type:
            result = compute_np_chart(samples, defects)
            st.session_state["attr_chart_type"] = "np"
        elif "P" in chart_type:
            result = compute_p_chart(samples, defects)
            st.session_state["attr_chart_type"] = "p"
        elif "C" in chart_type:
            result = compute_c_chart(defects)
            st.session_state["attr_chart_type"] = "c"
        else:  # U
            result = compute_u_chart(samples, defects)
            st.session_state["attr_chart_type"] = "u"

        st.session_state["attr_result"] = result
        st.session_state["attr_ai_answer"] = None

        # 自动评价
        eval_lines = []
        chart_type_name = st.session_state["attr_chart_type"]

        if chart_type_name == "np":
            violations = []
            for i in range(result["n"]):
                if result["np_values"][i] > result["ucl"][i] or result["np_values"][i] < result["lcl"][i]:
                    violations.append(i + 1)
            if not violations:
                eval_lines.append("##### ✅ 过程受控\n所有点均在控制限内，生产过程不合格品数稳定。")
            else:
                eval_lines.append(f"##### ⚠️ 存在异常\n第 {violations} 批超出控制限。")
            eval_lines.append(f"**NP 控制图**：CL={result['np_bar']:.2f}, UCL={result['ucl'][0]:.2f}, LCL={result['lcl'][0]:.2f}, p̄={result['p_bar']:.4%}")

        elif chart_type_name == "p":
            violations = []
            for i in range(result["n"]):
                if result["p_i"][i] > result["ucl"][i] or result["p_i"][i] < result["lcl"][i]:
                    violations.append(i + 1)
            if not violations:
                eval_lines.append("##### ✅ 过程受控\n所有点均在控制限内（考虑样本量变化），不合格率稳定。")
            else:
                eval_lines.append(f"##### ⚠️ 存在异常\n第 {violations} 批超出控制限。")
            eval_lines.append(f"**P 控制图**：p̄={result['p_bar']:.4%}")

        elif chart_type_name == "c":
            violations = []
            for i in range(result["n"]):
                if result["c_values"][i] > result["ucl"] or result["c_values"][i] < result["lcl"]:
                    violations.append(i + 1)
            if not violations:
                eval_lines.append("##### ✅ 过程受控\n所有点均在控制限内，缺陷数稳定。")
            else:
                eval_lines.append(f"##### ⚠️ 存在异常\n第 {violations} 批超出控制限。")
            eval_lines.append(f"**C 控制图**：c̄={result['c_bar']:.2f}, UCL={result['ucl']:.2f}, LCL={result['lcl']:.2f}")

        else:  # u
            violations = []
            for i in range(result["n"]):
                if result["u_i"][i] > result["ucl"][i] or result["u_i"][i] < result["lcl"][i]:
                    violations.append(i + 1)
            if not violations:
                eval_lines.append("##### ✅ 过程受控\n所有点均在控制限内，单位缺陷数稳定。")
            else:
                eval_lines.append(f"##### ⚠️ 存在异常\n第 {violations} 批超出控制限。")
            eval_lines.append(f"**U 控制图**：ū={result['u_bar']:.4f}")

        st.session_state["attr_auto_eval"] = "\n\n".join(eval_lines)
        st.rerun()

    # 显示结果
    result = st.session_state.get("attr_result")
    if result is None:
        return

    ct = st.session_state["attr_chart_type"]

    # KPI
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 关键指标概览")

    if ct == "np":
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(kpi_card("平均NP", f"{result['np_bar']:.2f}", ""), unsafe_allow_html=True)
        k2.markdown(kpi_card("UCL", f"{result['ucl'][0]:.2f}", ""), unsafe_allow_html=True)
        k3.markdown(kpi_card("LCL", f"{result['lcl'][0]:.2f}", ""), unsafe_allow_html=True)
        k4.markdown(kpi_card("p̄", f"{result['p_bar']:.4%}", ""), unsafe_allow_html=True)
        k5.markdown(kpi_card("n", f"{result['n_size']}", "子组大小"), unsafe_allow_html=True)
    elif ct == "p":
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(kpi_card("p̄", f"{result['p_bar']:.4%}", "平均不合格率"), unsafe_allow_html=True)
        k2.markdown(kpi_card("UCL范围", f"{result['ucl'].min():.4f}~{result['ucl'].max():.4f}", ""), unsafe_allow_html=True)
        k3.markdown(kpi_card("LCL范围", f"{result['lcl'].min():.4f}~{result['lcl'].max():.4f}", ""), unsafe_allow_html=True)
    elif ct == "c":
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(kpi_card("c̄", f"{result['c_bar']:.2f}", "平均缺陷数"), unsafe_allow_html=True)
        k2.markdown(kpi_card("UCL", f"{result['ucl']:.2f}", ""), unsafe_allow_html=True)
        k3.markdown(kpi_card("LCL", f"{result['lcl']:.2f}", ""), unsafe_allow_html=True)
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(kpi_card("ū", f"{result['u_bar']:.4f}", "平均单位缺陷数"), unsafe_allow_html=True)
        k2.markdown(kpi_card("UCL范围", f"{result['ucl'].min():.4f}~{result['ucl'].max():.4f}", ""), unsafe_allow_html=True)
        k3.markdown(kpi_card("LCL范围", f"{result['lcl'].min():.4f}~{result['lcl'].max():.4f}", ""), unsafe_allow_html=True)

    # 控制图
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 控制图")

    titles = {"np": "NP 控制图 (不合格品数)", "p": "P 控制图 (不合格率)", "c": "C 控制图 (缺陷数)", "u": "U 控制图 (单位缺陷数)"}
    title = titles.get(ct, "控制图")

    if ct == "np":
        st.pyplot(plot_np_chart_attrs(result, title))
    elif ct == "p":
        st.pyplot(plot_p_chart_attrs(result, title))
    elif ct == "c":
        st.pyplot(plot_c_chart_attrs(result, title))
    else:
        st.pyplot(plot_u_chart_attrs(result, title))

    # 自动评价
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    auto_eval = st.session_state.get("attr_auto_eval", "")
    if auto_eval:
        st.markdown(auto_eval)

    # 数据明细
    if ct in ("p", "u"):
        rate_label = "不合格率" if ct == "p" else "单位缺陷数"
        rate_vals = result["p_i"] if ct == "p" else result["u_i"]
        detail_df = pd.DataFrame({
            "批次": range(1, result["n"] + 1),
            "样品数量": result["samples"],
            "缺陷/不合格数": defects if 'defects' in dir() else (result.get("np_values", []) if ct == "np" else result.get("c_values", [])),
            rate_label: rate_vals.round(4),
        })
    else:
        val_key = "np_values" if ct == "np" else "c_values"
        detail_df = pd.DataFrame({
            "批次": range(1, result["n"] + 1),
            "值": result[val_key],
        })
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # AI 分析
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### AI 智能分析")
    ai_answer = st.session_state.get("attr_ai_answer")
    if ai_answer:
        st.markdown(ai_answer)

    if st.button("生成 AI 分析报告", key="attr_ai_btn"):
        ctx = f"{titles.get(ct)}结果：" + str({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items() if k in ("p_bar", "np_bar", "c_bar", "u_bar", "n")})
        try:
            answer = ask_ai("请评价该计数型控制图并判断过程是否受控，给出改进建议。", ctx)
            st.session_state["attr_ai_answer"] = answer
            st.rerun()
        except Exception as e:
            st.error(f"AI 调用失败：{e}")



    # PDF
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    try:
        ct = st.session_state["attr_chart_type"]
        result = st.session_state["attr_result"]
        if ct == "np":
            fig = plot_np_chart_attrs(result, "NP 控制图")
        elif ct == "p":
            fig = plot_p_chart_attrs(result, "P 控制图")
        elif ct == "c":
            fig = plot_c_chart_attrs(result, "C 控制图")
        else:
            fig = plot_u_chart_attrs(result, "U 控制图")
        pdf_bytes = generate_attributes_report(
            result=result, chart_type=ct,
            auto_eval=st.session_state.get("attr_auto_eval", ""),
            ai_analysis=st.session_state.get("attr_ai_answer", ""),
            fig_chart=fig,
        )
        st.download_button(
            label="导出 PDF 报告", data=pdf_bytes,
            file_name="计数控制图分析报告.pdf", mime="application/pdf",
            key="attr_download_btn",
        )
    except Exception:
        st.warning("PDF 生成失败，请确保已安装 reportlab")


# --- 入口 ---
attributes_page()
