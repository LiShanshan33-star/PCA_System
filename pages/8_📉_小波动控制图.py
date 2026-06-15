import streamlit as st
st.set_page_config(page_title="小波动控制图", page_icon="📉", layout="wide")
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
    if exp_type == "cusum":
        # 实验五 例1：塑料强度 (Table 12)
        data = [
            10.575913, 10.278616, 11.301409, 10.05637, 9.2193314,
            9.481033, 9.305792, 10.26081, 11.094854, 8.7883851,
            9.0554196, 9.8245419, 9.99606, 10.435533, 9.3337819,
            11.17357, 9.2222024, 9.9687704, 11.714891, 12.193764,
            9.4106213, 10.084178, 9.0978906, 11.345605, 9.60933,
            11.833308, 11.037904, 11.580857, 10.169734, 11.633219,
        ]
        return pd.DataFrame({"强度": data}), "cusum"
    elif exp_type == "ewma":
        # 实验五 例2：芯片研磨厚度 (Table 13)
        data = [
            2.99, 3.05, 3.02, 2.98, 3.05, 3.03, 3.06, 3.01, 2.98, 3.07,
            2.98, 2.99, 3.05, 3.02, 3.01, 3.04, 3.08, 3.12, 3.11, 3.13,
        ]
        return pd.DataFrame({"厚度": data}), "ewma"
    return None, None


# ==================================================
# I-MR 计算 (复用)
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
# CUSUM 计算 (双面Tabular CUSUM)
# ==================================================
def compute_cusum(data, target=None, k=0.5, h=5.0):
    """Tabular CUSUM: 双侧累积和
    k: 参考值 (slack value), 通常k=0.5σ
    h: 判定距 (decision interval), 通常h=4σ或5σ
    """
    n = len(data)
    if target is None:
        target = np.mean(data)
    sigma = np.std(data, ddof=1)
    if sigma < 1e-10:
        sigma = 1

    k_val = k * sigma
    h_val = h * sigma

    c_plus = np.zeros(n)
    c_minus = np.zeros(n)
    ucl_val = h_val

    for i in range(n):
        if i == 0:
            c_plus[i] = max(0, data[i] - (target + k_val))
            c_minus[i] = max(0, (target - k_val) - data[i])
        else:
            c_plus[i] = max(0, c_plus[i-1] + data[i] - (target + k_val))
            c_minus[i] = max(0, c_minus[i-1] + (target - k_val) - data[i])

    return {
        "c_plus": c_plus, "c_minus": c_minus,
        "target": target, "k": k, "h": h,
        "k_val": k_val, "h_val": h_val, "n": n,
    }


# ==================================================
# EWMA 计算
# ==================================================
def compute_ewma(data, lam=0.2, target=None):
    """EWMA 控制图
    lam: 平滑系数 (0 < lam <= 1)
    """
    n = len(data)
    if target is None:
        target = np.mean(data)
    sigma = np.std(data, ddof=1)
    if sigma < 1e-10:
        sigma = 1

    ewma = np.zeros(n)
    ewma[0] = data[0]

    for i in range(1, n):
        ewma[i] = lam * data[i] + (1 - lam) * ewma[i-1]

    # 时变控制限
    t = np.arange(1, n + 1)
    factor = np.sqrt((lam / (2 - lam)) * (1 - (1 - lam)**(2 * t)))
    ucl = target + 3 * sigma * factor
    lcl = target - 3 * sigma * factor

    return {
        "ewma": ewma, "target": target, "lam": lam,
        "ucl": ucl, "lcl": lcl, "n": n, "sigma": sigma,
    }


# ==================================================
# 绘图
# ==================================================
def plot_imr_chart(result, title, ylabel):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 0.7]})

    x = np.arange(1, result["n"] + 1)

    ax1.plot(x, result["values"], "bo-", markersize=5, linewidth=1.5, label="单值")
    ax1.axhline(result["mean"], color="green", linestyle="-", linewidth=1.5, label=f"CL={result['mean']:.4f}")
    ax1.axhline(result["i_ucl"], color="red", linestyle="--", linewidth=1.5, label=f"UCL={result['i_ucl']:.4f}")
    ax1.axhline(result["i_lcl"], color="red", linestyle="--", linewidth=1.5, label=f"LCL={result['i_lcl']:.4f}")
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, result["n"] + 0.5)

    mr_x = np.arange(2, result["n"] + 1)
    ax2.plot(mr_x, result["mr"], "bo-", markersize=5, linewidth=1.5, label="移动极差")
    ax2.axhline(result["mrbar"], color="green", linestyle="-", linewidth=1.5, label=f"CL={result['mrbar']:.4f}")
    ax2.axhline(result["mr_ucl"], color="red", linestyle="--", linewidth=1.5, label=f"UCL={result['mr_ucl']:.4f}")
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.5, label="LCL=0")
    ax2.set_title("移动极差控制图", fontsize=13, fontweight="bold")
    ax2.set_xlabel("序号", fontsize=11)
    ax2.set_ylabel("移动极差", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1.5, result["n"] + 0.5)

    plt.tight_layout()
    return fig


def plot_cusum_chart(result, title="累积和控制图"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    x = np.arange(1, result["n"] + 1)

    ax1.step(x, result["c_plus"], "b-", where="post", linewidth=2, label="C⁺ (上侧)")
    ax1.axhline(result["h_val"], color="red", linestyle="--", linewidth=1.5, label=f"UCL (H={result['h_val']:.3f})")
    ax1.fill_between(x, 0, result["c_plus"], alpha=0.2, color="blue")
    ax1.set_title(f"上侧累积和 C⁺ (k={result['k']}, h={result['h']})", fontsize=13, fontweight="bold")
    ax1.set_ylabel("C⁺", fontsize=11)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, result["n"] + 0.5)

    ax2.step(x, result["c_minus"], "r-", where="post", linewidth=2, label="C⁻ (下侧)")
    ax2.axhline(result["h_val"], color="red", linestyle="--", linewidth=1.5, label=f"UCL (H={result['h_val']:.3f})")
    ax2.fill_between(x, 0, result["c_minus"], alpha=0.2, color="red")
    ax2.set_title(f"下侧累积和 C⁻ (目标={result['target']:.4f})", fontsize=13, fontweight="bold")
    ax2.set_xlabel("样本序号", fontsize=11)
    ax2.set_ylabel("C⁻", fontsize=11)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, result["n"] + 0.5)

    plt.tight_layout()
    return fig


def plot_ewma_chart(result, title="EWMA 控制图"):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(1, result["n"] + 1)

    ax.plot(x, result["ewma"], "bo-", markersize=5, linewidth=2, label=f"EWMA (λ={result['lam']})")
    ax.axhline(result["target"], color="green", linestyle="-", linewidth=1.5, label=f"CL={result['target']:.4f}")
    ax.plot(x, result["ucl"], "r--", linewidth=1.5, label="UCL (时变)")
    ax.plot(x, result["lcl"], "r--", linewidth=1.5, label="LCL (时变)")
    ax.fill_between(x, result["lcl"], result["ucl"], alpha=0.05, color="red")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("样本序号", fontsize=11)
    ax.set_ylabel("EWMA 值", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, result["n"] + 0.5)

    plt.tight_layout()
    return fig


# ==================================================
# session_state
# ==================================================
def _init_state():
    defaults = {
        "sw_result": None, "sw_chart_type": None,
        "sw_imr_result": None, "sw_auto_eval": None,
        "sw_ai_answer": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v


# ==================================================
# 主页面
# ==================================================
def small_shift_page():
    _init_state()
    render_copilot_sidebar()

    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">
        <div style="width:40px;height:40px;background:var(--pca-primary);border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-size:1.2rem;">W</div>
        <div>
            <div style="font-weight:700;font-size:1.2rem;color:var(--pca-text-primary);">小波动控制图技术</div>
            <div style="font-size:0.8rem;color:var(--pca-text-secondary);">Small Shift Detection · CUSUM · EWMA</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    chart_type = st.selectbox(
        "图表类型",
        ["累积和控制图 (CUSUM)", "指数加权移动平均控制图 (EWMA)"]
    )

    example_map = {"累积和控制图 (CUSUM)": "cusum", "指数加权移动平均控制图 (EWMA)": "ewma"}

    with st.container(border=True):
        data_source = st.radio("数据来源", ["上传文件", "使用示例数据"], horizontal=True, label_visibility="collapsed")
        df = None

        labels = {
            "cusum": "已加载示例数据：塑料强度，30个单值（目标值≈10）",
            "ewma": "已加载示例数据：芯片研磨厚度，20个单值（λ=0.2）",
        }

        if data_source == "使用示例数据":
            exp_key = example_map[chart_type]
            df, ct_key = load_example_data(exp_key)
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

    st.markdown("##### 数据预览")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    st.caption(f"共 {len(df)} 行")

    # 参数设置
    if "CUSUM" in chart_type:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            k = st.number_input("参考值 k (σ倍数)", value=0.5, step=0.1, min_value=0.1, max_value=3.0,
                                help="通常 k=0.5，表示检测 1σ 偏移")
        with col_p2:
            h = st.number_input("判定距 h (σ倍数)", value=5.0, step=0.5, min_value=1.0, max_value=10.0,
                                help="通常 h=4 或 5")
    else:
        col_p1 = st.columns(1)[0]
        with col_p1:
            lam = st.number_input("平滑系数 λ", value=0.2, step=0.05, min_value=0.05, max_value=1.0,
                                  help="通常在 0.05~0.3，越小检测小偏移越灵敏")

    # 分析
    if st.button("执行分析", type="primary", use_container_width=True):
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            st.error("未找到数值列")
            return
        data = df[num_cols[0]].dropna().values

        if "CUSUM" in chart_type:
            # 先做 I-MR
            imr_result = compute_imr_simple(data)
            st.session_state["sw_imr_result"] = imr_result

            # CUSUM
            cusum_result = compute_cusum(data, k=k, h=h)
            st.session_state["sw_result"] = cusum_result
            st.session_state["sw_chart_type"] = "cusum"

            # 评价
            c_plus_out = cusum_result["c_plus"] > cusum_result["h_val"]
            c_minus_out = cusum_result["c_minus"] > cusum_result["h_val"]
            eval_lines = []

            i_violations = [i+1 for i in range(len(data)) if data[i] > imr_result["i_ucl"] or data[i] < imr_result["i_lcl"]]
            if not i_violations:
                eval_lines.append("**I-MR图**：单值控制图无异常点。")
            else:
                eval_lines.append(f"**I-MR图**：⚠️ 样本 {i_violations} 超出控制限。")

            if not np.any(c_plus_out) and not np.any(c_minus_out):
                eval_lines.append("**CUSUM**：✅ 无异常，过程均值未发生显著偏移。")
            else:
                if np.any(c_plus_out):
                    idx = np.where(c_plus_out)[0][0]
                    eval_lines.append(f"**CUSUM**：⚠️ 上侧累积和在样本 {idx+1} 超出判定距，过程均值可能向上偏移。")
                if np.any(c_minus_out):
                    idx = np.where(c_minus_out)[0][0]
                    eval_lines.append(f"**CUSUM**：⚠️ 下侧累积和在样本 {idx+1} 超出判定距，过程均值可能向下偏移。")
            eval_lines.append(f"CUSUM参数：目标={cusum_result['target']:.4f}, k={k}, h={h}, σ={np.std(data, ddof=1):.4f}")
            st.session_state["sw_auto_eval"] = "\n\n".join(eval_lines)

        else:  # EWMA
            imr_result = compute_imr_simple(data)
            st.session_state["sw_imr_result"] = imr_result

            ewma_result = compute_ewma(data, lam=lam)
            st.session_state["sw_result"] = ewma_result
            st.session_state["sw_chart_type"] = "ewma"

            # 评价
            out_of_control = [i+1 for i in range(len(data)) if ewma_result["ewma"][i] > ewma_result["ucl"][i] or ewma_result["ewma"][i] < ewma_result["lcl"][i]]
            eval_lines = []

            i_violations = [i+1 for i in range(len(data)) if data[i] > imr_result["i_ucl"] or data[i] < imr_result["i_lcl"]]
            if not i_violations:
                eval_lines.append("**I-MR图**：单值控制图无异常点。")
            else:
                eval_lines.append(f"**I-MR图**：⚠️ 样本 {i_violations} 超出控制限。")

            if not out_of_control:
                eval_lines.append("**EWMA**：✅ 无异常，过程均值未发生显著偏移。")
            else:
                eval_lines.append(f"**EWMA**：⚠️ 样本 {out_of_control} 超出控制限。")
            eval_lines.append(f"EWMA参数：λ={lam}, 目标={ewma_result['target']:.4f}, σ={ewma_result['sigma']:.4f}")
            st.session_state["sw_auto_eval"] = "\n\n".join(eval_lines)

        st.rerun()

    # 显示结果
    result = st.session_state.get("sw_result")
    if result is None:
        return

    ct = st.session_state["sw_chart_type"]
    imr_result = st.session_state.get("sw_imr_result")

    # I-MR 图（先展示）
    if imr_result:
        st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
        st.markdown("##### I-MR 控制图（常规）")
        label = "塑料强度" if ct == "cusum" else ("芯片研磨厚度" if ct == "ewma" else "测量值")
        st.pyplot(plot_imr_chart(imr_result, f"单值-移动极差控制图 — {label}", label))

    # KPI
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 关键指标概览")

    if ct == "cusum":
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(kpi_card("目标值", f"{result['target']:.4f}", ""), unsafe_allow_html=True)
        k2.markdown(kpi_card("k 值", f"{result['k']}", "参考值(σ倍数)"), unsafe_allow_html=True)
        k3.markdown(kpi_card("h 值", f"{result['h']}", "判定距(σ倍数)"), unsafe_allow_html=True)
        k4.markdown(kpi_card("Max C⁺", f"{result['c_plus'].max():.4f}", ""), unsafe_allow_html=True)
        k5.markdown(kpi_card("Max C⁻", f"{result['c_minus'].max():.4f}", ""), unsafe_allow_html=True)
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(kpi_card("目标值", f"{result['target']:.4f}", ""), unsafe_allow_html=True)
        k2.markdown(kpi_card("λ", f"{result['lam']}", "平滑系数"), unsafe_allow_html=True)
        k3.markdown(kpi_card("σ", f"{result['sigma']:.4f}", ""), unsafe_allow_html=True)
        k4.markdown(kpi_card("最终UCL", f"{result['ucl'][-1]:.4f}", ""), unsafe_allow_html=True)
        k5.markdown(kpi_card("最终LCL", f"{result['lcl'][-1]:.4f}", ""), unsafe_allow_html=True)

    # 控制图
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 小波动检测控制图")

    if ct == "cusum":
        st.pyplot(plot_cusum_chart(result, "累积和控制图 (CUSUM)"))
    else:
        st.pyplot(plot_ewma_chart(result, "指数加权移动平均控制图 (EWMA)"))

    # 自动评价
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    auto_eval = st.session_state.get("sw_auto_eval", "")
    if auto_eval:
        st.markdown(auto_eval)

    # 数据明细
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    if ct == "cusum":
        detail_df = pd.DataFrame({
            "序号": range(1, result["n"] + 1),
            "C⁺": result["c_plus"].round(4),
            "C⁻": result["c_minus"].round(4),
        })
    else:
        detail_df = pd.DataFrame({
            "序号": range(1, result["n"] + 1),
            "EWMA": result["ewma"].round(4),
            "UCL": result["ucl"].round(4),
            "LCL": result["lcl"].round(4),
        })
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # AI 分析
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### AI 智能分析")
    ai_answer = st.session_state.get("sw_ai_answer")
    if ai_answer:
        st.markdown(ai_answer)

    if st.button("生成 AI 分析报告", key="sw_ai_btn"):
        ctx = f"{'CUSUM' if ct == 'cusum' else 'EWMA'}控制图结果：" + (
            f"目标={result['target']:.4f}, {'k='+str(result['k'])+', h='+str(result['h']) if ct=='cusum' else 'λ='+str(result['lam'])}"
        )
        try:
            answer = ask_ai("请评价该小波动检测控制图，判断过程是否存在微小偏移，并给出改进建议。", ctx)
            st.session_state["sw_ai_answer"] = answer
            st.rerun()
        except Exception as e:
            st.error(f"AI 调用失败：{e}")



    # PDF
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    try:
        ct = st.session_state.get("sw_chart_type", "")
        result = st.session_state.get("sw_result")
        imr_result = st.session_state.get("sw_imr_result")
        if ct == "cusum":
            fig_main = plot_cusum_chart(result, "CUSUM 控制图")
        else:
            fig_main = plot_ewma_chart(result, "EWMA 控制图")
        fig_imr = None
        if imr_result is not None:
            fig_imr = plot_imr_chart(imr_result, "I-MR 控制图", "测量值")
        pdf_bytes = generate_small_shift_report(
            result=result, chart_type=ct,
            auto_eval=st.session_state.get("sw_auto_eval", ""),
            ai_analysis=st.session_state.get("sw_ai_answer", ""),
            fig_chart=fig_main, fig_imr=fig_imr,
        )
        st.download_button(
            label="导出 PDF 报告", data=pdf_bytes,
            file_name="小波动控制图分析报告.pdf", mime="application/pdf",
            key="sw_download_btn",
        )
    except Exception:
        st.warning("PDF 生成失败，请确保已安装 reportlab")


# --- 入口 ---
small_shift_page()
