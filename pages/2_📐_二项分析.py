import streamlit as st
st.set_page_config(page_title="二项过程能力分析", page_icon="📐", layout="wide")
import pandas as pd
import numpy as np

from utils.data_loader import load_file
from utils.binomial_capability import calculate_binomial_capability
from utils.plotting import plot_p_chart, plot_cumulative_rate, plot_defect_distribution, plot_binomial_fit
from utils.ai_assistant import ask_ai
from utils.report_generator import generate_binomial_report
from utils.theme import apply_theme, Colors, kpi_card, grade_badge, status_indicator, render_copilot_sidebar

apply_theme()


# ==================================================
# 数据与辅助
# ==================================================

def load_example_data(exp_type):
    if exp_type == "binomial":
        defectives = [3, 4, 6, 8, 6, 6, 3, 4, 8, 7, 9, 6, 12, 3, 7,
                      10, 6, 3, 7, 4, 5, 4, 12, 11, 7, 8, 8, 7, 7, 5]
        samples = [150] * 30
        return pd.DataFrame({"样品数量": samples, "不合格品数量": defectives})
    return None


def _sigma_grade(sigma):
    if sigma >= 6: return "世界级 (6σ)"
    elif sigma >= 5: return "优秀 (5σ)"
    elif sigma >= 4: return "良好 (4σ)"
    elif sigma >= 3: return "一般 (3σ)"
    elif sigma >= 2: return "较差 (2σ)"
    else: return "极差 (<2σ)"


def _sigma_color(sigma):
    level = int(min(sigma, 6))
    return Colors.SIGMA_COLORS.get(level, Colors.DANGER)


def _detect_sample_defect_columns(df):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        return (num_cols[0] if num_cols else None, num_cols[0] if num_cols else None)
    col_means = {c: df[c].mean() for c in num_cols}
    sorted_cols = sorted(num_cols, key=lambda c: col_means[c], reverse=True)
    return sorted_cols[0], sorted_cols[-1]


def _auto_evaluation(result, out_of_control, detail_df):
    lines = []
    if not out_of_control:
        lines.append("##### ✅ 过程受控\n所有批次均位于 P 控制图的上下控制限内，未检测到异常波动，过程处于统计受控状态。")
    else:
        lines.append(f"##### ⚠️ 过程存在异常\n批次 {out_of_control} 超出控制限，建议排查特殊原因（如原材料批次差异、操作员变更、设备调整等），确认原因后考虑剔除异常批次重新分析。")
    pct = result["p_bar"] * 100
    if pct < 1: lines.append(f"**缺陷率水平**：平均不合格率 {pct:.2f}%（{result['ppm']:.0f} PPM），处于较优水平。")
    elif pct < 5: lines.append(f"**缺陷率水平**：平均不合格率 {pct:.2f}%（{result['ppm']:.0f} PPM），存在改进空间。")
    else: lines.append(f"**缺陷率水平**：平均不合格率 {pct:.2f}%（{result['ppm']:.0f} PPM），偏高，建议启动质量改进项目。")
    lines.append(f"**Sigma 水平**：{result['sigma']:.2f}σ（{_sigma_grade(result['sigma'])}），Zbench = {result['z_bench']:.2f}")
    rates = detail_df["不合格率"].values
    cv = np.std(rates, ddof=1) / max(np.mean(rates), 1e-10)
    if cv < 0.2: lines.append(f"**过程稳定性**：各批次不合格率 CV = {cv:.2f}，波动较小，一致性良好。")
    elif cv < 0.5: lines.append(f"**过程稳定性**：各批次不合格率 CV = {cv:.2f}，波动适中。")
    else: lines.append(f"**过程稳定性**：各批次不合格率 CV = {cv:.2f}，波动较大，需加强过程控制。")
    p_fit = result["p_fit"]
    if p_fit > 0.05: lines.append(f"**分布拟合**：P = {p_fit:.3f} > 0.05，数据符合二项分布假设。")
    else: lines.append(f"**分布拟合**：P = {p_fit:.3f} ≤ 0.05，数据可能不完全符合二项分布假设。")
    return "\n\n".join(lines)


def _init_state():
    defaults = {
        "bin_result": None, "bin_grade": None, "bin_samples": None,
        "bin_defectives": None, "bin_detail_df": None, "bin_auto_eval": None,
        "bin_oc_list": None, "bin_ai_answer": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v


# ==================================================
# 主页面
# ==================================================

def binomial_page():
    _init_state()
    render_copilot_sidebar()

    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">
        <div style="width:40px;height:40px;background:var(--pca-primary);border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-size:1.2rem;">B</div>
        <div>
            <div style="font-weight:700;font-size:1.2rem;color:var(--pca-text-primary);">二项过程能力分析</div>
            <div style="font-size:0.8rem;color:var(--pca-text-secondary);">Binomial Capability Analysis · 计数型数据 · P 控制图</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 数据源
    with st.container(border=True):
        data_source = st.radio("数据来源", ["上传文件", "使用示例数据"], horizontal=True, label_visibility="collapsed")
        df = None; is_example = (data_source == "使用示例数据")
        if is_example:
            df = load_example_data("binomial")
            st.caption("已加载示例数据：30 批次，每批 n = 150")
        else:
            uploaded_file = st.file_uploader("上传 Excel 或 CSV 文件", type=["xlsx", "csv"], label_visibility="collapsed")
            if uploaded_file is None:
                st.info("请先上传数据文件，或切换到「使用示例数据」")
                return
            try:
                df = load_file(uploaded_file)
            except Exception as e:
                st.error(f"文件读取失败：{e}")
                return

    with st.expander("数据预览与结构"):
        if df is not None:
            st.dataframe(df.head(20), use_container_width=True)

    # 数据配置
    with st.container(border=True):
        auto_sample, auto_defect = _detect_sample_defect_columns(df)
        sample_col = st.selectbox("样品数量列", df.columns.tolist(),
                                  index=df.columns.tolist().index(auto_sample) if auto_sample in df.columns else 0)
        defect_col = st.selectbox("不合格品数量列", df.columns.tolist(),
                                  index=df.columns.tolist().index(auto_defect) if auto_defect in df.columns else 0)

    if st.button("开始分析", use_container_width=True, type="primary"):
        samples = df[sample_col].dropna().astype(int).values
        defectives = df[defect_col].dropna().astype(int).values
        min_len = min(len(samples), len(defectives))
        samples = samples[:min_len]
        defectives = defectives[:min_len]

        if len(samples) < 3:
            st.error("数据量不足（需至少 3 个批次）")
            return

        result = calculate_binomial_capability(samples, defectives)
        grade = _sigma_grade(result["sigma"])

        detail_df = pd.DataFrame({
            "批次": np.arange(1, len(samples) + 1),
            "样品数": samples,
            "不合格品数": defectives,
            "不合格率": np.round(defectives / np.maximum(samples, 1), 4),
        })
        p_bar = result["p_bar"]
        oc_list = []
        for i, n in enumerate(samples):
            sigma_p = np.sqrt(p_bar * (1 - p_bar) / n) if n > 0 else 0
            ucl = p_bar + 3 * sigma_p; lcl = max(0, p_bar - 3 * sigma_p)
            if defectives[i] / max(n, 1) > ucl or defectives[i] / max(n, 1) < lcl:
                oc_list.append(i + 1)
        auto_eval = _auto_evaluation(result, oc_list, detail_df)

        for key, val in [
            ("bin_result", result), ("bin_grade", grade), ("bin_samples", samples),
            ("bin_defectives", defectives), ("bin_detail_df", detail_df),
            ("bin_auto_eval", auto_eval), ("bin_oc_list", oc_list),
        ]:
            st.session_state[key] = val
        st.session_state["bin_ai_answer"] = None
        st.rerun()

    # --- 结果区 ---
    result = st.session_state["bin_result"]
    if result is None: return

    grade = st.session_state["bin_grade"]
    samples = st.session_state["bin_samples"]
    defectives = st.session_state["bin_defectives"]
    detail_df = st.session_state["bin_detail_df"]
    auto_eval = st.session_state["bin_auto_eval"]
    oc_list = st.session_state["bin_oc_list"]
    sigma = result["sigma"]
    s_color = _sigma_color(sigma)

    # ========== KPI 仪表板 ==========
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 关键指标概览")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(kpi_card("总样本数", f"{result['total_samples']:,}", ""), unsafe_allow_html=True)
    k2.markdown(kpi_card("总不合格数", f"{result['total_defects']:,}", ""), unsafe_allow_html=True)
    k3.markdown(kpi_card("平均不合格率", f"{result['p_bar']:.4%}",
                          f"PPM: {result['ppm']:,.0f}"), unsafe_allow_html=True)
    k4.markdown(kpi_card("Sigma 水平", f"{sigma:.2f}σ",
                          grade, color=s_color), unsafe_allow_html=True)
    k5.markdown(kpi_card("控制状态",
                          "受控" if not oc_list else f"{len(oc_list)}批异常",
                          "", color=Colors.SUCCESS if not oc_list else Colors.DANGER),
                unsafe_allow_html=True)

    k6, k7, k8, k9, k10 = st.columns(5)
    k6.markdown(kpi_card("Zbench", f"{result['z_bench']:.2f}", ""), unsafe_allow_html=True)
    k7.markdown(kpi_card("DPMO", f"{result['dpmo']:,.0f}", ""), unsafe_allow_html=True)
    k8.markdown(kpi_card("拟合检验 P", f"{result['p_fit']:.4f}",
                          "符合二项分布" if result['p_fit'] > 0.05 else "可能偏离",
                          color=Colors.SUCCESS if result['p_fit'] > 0.05 else Colors.WARNING),
                unsafe_allow_html=True)
    k9.markdown(kpi_card("95% CI 下限", f"{result['ci_ppm_lower']:,.0f} PPM", ""), unsafe_allow_html=True)
    k10.markdown(kpi_card("95% CI 上限", f"{result['ci_ppm_upper']:,.0f} PPM", ""), unsafe_allow_html=True)

    # ========== 图表 ==========
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 图形分析")
    tab1, tab2, tab3, tab4 = st.tabs(["P 控制图", "累计趋势", "分布分析", "拟合检验"])
    with tab1: st.pyplot(plot_p_chart(samples, defectives))
    with tab2: st.pyplot(plot_cumulative_rate(samples, defectives))
    with tab3: st.pyplot(plot_defect_distribution(samples, defectives))
    with tab4: st.pyplot(plot_binomial_fit(samples, defectives, result["p_bar"]))

    # ========== 明细与评价 ==========
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("##### 自动评价报告")
        st.markdown(auto_eval)
    with col_right:
        st.markdown("##### 批次明细")
        st.dataframe(detail_df, use_container_width=True, height=300)

    # ========== AI 智能分析 ==========
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### AI 智能分析")

    ai_answer = st.session_state.get("bin_ai_answer")
    if ai_answer:
        st.markdown(ai_answer)

    context = f"""二项过程能力分析结果：
总样本数：{result['total_samples']:,}  总不合格数：{result['total_defects']:,}
平均不合格率：{result['p_bar']:.4%}  PPM：{result['ppm']:,.0f}  DPMO：{result['dpmo']:,.0f}
Sigma 水平：{result['sigma']:.2f}σ  Zbench：{result['z_bench']:.2f}
95% CI：[{result['ci_ppm_lower']:,.0f}, {result['ci_ppm_upper']:,.0f}] PPM
控制状态：{'受控' if not oc_list else '部分异常'}  """

    if st.button("生成 AI 分析报告", key="bin_ai_btn"):
        with st.spinner("AI 分析中..."):
            try:
                answer = ask_ai("请评价该二项过程能力并给出改进建议。", context)
                st.session_state["bin_ai_answer"] = answer
                st.rerun()
            except Exception as e:
                st.error(f"AI 调用失败：{e}")

    # ========== PDF 导出 ==========
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    try:
        pdf_bytes = generate_binomial_report(
            result=result, ai_analysis=ai_answer if ai_answer else "",
            fig_p_chart=plot_p_chart(samples, defectives),
            fig_cumulative=plot_cumulative_rate(samples, defectives),
            fig_distribution=plot_defect_distribution(samples, defectives),
            fig_fit=plot_binomial_fit(samples, defectives, result["p_bar"]),
        )
        label = "导出 PDF 报告（含 AI 分析）" if ai_answer else "导出 PDF 报告"
        st.download_button(label=label, data=pdf_bytes,
                           file_name="二项过程能力分析报告.pdf",
                           mime="application/pdf", key="bin_download_btn")
    except Exception:
        st.warning("PDF 生成失败，请确保已安装 reportlab")


# --- 入口 ---
binomial_page()
