import streamlit as st
st.set_page_config(page_title="Box-Cox变换分析", page_icon="📈", layout="wide")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.data_loader import load_file, parse_long_format, parse_wide_format, generate_profile
from utils.normality_test import shapiro_test
from utils.boxcox_utils import boxcox_transform
from utils.capability import calculate_capability
from utils.plotting import plot_capability, plot_qq, plot_probability, plot_boxplot
from utils.ai_assistant import ask_ai
from utils.report_generator import generate_boxcox_report
from utils.theme import apply_theme, Colors, kpi_card, render_copilot_sidebar
from config import *
from scipy.stats import norm
from scipy import stats as sp_stats

apply_theme()


# ==================================================
# 内置示例数据
# ==================================================

def load_example_data(exp_type):
    if exp_type == "non_normal":
        data = [
            5.99, 8.05, 7.09, 13.9, 14.05, 11.46, 4.35, 10.57, 24.52, 12.79,
            11.4, 12.79, 11.14, 4.34, 11.48, 3.7, 4.06, 14.25, 4.79, 6.8,
            12.88, 23.43, 17.58, 29.77, 7.55, 14.54, 9.33, 25.3, 6.86, 10.07,
            16.76, 12.15, 5.3, 10.15, 19.62, 7.29, 9.33, 12.05, 8.89, 16.79,
            5.9, 9.23, 16.25, 10.68, 27.36, 22.05, 4.82, 7.51, 11.34, 13.01,
            9.3, 6.84, 14.69, 15.17, 6.2, 12.51, 7.67, 3.69, 7.57, 9.4,
            13.6, 6.99, 18.04, 7.79, 15.9, 4.64, 3.96, 8.9, 5.14, 9.78,
            10.34, 14.3, 17.05, 17.72, 9.81, 4.39, 7.08, 20.23, 14.7, 23.26,
            9.6, 13.69, 9.38, 18.76, 5.06, 13.43, 9.69, 3.53, 18.66, 5.92,
            5.51, 12.27, 14.27, 10.26, 9.38, 4.17, 20.15, 3.41, 6.48, 14.97,
            9.76, 8.33, 4.38, 6.32, 8.2, 6.07, 6.23, 5.9, 12.21, 40.27,
            13.74, 8.87, 31.32, 11.37, 36.68, 15.35, 18.71, 3.44, 4.33, 9.27,
            8.98, 13.22, 24.78, 6.66, 8.16,
        ]
        return pd.DataFrame({"杂质含量": data})
    return None


# ==================================================
# 辅助函数
# ==================================================

def _detect_numeric_columns(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _detect_group_columns(df):
    candidates = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            candidates.append(c)
        elif df[c].nunique() < len(df) * 0.3:
            candidates.append(c)
    return candidates


def _capability_grade(cpk):
    if cpk >= 1.67: return "A+ (特优)"
    elif cpk >= 1.33: return "A (良好)"
    elif cpk >= 1.00: return "B (尚可)"
    elif cpk >= 0.67: return "C (不足)"
    else: return "D (严重不足)"


def _grade_color(grade):
    return Colors.CAPABILITY_GRADE_COLORS.get(grade, Colors.TEXT_SECONDARY)


def _sigma_level_from_ppm(ppm):
    if ppm <= 0: return 6.0
    return round(0.8406 + np.sqrt(29.37 - 2.221 * np.log(max(ppm, 1e-10))), 2)


# ==================================================
# session_state
# ==================================================

def _init_state():
    defaults = {
        "bc_result": None, "bc_lam": None, "bc_p_orig": None,
        "bc_p_trans": None, "bc_lsl": None, "bc_usl": None,
        "bc_target": None, "bc_data_orig": None, "bc_data_trans": None,
        "bc_grade": None, "bc_sigma": None, "bc_ai_answer": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ==================================================
# 主页面
# ==================================================

def boxcox_page():
    _init_state()
    render_copilot_sidebar()

    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">
        <div style="width:40px;height:40px;background:var(--pca-primary);border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-size:1.2rem;">C</div>
        <div>
            <div style="font-weight:700;font-size:1.2rem;color:var(--pca-text-primary);">Box-Cox 变换分析</div>
            <div style="font-size:0.8rem;color:var(--pca-text-secondary);">Box-Cox Transformation · 非正态数据幂变换 · 变换后能力评估</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 数据源
    with st.container(border=True):
        data_source = st.radio(
            "数据来源", ["上传文件", "使用示例数据"],
            horizontal=True, label_visibility="collapsed",
        )
        df = None
        is_example = (data_source == "使用示例数据")
        if is_example:
            df = load_example_data("non_normal")
            st.caption("已加载示例数据：125 条非正态偏态数据")
        else:
            uploaded_file = st.file_uploader(
                "上传 Excel 或 CSV 文件", type=["xlsx", "csv"],
                label_visibility="collapsed",
            )
            if uploaded_file is None:
                st.info("请先上传数据文件，或切换到「使用示例数据」")
                return
            try:
                df = load_file(uploaded_file)
            except Exception as e:
                st.error(f"文件读取失败：{e}")
                return

    # 数据预览
    with st.expander(f"数据预览与结构（共 {len(df)} 条）"):
        if df is not None:
            cols = st.columns([1, 1])
            with cols[0]:
                st.caption("前 20 条")
                st.dataframe(df.head(20), use_container_width=True)
            with cols[1]:
                st.caption("数据结构")
                st.dataframe(generate_profile(df), use_container_width=True)

    # 数据配置
    with st.container(border=True):
        numeric_cols = _detect_numeric_columns(df)
        group_cols = _detect_group_columns(df)

        data_format = st.radio(
            "数据格式", ["单列 (堆叠)", "多列 (展开)", "单列（自动检测）"],
            horizontal=True, key="bc_data_format",
        )

        if data_format == "多列 (展开)":
            subgroup_col = st.selectbox(
                "分组标识列 (如批次号)",
                ["（无）"] + group_cols,
                key="bc_subgroup_col",
            )
            selected_cols = st.multiselect(
                "测量值列", numeric_cols, key="bc_selected_cols",
            )
            if selected_cols:
                try:
                    parsed = parse_wide_format(df, selected_cols, subgroup_col if subgroup_col != "（无）" else None)
                    column_name = st.text_input(
                        "测量值列名", value=parsed.columns[0] if len(parsed.columns) > 0 else "Value",
                        key="bc_column_name",
                    )
                    measure_col = parsed.columns[0] if len(parsed.columns) > 0 else "Value"
                    measure_df = parsed.rename(columns={measure_col: column_name})
                    measure_col = column_name
                except Exception:
                    st.warning("多列解析失败，将自动检测数值列")
                    measure_df = df.copy()
                    if numeric_cols:
                        measure_col = numeric_cols[0]
                    else:
                        st.error("未检测到数值列")
                        return
            else:
                measure_df = df.copy()
                measure_col = numeric_cols[0] if numeric_cols else None
                if measure_col is None:
                    st.error("数据无有效列")
                    return
        elif data_format == "单列 (堆叠)":
            value_col = st.selectbox("测量值列", numeric_cols, key="bc_value_col")
            batch_col = st.selectbox(
                "批次列 (可选)", ["（无批次列）"] + group_cols,
                key="bc_batch_col",
            )
            try:
                batch = batch_col if batch_col != "（无批次列）" else None
                data_arr, sub_size = parse_long_format(df, batch, value_col)
                measure_df = pd.DataFrame({value_col: data_arr})
                measure_col = value_col
                st.caption(f"检测到子组大小 ≈ {sub_size}")
            except Exception:
                measure_df = df[[value_col]].copy()
                measure_col = value_col
        else:  # 单列自动检测
            col = st.selectbox("测量值列（自动检测）", numeric_cols, key="bc_auto_col")
            measure_df = df[[col]].copy()
            measure_col = col

        # 规格限（LSL 可选，非正态数据可能只有上规格限）
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        with c1:
            has_lsl = st.checkbox("启用规格下限 (LSL)", value=False, key="bc_has_lsl")
        with c2:
            lsl = st.number_input("LSL", value=0.0, format="%.4f", key="bc_lsl_input") if has_lsl else None
        with c3:
            usl = st.number_input("规格上限 (USL)", value=30.0, format="%.4f", key="bc_usl_input")
        with c4:
            target = st.number_input("目标值 (Target)", value=10.0, format="%.4f", key="bc_target_input")

    # 执行分析
    if st.button("开始分析", use_container_width=True, type="primary"):
        data = measure_df[measure_col].dropna().values
        if len(data) < 5:
            st.error("数据量不足（需至少 5 个观测值）")
            return

        # 正态性检验（变换前）
        _, p_orig = shapiro_test(data)

        # Box-Cox 变换
        try:
            data_trans, lam = boxcox_transform(data)
        except Exception as e:
            st.error(f"Box-Cox 变换失败：{e}（可能数据包含非正值，请检查）")
            return

        # 变换后正态性检验
        _, p_trans = shapiro_test(data_trans)

        # 变换规格限（与 BoxCox 变换一致，lsl ≤ 0 时无法变换→单边规格）
        def _bc_xform(x):
            if x is not None and x > 0:
                return np.log(x) if abs(lam) < 1e-8 else (x**lam - 1) / lam
            return None
        t_lsl = _bc_xform(lsl) if (lsl is not None and lsl > 0) else None
        t_usl = _bc_xform(usl)
        t_target = _bc_xform(target)
        # 用变换后规格限分析变换后数据
        result = calculate_capability(data_trans, t_lsl, t_usl, 1, t_target)
        grade = _capability_grade(result["cpk"])
        sigma_level = _sigma_level_from_ppm(result["ppm"])

        for key, val in [
            ("bc_result", result), ("bc_lam", lam), ("bc_p_orig", p_orig),
            ("bc_p_trans", p_trans), ("bc_lsl", lsl), ("bc_usl", usl),
            ("bc_target", target), ("bc_data_orig", data), ("bc_data_trans", data_trans),
            ("bc_grade", grade), ("bc_sigma", sigma_level),
            ("bc_t_lsl", t_lsl), ("bc_t_usl", t_usl), ("bc_t_target", t_target),
        ]:
            st.session_state[key] = val
        st.session_state["bc_ai_answer"] = None
        st.rerun()

    result = st.session_state["bc_result"]
    if result is None:
        return

    lam = st.session_state["bc_lam"]
    p_orig = st.session_state["bc_p_orig"]
    p_trans = st.session_state["bc_p_trans"]
    lsl = st.session_state["bc_lsl"]
    usl = st.session_state["bc_usl"]
    target = st.session_state["bc_target"]
    t_lsl = st.session_state["bc_t_lsl"]
    t_usl = st.session_state["bc_t_usl"]
    t_target = st.session_state["bc_t_target"]
    data_orig = st.session_state["bc_data_orig"]
    data_trans = st.session_state["bc_data_trans"]
    grade = st.session_state["bc_grade"]
    sigma_level = st.session_state["bc_sigma"]

    # ========== KPI 仪表板 ==========
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 关键指标概览")

    g_color = _grade_color(grade)
    is_normal_orig = p_orig > 0.05
    is_normal_trans = p_trans > 0.05

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(kpi_card("最佳 λ", f"{lam:.4f}", "Box-Cox 变换参数"), unsafe_allow_html=True)
    k2.markdown(kpi_card("变换前 P 值", f"{p_orig:.4f}",
                          "正态" if is_normal_orig else "非正态",
                          color=Colors.SUCCESS if is_normal_orig else Colors.WARNING),
                unsafe_allow_html=True)
    k3.markdown(kpi_card("变换后 P 值", f"{p_trans:.4f}",
                          "正态" if is_normal_trans else "非正态",
                          color=Colors.SUCCESS if is_normal_trans else Colors.WARNING),
                unsafe_allow_html=True)
    k4.markdown(kpi_card("Cp (变换后)", f"{result['cp']:.2f}", "潜在组内能力"), unsafe_allow_html=True)
    k5.markdown(kpi_card("Cpk (变换后)", f"{result['cpk']:.2f}", grade, color=g_color), unsafe_allow_html=True)

    k6, k7, k8, k9, k10 = st.columns(5)
    k6.markdown(kpi_card("Pp", f"{result['pp']:.2f}", "整体性能"), unsafe_allow_html=True)
    k7.markdown(kpi_card("Ppk", f"{result['ppk']:.2f}", ""), unsafe_allow_html=True)
    cpm_str = f"{result['cpm']:.2f}" if "cpm" in result else "—"
    k8.markdown(kpi_card("Cpm", cpm_str, ""), unsafe_allow_html=True)
    k9.markdown(kpi_card("PPM (变换后)", f"{result['ppm']:.0f}", ""), unsafe_allow_html=True)
    k10.markdown(kpi_card("Sigma", f"{sigma_level}σ", ""), unsafe_allow_html=True)

    k11, k12, k13, k14, k15 = st.columns(5)
    k11.markdown(kpi_card("变换后均值", f"{result['mean']:.4f}", ""), unsafe_allow_html=True)
    k12.markdown(kpi_card("整体 σ", f"{result['std_overall']:.4f}", ""), unsafe_allow_html=True)
    k13.markdown(kpi_card("组内 σ", f"{result['std_within']:.4f}", ""), unsafe_allow_html=True)
    k14.markdown(kpi_card("原始均值", f"{np.mean(data_orig):.4f}", ""), unsafe_allow_html=True)
    k15.markdown(kpi_card("原始 σ", f"{np.std(data_orig, ddof=1):.4f}", ""), unsafe_allow_html=True)

    # ========== 图表 ==========
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 图形分析")

    tab1, tab2, tab3, tab4 = st.tabs(["直方图对比", "Q-Q 图对比", "概率图对比", "能力直方图"])

    with tab1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(data_orig, bins="auto", density=True, alpha=0.6, edgecolor="#cbd5e1", color=Colors.PRIMARY, linewidth=0.5)
        x1 = np.linspace(np.min(data_orig), np.max(data_orig), 500)
        ax1.plot(x1, norm.pdf(x1, np.mean(data_orig), np.std(data_orig, ddof=1)),
                 color=Colors.DANGER, linewidth=2)
        ax1.set_title("原始数据直方图")
        ax1.set_xlabel("测量值")
        ax1.set_ylabel("密度")

        ax2.hist(data_trans, bins="auto", density=True, alpha=0.6, edgecolor="#cbd5e1", color=Colors.PRIMARY_LIGHT, linewidth=0.5)
        x2 = np.linspace(np.min(data_trans), np.max(data_trans), 500)
        ax2.plot(x2, norm.pdf(x2, np.mean(data_trans), np.std(data_trans, ddof=1)),
                 color=Colors.DANGER, linewidth=2)
        ax2.set_title(f"变换后直方图 (λ={lam:.3f})")
        ax2.set_xlabel("变换后值")
        ax2.set_ylabel("密度")
        st.pyplot(fig)

    with tab2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        sp_stats.probplot(data_orig, dist="norm", plot=ax1)
        ax1.set_title("原始数据 Q-Q 图")
        ax1.get_lines()[0].set_color(Colors.PRIMARY)
        ax1.get_lines()[1].set_color(Colors.DANGER)
        sp_stats.probplot(data_trans, dist="norm", plot=ax2)
        ax2.set_title(f"变换后 Q-Q 图 (λ={lam:.3f})")
        ax2.get_lines()[0].set_color(Colors.PRIMARY_LIGHT)
        ax2.get_lines()[1].set_color(Colors.DANGER)
        st.pyplot(fig)

    with tab3:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        so = np.sort(data_orig)
        ax1.scatter(
            so, norm.ppf(np.arange(1, len(data_orig) + 1) / (len(data_orig) + 1)),
            alpha=0.5, color=Colors.PRIMARY,
        )
        ax1.set_title("原始数据概率图")
        ax1.set_xlabel("观测值")
        ax1.set_ylabel("期望 Z")
        st_sorted = np.sort(data_trans)
        ax2.scatter(
            st_sorted, norm.ppf(np.arange(1, len(data_trans) + 1) / (len(data_trans) + 1)),
            alpha=0.5, color=Colors.PRIMARY_LIGHT,
        )
        ax2.set_title(f"变换后概率图 (λ={lam:.3f})")
        ax2.set_xlabel("变换后观测值")
        ax2.set_ylabel("期望 Z")
        st.pyplot(fig)

    with tab4:
        fig = plot_capability(
            data_trans, result["mean"], result["std_overall"],
            result["std_within"], t_lsl, t_usl, t_target,
        )
        st.pyplot(fig)

    # ========== 摘要表格 ==========
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 统计摘要")

    summary_df = pd.DataFrame({
        "指标": [
            "原始数据均值", "原始数据标准差", "最佳 λ",
            "变换后均值", "变换后整体标准差", "变换后组内标准差",
            "Cp (变换后)", "Cpk (变换后)", "Pp (变换后)", "Ppk (变换后)",
            "Cpm (变换后)", "PPM (变换后)", "Sigma (变换后)",
            "原始 P 值", "变换后 P 值",
        ],
        "数值": [
            f"{np.mean(data_orig):.4f}", f"{np.std(data_orig, ddof=1):.4f}", f"{lam:.4f}",
            f"{result['mean']:.4f}", f"{result['std_overall']:.4f}", f"{result['std_within']:.4f}",
            f"{result['cp']:.3f}", f"{result['cpk']:.3f}", f"{result['pp']:.3f}", f"{result['ppk']:.3f}",
            f"{result['cpm']:.3f}" if "cpm" in result else "—",
            f"{result['ppm']:.0f}", f"{sigma_level}σ",
            f"{p_orig:.4f}", f"{p_trans:.4f}",
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ========== AI 智能分析 ==========
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### AI 智能分析")

    ai_answer = st.session_state.get("bc_ai_answer")
    if ai_answer:
        st.markdown(ai_answer)

    context = f"""Box-Cox 变换分析结果：
最佳 λ 值：{lam:.4f}
变换前正态性 P 值：{p_orig:.4f}（{'通过' if is_normal_orig else '未通过'}）
变换后正态性 P 值：{p_trans:.4f}（{'通过' if is_normal_trans else '未通过'}）
变换后能力指标：
均值：{result['mean']:.4f}  整体σ：{result['std_overall']:.4f}  组内σ：{result['std_within']:.4f}
Cp：{result['cp']:.2f}  Cpk：{result['cpk']:.2f}  Pp：{result['pp']:.2f}  Ppk：{result['ppk']:.2f}
Cpm：{result.get('cpm', '—')}  PPM：{result['ppm']:.0f}
能力等级：{grade}  Sigma：{sigma_level}σ"""

    if st.button("生成 AI 分析报告", key="bc_ai_btn"):
        with st.spinner("AI 分析中..."):
            try:
                answer = ask_ai("请评价该 Box-Cox 变换效果及变换后过程能力。", context)
                st.session_state["bc_ai_answer"] = answer
                st.rerun()
            except Exception as e:
                st.error(f"AI 调用失败：{e}")

    # ========== PDF 导出 ==========
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    try:
        # 生成报告用图表
        fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(data_orig, bins="auto", density=True, alpha=0.6, edgecolor="#cbd5e1", color=Colors.PRIMARY, linewidth=0.5)
        x1 = np.linspace(np.min(data_orig), np.max(data_orig), 500)
        ax1.plot(x1, norm.pdf(x1, np.mean(data_orig), np.std(data_orig, ddof=1)), color=Colors.DANGER, linewidth=2)
        ax1.set_title("原始数据直方图"); ax1.set_xlabel("测量值"); ax1.set_ylabel("密度")
        ax2.hist(data_trans, bins="auto", density=True, alpha=0.6, edgecolor="#cbd5e1", color=Colors.PRIMARY_LIGHT, linewidth=0.5)
        x2 = np.linspace(np.min(data_trans), np.max(data_trans), 500)
        ax2.plot(x2, norm.pdf(x2, np.mean(data_trans), np.std(data_trans, ddof=1)), color=Colors.DANGER, linewidth=2)
        ax2.set_title(f"变换后直方图 (λ={lam:.3f})"); ax2.set_xlabel("变换后值"); ax2.set_ylabel("密度")

        fig_qq, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
        sp_stats.probplot(data_orig, dist="norm", plot=ax3)
        ax3.set_title("原始数据 Q-Q 图")
        ax3.get_lines()[0].set_color(Colors.PRIMARY); ax3.get_lines()[1].set_color(Colors.DANGER)
        sp_stats.probplot(data_trans, dist="norm", plot=ax4)
        ax4.set_title(f"变换后 Q-Q 图 (λ={lam:.3f})")
        ax4.get_lines()[0].set_color(Colors.PRIMARY_LIGHT); ax4.get_lines()[1].set_color(Colors.DANGER)

        fig_prob, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
        so = np.sort(data_orig)
        ax5.scatter(so, norm.ppf(np.arange(1, len(data_orig)+1)/(len(data_orig)+1)), alpha=0.5, color=Colors.PRIMARY)
        ax5.set_title("原始数据概率图"); ax5.set_xlabel("观测值"); ax5.set_ylabel("期望 Z")
        st_s = np.sort(data_trans)
        ax6.scatter(st_s, norm.ppf(np.arange(1, len(data_trans)+1)/(len(data_trans)+1)), alpha=0.5, color=Colors.PRIMARY_LIGHT)
        ax6.set_title(f"变换后概率图 (λ={lam:.3f})"); ax6.set_xlabel("变换后观测值"); ax6.set_ylabel("期望 Z")

        fig_cap = plot_capability(data_trans, result["mean"], result["std_overall"], result["std_within"], t_lsl, t_usl, t_target)

        pdf_bytes = generate_boxcox_report(
            result=result, lam=lam, p_orig=p_orig, p_trans=p_trans,
            lsl=lsl, usl=usl, target=target,
            data_orig=data_orig, data_trans=data_trans,
            ai_analysis=ai_answer if ai_answer else "",
            fig_hist=fig_hist, fig_qq=fig_qq, fig_prob=fig_prob, fig_cap=fig_cap,
        )
        label = "导出 PDF 报告（含 AI 分析）" if ai_answer else "导出 PDF 报告"
        st.download_button(
            label=label, data=pdf_bytes,
            file_name="BoxCox变换分析报告.pdf",
            mime="application/pdf", key="bc_download_btn",
        )
    except Exception:
        st.warning("PDF 生成失败，请确保已安装 reportlab")


# --- 入口 ---
boxcox_page()
