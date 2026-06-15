import streamlit as st
st.set_page_config(page_title="正常过程能力分析", page_icon="📊", layout="wide")
import pandas as pd
import numpy as np

from utils.data_loader import load_file, parse_long_format, parse_wide_format, generate_profile
from utils.capability import calculate_capability
from utils.plotting import plot_capability, plot_boxplot, plot_qq, plot_probability, plot_subgroup_means
from utils.normality_test import shapiro_test
from utils.ai_assistant import ask_ai
from utils.report_generator import generate_normal_report
from utils.theme import apply_theme, Colors, kpi_card, render_copilot_sidebar
from config import *

apply_theme()


def load_example_data(exp_type):
    if exp_type == "normal_25":
        data = [
            10.948, 10.913, 10.973, 10.923, 11.02,
            10.92, 10.983, 10.959, 10.939, 10.91,
            10.937, 10.968, 10.984, 10.925, 11.026,
            10.95, 10.961, 10.974, 10.999, 10.949,
            10.952, 10.923, 10.947, 11, 10.939,
        ]
        return pd.DataFrame({"直径": data})
    return None


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


def _detect_numeric_columns(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _is_likely_group_column(name, series):
    """判断某列是否可能是分组/标识列（即使为数值型、高基数）"""
    name_lower = str(name).lower()
    group_keywords = ["批次", "batch", "组", "group", "序号", "id", "no", "编号", "样本"]
    if any(kw in name_lower for kw in group_keywords):
        return True
    if pd.api.types.is_numeric_dtype(series):
        vals = series.dropna()
        if len(vals) == 0:
            return False
        # 整数型且每个值唯一 → 可能是序号/ID列
        if pd.api.types.is_integer_dtype(series):
            if vals.nunique() == len(vals):
                return True
        # 浮点型但实际为整数值（如 1.0, 2.0...）→ 序号列
        # 测量数据如 10.948 不会被误判
        if vals.nunique() == len(vals):
            int_vals = np.round(vals).astype(np.int64)
            if np.allclose(vals, int_vals.astype(float)):
                return True
    return False


def _detect_group_columns(df):
    candidates = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            candidates.append(c)
        elif df[c].nunique() < len(df) * 0.3:
            candidates.append(c)
        elif _is_likely_group_column(c, df[c]):
            candidates.append(c)
    return candidates


def _init_state():
    defaults = {
        "normal_result": None, "normal_p": None, "normal_grade": None,
        "normal_sigma": None, "normal_lsl": None, "normal_usl": None,
        "normal_target": None, "normal_data": None, "normal_subgroup_size": 1,
        "normal_ai_answer": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v


def normal_page():
    _init_state()
    render_copilot_sidebar()

    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">
        <div style="width:40px;height:40px;background:var(--pca-primary);border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-size:1.2rem;">N</div>
        <div>
            <div style="font-weight:700;font-size:1.2rem;color:var(--pca-text-primary);">正态过程能力分析</div>
            <div style="font-size:0.8rem;color:var(--pca-text-secondary);">Normal Capability Analysis · 计量型数据 Cp/Cpk/Pp/Ppk</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 数据源
    with st.container(border=True):
        data_source = st.radio("数据来源", ["上传文件", "使用示例数据"], horizontal=True, label_visibility="collapsed")
        df = None; is_example = (data_source == "使用示例数据")
        if is_example:
            df = load_example_data("normal_25")
            st.caption("已加载示例数据：25 个测量值，建议子组大小 = 5")
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
            st.dataframe(generate_profile(df), use_container_width=True)

    # 数据配置
    with st.container(border=True):
        numeric_cols = _detect_numeric_columns(df)
        group_cols = _detect_group_columns(df)

        data_format = st.radio("数据格式", ["单列 (堆叠)", "多列 (展开)", "单列（自动检测）"], horizontal=True, key="normal_data_format")

        if data_format == "多列 (展开)":
            subgroup_col = st.selectbox("分组标识列 (如批次号)", ["（无）"] + list(df.columns), key="normal_subgroup_col")
            # 测量值列不应包含已识别的分组列
            measure_candidates = [c for c in numeric_cols if c not in group_cols]
            if not measure_candidates:
                measure_candidates = numeric_cols
            selected_cols = st.multiselect("测量值列", measure_candidates, key="normal_selected_cols")
            if selected_cols:
                try:
                    parsed_df, auto_subgroup = parse_wide_format(
                        df, selected_cols,
                        subgroup_col if subgroup_col != "（无）" else None
                    )
                except Exception as e:
                    st.warning("多列展开解析失败，回退为自动检测")
                    import traceback
                    st.caption(f"调试: {e}")
                    measure_df = df.copy()
                    measure_col = measure_candidates[0] if measure_candidates else None
                    if measure_col is None:
                        st.error("未检测到数值列")
                        return
                else:
                    column_name = st.text_input(
                        "测量值列名",
                        value=parsed_df.columns[0] if len(parsed_df.columns) > 0 else "Value",
                        key="normal_column_name"
                    )
                    measure_col = parsed_df.columns[0] if len(parsed_df.columns) > 0 else "Value"
                    measure_df = parsed_df.rename(columns={measure_col: column_name})
                    measure_col = column_name
            else:
                measure_df = df.copy()
                measure_col = measure_candidates[0] if measure_candidates else None
                if measure_col is None:
                    st.error("数据无有效列")
                    return
        elif data_format == "单列 (堆叠)":
            value_col = st.selectbox("测量值列", numeric_cols, key="normal_value_col")
            batch_col = st.selectbox("批次列 (可选)", ["（无批次列）"] + group_cols, key="normal_batch_col")
            try:
                measure_df, _ = parse_long_format(
                    df, value_col,
                    batch_col if batch_col != "（无批次列）" else None
                )
                measure_col = measure_df.columns[0]
            except Exception:
                measure_df = df[[value_col]].copy()
                measure_col = value_col
        else:
            col = st.selectbox("测量值列（自动检测）", numeric_cols, key="normal_auto_col")
            measure_df = df[[col]].copy()
            measure_col = col

        data = measure_df[measure_col].dropna().values

        if data_format == "多列 (展开)" and selected_cols:
            default_subgroup = len(selected_cols)
        elif is_example:
            default_subgroup = 5
        else:
            default_subgroup = 1
        subgroup_size = st.number_input("子组大小", min_value=1, max_value=len(data), value=default_subgroup, key="normal_subgroup")

        c1, c2, c3 = st.columns(3)
        with c1:
            lsl = st.number_input("规格下限 (LSL)", value=DEFAULT_LSL, format="%.4f")
        with c2:
            usl = st.number_input("规格上限 (USL)", value=DEFAULT_USL, format="%.4f")
        with c3:
            target = st.number_input("目标值 (Target)", value=DEFAULT_TARGET, format="%.4f")

    if st.button("开始分析", use_container_width=True, type="primary"):
        result = calculate_capability(data, lsl, usl, subgroup_size, target)
        stat, p = shapiro_test(data)
        grade = _capability_grade(result["cpk"])
        sigma_level = _sigma_level_from_ppm(result["ppm"])
        for key, val in [("normal_result", result), ("normal_p", p), ("normal_grade", grade),
                          ("normal_sigma", sigma_level), ("normal_lsl", lsl), ("normal_usl", usl),
                          ("normal_target", target), ("normal_data", data), ("normal_subgroup_size", subgroup_size)]:
            st.session_state[key] = val
        st.rerun()

    # 结果展示
    result = st.session_state["normal_result"]
    if result is None:
        return

    p = st.session_state["normal_p"]
    grade = st.session_state["normal_grade"]
    sigma_level = st.session_state["normal_sigma"]
    lsl = st.session_state["normal_lsl"]
    usl = st.session_state["normal_usl"]
    target = st.session_state["normal_target"]
    data = st.session_state["normal_data"]
    subgroup_size = st.session_state["normal_subgroup_size"]

    # KPI
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 关键指标概览")

    g_color = _grade_color(grade)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(kpi_card("Cp", f"{result['cp']:.2f}", "潜在组内能力"), unsafe_allow_html=True)
    k2.markdown(kpi_card("Cpk", f"{result['cpk']:.2f}", grade, color=g_color), unsafe_allow_html=True)
    k3.markdown(kpi_card("Pp", f"{result['pp']:.2f}", "整体性能"), unsafe_allow_html=True)
    k4.markdown(kpi_card("Ppk", f"{result['ppk']:.2f}", ""), unsafe_allow_html=True)
    cpm_str = f"{result['cpm']:.2f}" if "cpm" in result else "—"
    k5.markdown(kpi_card("Cpm", cpm_str, "田口损失函数"), unsafe_allow_html=True)

    k6, k7, k8, k9, k10 = st.columns(5)
    k6.markdown(kpi_card("均值", f"{result['mean']:.4f}", ""), unsafe_allow_html=True)
    k7.markdown(kpi_card("整体 σ", f"{result['std_overall']:.4f}", ""), unsafe_allow_html=True)
    k8.markdown(kpi_card("组内 σ", f"{result['std_within']:.4f}", ""), unsafe_allow_html=True)
    k9.markdown(kpi_card("PPM", f"{result['ppm']:.0f}", ""), unsafe_allow_html=True)
    is_normal = p > 0.05
    k10.markdown(kpi_card("正态性", "通过" if is_normal else "未通过",
                           f"P = {p:.4f}",
                           color=Colors.SUCCESS if is_normal else Colors.WARNING),
                 unsafe_allow_html=True)

    # 图表
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 图形分析")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["能力直方图", "箱线图", "Q-Q 图", "概率图", "子组控制图"])
    with tab1: st.pyplot(plot_capability(data, result["mean"], result["std_overall"], result["std_within"], lsl, usl, target))
    with tab2: st.pyplot(plot_boxplot(data))
    with tab3: st.pyplot(plot_qq(data))
    with tab4: st.pyplot(plot_probability(data))
    with tab5:
        if subgroup_size > 1: st.pyplot(plot_subgroup_means(data, subgroup_size))
        else: st.info("子组大小为 1，无法绘制")

    # 摘要
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        "指标": ["Cp", "CPL", "CPU", "Cpk", "Pp", "PPL", "PPU", "Ppk", "Cpm", "PPM", "Sigma"],
        "数值": [f"{result['cp']:.2f}", f"{result['cpl']:.2f}", f"{result['cpu']:.2f}",
                 f"{result['cpk']:.2f}", f"{result['pp']:.2f}", f"{result['ppl']:.2f}",
                 f"{result['ppu']:.2f}", f"{result['ppk']:.2f}",
                 f"{result.get('cpm', '—'):.2f}" if "cpm" in result else "—",
                 f"{result['ppm']:.0f}", f"{sigma_level}σ"],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # AI 智能分析
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### AI 智能分析")

    ai_answer = st.session_state.get("normal_ai_answer")
    if ai_answer:
        st.markdown(ai_answer)

    context = f"""过程能力分析结果：
均值：{result['mean']:.4f}  整体σ：{result['std_overall']:.4f}  组内σ：{result['std_within']:.4f}
Cp：{result['cp']:.2f}  Cpk：{result['cpk']:.2f}  Pp：{result['pp']:.2f}  Ppk：{result['ppk']:.2f}
Cpm：{result.get('cpm', 'N/A'):.2f}  PPM：{result['ppm']:.0f}
能力等级：{grade}  Sigma：{sigma_level}σ  Shapiro P：{p:.4f}"""

    if st.button("生成 AI 分析报告", key="normal_ai_btn", use_container_width=False):
        with st.spinner("AI 分析中..."):
            try:
                answer = ask_ai("请评价该过程能力并给出改进建议。", context)
                st.session_state["normal_ai_answer"] = answer
                st.rerun()
            except Exception as e:
                st.error(f"AI 调用失败：{e}")

    # PDF
    st.markdown('<div class="pca-divider"></div>', unsafe_allow_html=True)
    try:
        fig_cap = plot_capability(data, result["mean"], result["std_overall"], result["std_within"], lsl, usl, target)
        pdf_bytes = generate_normal_report(
            result=result, lsl=lsl, usl=usl, target=target, p_value=p,
            fig_capability=fig_cap, fig_qq=plot_qq(data),
            fig_boxplot=plot_boxplot(data), fig_probability=plot_probability(data),
            ai_analysis=ai_answer if ai_answer else "",
        )
        label = "导出 PDF 报告（含 AI 分析）" if ai_answer else "导出 PDF 报告"
        st.download_button(label=label, data=pdf_bytes, file_name="过程能力分析报告.pdf",
                           mime="application/pdf", key="normal_download_btn")
    except Exception:
        st.warning("PDF 生成失败，请确保已安装 reportlab")

# --- 入口 ---
normal_page()



