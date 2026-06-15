"""
PCA System — 企业级视觉主题
通过 pca_theme.css 实现完整双主题（浅色/深色自动跟随系统）
"""

import streamlit as st
from pathlib import Path

# ==================================================
# 企业配色常量
# ==================================================

class Colors:
    PRIMARY = "#2563eb"
    PRIMARY_LIGHT = "#3b82f6"
    PRIMARY_DARK = "#1d4ed8"

    SUCCESS = "#059669"
    WARNING = "#d97706"
    DANGER = "#dc2626"
    INFO = "#0891b2"

    TEXT_SECONDARY = "#475569"

    SIGMA_COLORS = {
        6: "#059669", 5: "#10b981", 4: "#22c55e",
        3: "#eab308", 2: "#f97316", 1: "#ef4444", 0: "#dc2626",
    }

    CAPABILITY_GRADE_COLORS = {
        "A+ (特优)": "#059669", "A (良好)": "#10b981",
        "B (尚可)": "#eab308", "C (不足)": "#f97316",
        "D (严重不足)": "#dc2626",
    }

    WORLD_CLASS = {
        "世界级 (6σ)": "#059669", "优秀 (5σ)": "#10b981",
        "良好 (4σ)": "#22c55e", "一般 (3σ)": "#eab308",
        "较差 (2σ)": "#f97316", "极差 (<2σ)": "#dc2626",
    }


# ==================================================
# 主题注入
# ==================================================

_CSS_PATH = Path(__file__).parent / "pca_theme.css"

def apply_theme():
    """注入完整双主题 CSS（从 pca_theme.css 加载）"""
    if _CSS_PATH.exists():
        css = _CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ==================================================
# KPI 卡片
# ==================================================

def kpi_card(label, value, sub="", color=None, delta=None):
    color_style = f"color:{color};" if color else ""
    delta_html = ""
    if delta:
        d = Colors.SUCCESS if delta >= 0 else Colors.DANGER
        s = "+" if delta >= 0 else ""
        delta_html = f'<div style="font-size:0.8rem;color:{d};">{s}{delta}</div>'
    return (
        f'<div class="pca-card">'
        f'<div class="pca-card-header">{label}</div>'
        f'<div class="pca-kpi-value" style="{color_style}">{value}</div>'
        f'{delta_html}'
        f'<div class="pca-kpi-sub">{sub}</div>'
        f'</div>'
    )


def grade_badge(grade, colors):
    c = colors.get(grade, Colors.DANGER)
    return f'<span class="pca-badge" style="background:{c};color:#fff;">{grade}</span>'


def status_indicator(ok, ok_text="受控", fail_text="异常"):
    if ok:
        return f'<span class="pca-status"><span class="pca-dot pca-dot-success"></span> {ok_text}</span>'
    return f'<span class="pca-status"><span class="pca-dot pca-dot-danger"></span> {fail_text}</span>'


# ==================================================
# 品控顾问 侧边栏
# ==================================================

def render_copilot_sidebar():
    from utils.ai_assistant import ask_ai

    with st.sidebar:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
            <span style="font-size:1rem;">🤖</span>
            <span style="font-weight:600;font-size:0.85rem;">品控顾问</span>
            <span style="font-size:0.7rem;color:var(--text-muted);">质量分析助手</span>
        </div>
        """, unsafe_allow_html=True)

        if "copilot_history" not in st.session_state:
            st.session_state["copilot_history"] = []

        cc = st.container(height=320)
        with cc:
            if not st.session_state["copilot_history"]:
                st.markdown(
                    '<div style="font-size:0.78rem;color:var(--text-muted);'
                    'padding:1rem 0;text-align:center;">'
                    '可以问我任何质量分析相关问题</div>',
                    unsafe_allow_html=True,
                )
            for msg in st.session_state["copilot_history"]:
                role = msg["role"]
                cls = "pca-chat-user" if role == "user" else "pca-chat-ai"
                label = "You" if role == "user" else "品控顾问"
                st.markdown(
                    f'<div style="font-size:0.68rem;color:var(--text-muted);'
                    f'margin-bottom:0.12rem;">{label}</div>'
                    f'<div class="pca-chat-bubble {cls}">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

        with st.form("copilot_form", clear_on_submit=True):
            ui = st.text_input(
                "输入", key="copilot_input",
                placeholder="输入问题...", label_visibility="collapsed",
            )
            submitted = st.form_submit_button("发送", use_container_width=True)

        if submitted and ui:
            st.session_state["copilot_history"].append({"role": "user", "content": ui})
            with st.spinner("思考中..."):
                try:
                    ctxs = []
                    for k, lab in [
                        ("normal_result", "正态"), ("bin_result", "二项"), ("bc_result", "BoxCox"), ("met_result", "计量"), ("attr_result", "计数"), ("sb_result", "小批量"), ("sw_result", "小波动"),
                    ]:
                        r = st.session_state.get(k)
                        if r:
                            ctxs.append(f"[{lab}] Cpk:{r.get('cpk','?'):.2f} PPM:{r.get('ppm','?'):.0f}")
                    cx = "\n".join(ctxs) if ctxs else "无分析结果"
                    a = ask_ai(ui, cx)
                    st.session_state["copilot_history"].append({"role": "assistant", "content": a})
                except Exception as e:
                    st.session_state["copilot_history"].append(
                        {"role": "assistant", "content": f"调用失败：{e}"}
                    )
            st.rerun()

        if st.session_state["copilot_history"]:
            if st.button("清空对话", key="clear_copilot", use_container_width=True):
                st.session_state["copilot_history"] = []
                st.rerun()


# ==================================================
# 图表主题
# ==================================================

import matplotlib.pyplot as plt
import matplotlib as mpl

CHART_THEME = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#cbd5e1",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#cbd5e1",
    "axes.labelcolor": "#475569",
    "text.color": "#334155",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#e2e8f0",
    "legend.framealpha": 0.9,
    "lines.linewidth": 1.8,
}


def apply_chart_theme():
    for k, v in CHART_THEME.items():
        mpl.rcParams[k] = v

