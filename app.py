import streamlit as st

st.set_page_config(
    page_title="PCA 质量分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.theme import apply_theme, Colors

apply_theme()

# ==================================================
# 侧边栏 — Quality AI 品牌标识
# ==================================================

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">
        <div style="width:32px;height:32px;background:var(--pca-primary);border-radius:6px;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:1rem;">Q</div>
        <div>
            <div style="font-weight:600;font-size:0.9rem;color:var(--pca-text-primary);">Quality AI</div>
            <div style="font-size:0.7rem;color:var(--pca-text-muted);">智能质量助手</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("— 导航 —")

    st.markdown("---")
    st.caption("— 系统状态 —")
    st.markdown("""
    <div style="font-size:0.75rem;color:var(--pca-text-muted);line-height:1.6;">
        <div class="pca-status"><span class="pca-dot pca-dot-success"></span> 系统就绪</div>
        <div style="margin-top:0.25rem;">当前时间：—</div>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# 欢迎首页
# ==================================================

st.markdown("""
<div style="margin-bottom:1.5rem;">
    <div style="font-size:1.5rem;font-weight:700;color:var(--pca-text-primary);">过程能力分析平台</div>
    <div style="font-size:0.85rem;color:var(--pca-text-secondary);">Statistical Process Control · Quality Analytics · Six Sigma</div>
</div>
""", unsafe_allow_html=True)

# 第一行功能卡片 (4列)
row1_cols = st.columns(4)
cards_row1 = [
    ("正态过程能力分析", "Cp · Cpk · Pp · Ppk\n能力评级 · Sigma 水平\n控制图 · 正态性检验",
     "适用于计量型数据的标准\n过程能力评估"),
    ("二项过程能力分析", "不合格率 · PPM · DPMO\nSigma 水平 · P 控制图\n计数型数据质量评价",
     "适用于合格/不合格\n判定数据"),
    ("Box-Cox 变换分析", "非正态数据幂变换\n最优λ求解\n变换前后分布对比",
     "适用于非正态分布的\n数据转换分析"),
    ("常规计量控制图", "Xbar-R · I-MR\n均值-极差 · 单值-移动极差\n判异准则自动检测",
     "适用于正态计量型\n常规控制图"),
]
icons_row1 = ["N", "B", "C", "M"]

for i, (title, features, desc) in enumerate(cards_row1):
    with row1_cols[i]:
        st.markdown(f"""
        <div class="pca-feature-card">
            <div style="width:36px;height:36px;background:var(--pca-primary);border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:1rem;margin-bottom:0.75rem;">{icons_row1[i]}</div>
            <div class="pca-feature-title">{title}</div>
            <div class="pca-feature-body">{features}</div>
            <div class="pca-feature-footer">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

# 第二行功能卡片 (3列 + 空)
row2_cols = st.columns(4)
cards_row2 = [
    ("常规计数控制图", "P · NP · U · C\n计件/计点型控制图\n不等样本量自动处理",
     "适用于计数型\n离散数据控制图"),
    ("小批量控制图技术", "目标图 · 比例图\n标准变换控制图\n多品种小批量场景",
     "适用于样本量不足\n的小批量生产过程"),
    ("小波动控制图技术", "CUSUM · EWMA\n累积和 · 指数加权移动平均\n微小偏移检测",
     "适用于检测过程均值\n微小偏移的高级控制图"),
]
icons_row2 = ["A", "S", "W"]

for i, (title, features, desc) in enumerate(cards_row2):
    with row2_cols[i]:
        st.markdown(f"""
        <div class="pca-feature-card">
            <div style="width:36px;height:36px;background:var(--pca-primary);border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:1rem;margin-bottom:0.75rem;">{icons_row2[i]}</div>
            <div class="pca-feature-title">{title}</div>
            <div class="pca-feature-body">{features}</div>
            <div class="pca-feature-footer">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# 快速开始
st.markdown("##### 快速开始")

steps_col1, steps_col2, steps_col3, steps_col4, steps_col5 = st.columns(5)

steps = [
    ("1", "选择模块", "从左侧导航栏\n进入分析页面"),
    ("2", "导入数据", "上传 Excel/CSV\n或使用示例数据"),
    ("3", "设置参数", "配置规格限\n与子组大小"),
    ("4", "执行分析", "一键计算\n查看完整报告"),
    ("5", "导出结果", "AI 解读 +\nPDF 报告导出"),
]

for col, (num, title, desc) in zip(
    [steps_col1, steps_col2, steps_col3, steps_col4, steps_col5], steps
):
    with col:
        st.markdown(f"""
        <div style="text-align:center;">
            <div class="pca-step-circle">{num}</div>
            <div style="font-weight:600;font-size:0.85rem;color:var(--pca-text-primary);margin-bottom:0.25rem;">{title}</div>
            <div style="font-size:0.75rem;color:var(--pca-text-muted);white-space:pre-line;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("SPC 过程能力分析平台 · Enterprise Quality Analytics")
