"""
PDF 报告生成模块
支持生成正态过程能力分析报告、二项分析报告、Box-Cox 变换分析报告
"""

import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    HRFlowable,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ==================================================
# 中文字体注册
# ==================================================

_FONT_REGISTERED = False


def _register_chinese_font():
    """注册中文字体，尝试多个常见路径"""
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return True

    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
        "C:/Windows/Fonts/simkai.ttf",     # 楷体
    ]

    for path in font_paths:
        try:
            pdfmetrics.registerFont(TTFont("ChineseFont", path))
            _FONT_REGISTERED = True
            return True
        except Exception:
            continue

    return False


def _get_font_name():
    """获取可用中文字体名，无中文字体则返回 Helvetica"""
    if _register_chinese_font():
        return "ChineseFont"
    return "Helvetica"


# ==================================================
# 样式定义
# ==================================================

def _get_styles(font_name):
    """构建 Paragraph 样式集合"""
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            "ChineseTitle",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=22,
            leading=30,
            spaceAfter=20,
            alignment=TA_CENTER,
        )
    )

    styles.add(
        ParagraphStyle(
            "ChineseH1",
            parent=styles["Heading1"],
            fontName=font_name,
            fontSize=16,
            leading=22,
            spaceBefore=16,
            spaceAfter=10,
        )
    )

    styles.add(
        ParagraphStyle(
            "ChineseH2",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=13,
            leading=18,
            spaceBefore=12,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            "ChineseBody",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=10,
            leading=16,
            spaceAfter=6,
        )
    )

    styles.add(
        ParagraphStyle(
            "ChineseSmall",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=8,
            leading=12,
            textColor=grey,
        )
    )

    return styles


# ==================================================
# 辅助函数
# ==================================================

def _capability_grade(cpk):
    """根据 Cpk 返回能力等级与颜色"""
    if cpk >= 1.67:
        return "A+ (特优)", HexColor("#1B5E20")
    elif cpk >= 1.33:
        return "A (良好)", HexColor("#2E7D32")
    elif cpk >= 1.00:
        return "B (尚可)", HexColor("#F9A825")
    elif cpk >= 0.67:
        return "C (不足)", HexColor("#E65100")
    else:
        return "D (严重不足)", HexColor("#B71C1C")


def _sigma_level_from_ppm(ppm):
    """PPM → Sigma 水平近似换算（长期）"""
    import numpy as np
    if ppm <= 0:
        return 6.0
    # 简化换算公式
    sigma = 0.8406 + np.sqrt(29.37 - 2.221 * np.log(max(ppm, 1e-10)))
    return round(sigma, 2)


def _build_indicator_table(results, font_name):
    """构建能力指标表格"""
    data = [
        ["指标", "数值", "指标", "数值"],
        ["均值", f"{results['mean']:.4f}", "整体标准差", f"{results['std_overall']:.4f}"],
        ["组内标准差", f"{results['std_within']:.4f}", "Cp", f"{results['cp']:.3f}"],
        ["Cpk", f"{results['cpk']:.3f}", "Pp", f"{results['pp']:.3f}"],
        ["Ppk", f"{results['ppk']:.3f}", "PPM", f"{results['ppm']:.0f}"],
    ]

    col_widths = [80, 80, 100, 80]
    table = Table(data, colWidths=col_widths)

    style = TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F5F5F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])

    table.setStyle(style)
    return table


def _build_grade_table(cpk, sigma_level, font_name):
    """构建评级表格"""
    grade, color = _capability_grade(cpk)

    data = [
        ["评价项目", "结果"],
        ["Cpk 值", f"{cpk:.3f}"],
        ["能力等级", grade],
        ["Sigma 水平", f"{sigma_level:.2f}σ"],
    ]

    col_widths = [120, 120]
    table = Table(data, colWidths=col_widths)

    style = TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
        ("BACKGROUND", (0, 3), (1, 3), color),
        ("TEXTCOLOR", (0, 3), (1, 3), white),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])

    table.setStyle(style)
    return table


# ==================================================
# 正态过程能力分析报告
# ==================================================


def generate_normal_report(
    result,
    lsl,
    usl,
    target,
    p_value,
    fig_capability=None,
    fig_qq=None,
    fig_boxplot=None,
    fig_probability=None,
    ai_analysis="",
    output_path=None,
):
    """
    生成正态过程能力分析 PDF 报告

    Parameters
    ----------
    result : dict
        calculate_capability 的返回结果
    lsl, usl, target : float
        规格限与目标值
    p_value : float
        Shapiro-Wilk 检验 P 值
    fig_capability, fig_qq, fig_boxplot, fig_probability : matplotlib.figure.Figure
        可选图形对象
    ai_analysis : str
        AI 分析文本
    output_path : str, optional
        输出文件路径，若为 None 则返回 bytes

    Returns
    -------
    bytes or None
    """
    font_name = _get_font_name()
    styles = _get_styles(font_name)

    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf if output_path is None else output_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    story = []

    # --- 标题 ---
    story.append(Paragraph("过程能力分析报告", styles["ChineseTitle"]))

    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#1565C0")))
    story.append(Spacer(1, 6 * mm))

    # --- 报告信息 ---
    info_table_data = [
        ["报告生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["分析类型", "正态过程能力分析"],
        ["规格下限 (LSL)", f"{lsl:.4f}"],
        ["规格上限 (USL)", f"{usl:.4f}"],
        ["目标值 (Target)", f"{target:.4f}"],
        ["样本数量", f"{int(0)}"],  # 需要外部传入
    ]

    # 从 result 推断样本数
    if "mean" in result:
        info_table_data[-1][1] = "—"

    info_table = Table(info_table_data, colWidths=[120, 200])
    info_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, HexColor("#F5F5F5")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ])
    )

    story.append(Spacer(1, 4 * mm))
    story.append(info_table)

    # --- 能力指标 ---
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("一、过程能力指标", styles["ChineseH1"]))

    indicator_table = _build_indicator_table(result, font_name)
    story.append(indicator_table)

    # --- 能力评级 ---
    sigma_level = _sigma_level_from_ppm(result.get("ppm", 0))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("二、能力评级", styles["ChineseH1"]))

    grade_table = _build_grade_table(result["cpk"], sigma_level, font_name)
    story.append(grade_table)

    # --- 正态性检验 ---
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("三、正态性检验", styles["ChineseH1"]))

    normality_table_data = [
        ["检验方法", "Shapiro-Wilk"],
        ["P 值", f"{p_value:.4f}"],
        ["结论", "数据服从正态分布 (P > 0.05)" if p_value > 0.05 else "数据不服从正态分布 (P ≤ 0.05)，建议使用 Box-Cox 变换"],
    ]

    normality_table = Table(normality_table_data, colWidths=[120, 300])
    normality_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, HexColor("#F5F5F5")]),
            ("BACKGROUND", (1, 2), (1, 2),
             HexColor("#E8F5E9") if p_value > 0.05 else HexColor("#FFEBEE")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )

    story.append(normality_table)

    # --- 图形 ---
    figures = [
        ("过程能力直方图", fig_capability),
        ("Q-Q 图", fig_qq),
        ("箱线图", fig_boxplot),
        ("正态概率图", fig_probability),
    ]

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("四、图形分析", styles["ChineseH1"]))

    for title, fig in figures:
        if fig is not None:
            story.append(Spacer(1, 4 * mm))
            story.append(Paragraph(title, styles["ChineseH2"]))

            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
            img_buf.seek(0)
            img = Image(img_buf, width=160 * mm, height=80 * mm)
            story.append(img)

    # --- AI 分析 ---
    if ai_analysis:
        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph("五、AI 智能分析", styles["ChineseH1"]))
        story.append(Paragraph(ai_analysis.replace("\n", "<br/>"), styles["ChineseBody"]))

    # --- 页脚 ---
    story.append(Spacer(1, 12 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=grey))
    story.append(
        Paragraph(
            f"本报告由 PCA 过程能力分析系统自动生成 | {datetime.now().strftime('%Y-%m-%d')}",
            styles["ChineseSmall"],
        )
    )

    # --- 构建 ---
    doc.build(story)

    if output_path is None:
        buf.seek(0)
        return buf.read()
    else:
        return None


# ==================================================
# 二项分析报告
# ==================================================


def generate_binomial_report(
    result,
    ai_analysis="",
    output_path=None,
    fig_p_chart=None,
    fig_cumulative=None,
    fig_distribution=None,
    fig_fit=None,
):
    """
    生成二项过程能力分析 PDF 报告

    Parameters
    ----------
    result : dict
        calculate_binomial_capability 的返回结果
    ai_analysis : str
        AI 分析文本
    output_path : str, optional
        输出文件路径，若为 None 则返回 bytes

    Returns
    -------
    bytes or None
    """
    font_name = _get_font_name()
    styles = _get_styles(font_name)

    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf if output_path is None else output_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    story = []

    # --- 标题 ---
    story.append(Paragraph("二项过程能力分析报告", styles["ChineseTitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#1565C0")))
    story.append(Spacer(1, 6 * mm))

    # --- 基本信息 ---
    info_data = [
        ["报告生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["分析类型", "二项过程能力分析"],
        ["总样本数", f"{result['total_samples']}"],
        ["总不合格数", f"{result['total_defects']}"],
        ["平均不合格率 (p̄)", f"{result['p_bar']:.4%}"],
    ]

    info_table = Table(info_data, colWidths=[140, 200])
    info_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, HexColor("#F5F5F5")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )

    story.append(info_table)

    # --- 能力指标 ---
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("一、能力指标", styles["ChineseH1"]))

    indicator_data = [
        ["指标", "数值"],
        ["PPM (百万分之不合格)", f"{result['ppm']:.0f}"],
        ["DPMO", f"{result['dpmo']:.0f}"],
        ["Sigma 水平", f"{result['sigma']:.2f}σ"],
        ["不合格率 p̄", f"{result['p_bar']:.4%}"],
    ]

    indicator_table = Table(indicator_data, colWidths=[180, 160])
    indicator_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F5F5F5")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(indicator_table)

    # --- 能力评级 ---
    sigma_val = result["sigma"]
    if sigma_val >= 6:
        grade, color = "世界级 (6σ)", HexColor("#1B5E20")
    elif sigma_val >= 5:
        grade, color = "优秀 (5σ)", HexColor("#2E7D32")
    elif sigma_val >= 4:
        grade, color = "良好 (4σ)", HexColor("#558B2F")
    elif sigma_val >= 3:
        grade, color = "一般 (3σ)", HexColor("#F9A825")
    elif sigma_val >= 2:
        grade, color = "较差 (2σ)", HexColor("#E65100")
    else:
        grade, color = "极差 (<2σ)", HexColor("#B71C1C")

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("二、能力评级", styles["ChineseH1"]))

    grade_data = [
        ["评价项目", "结果"],
        ["Sigma 水平", f"{sigma_val:.2f}σ"],
        ["能力等级", grade],
    ]

    grade_table = Table(grade_data, colWidths=[140, 140])
    grade_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("BACKGROUND", (0, 2), (1, 2), color),
            ("TEXTCOLOR", (0, 2), (1, 2), white),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(grade_table)

    # --- 图形 ---
    figures = [
        ("P 控制图", fig_p_chart),
        ("累计缺陷率趋势", fig_cumulative),
        ("缺陷率分布图", fig_distribution),
        ("二项分布拟合检验", fig_fit),
    ]
    if any(f is not None for _, f in figures):
        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph("三、图形分析", styles["ChineseH1"]))
        for title, fig in figures:
            if fig is not None:
                story.append(Spacer(1, 4 * mm))
                story.append(Paragraph(title, styles["ChineseH2"]))
                ib = io.BytesIO()
                fig.savefig(ib, format="png", dpi=150, bbox_inches="tight")
                ib.seek(0)
                story.append(Image(ib, width=160 * mm, height=80 * mm))
        sec = "四"
    else:
        sec = "三"

    # --- AI 分析 ---
    if ai_analysis:
        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph(f"{sec}、AI 智能分析", styles["ChineseH1"]))
        story.append(Paragraph(ai_analysis.replace("\n", "<br/>"), styles["ChineseBody"]))

    # --- 页脚 ---
    story.append(Spacer(1, 12 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=grey))
    story.append(
        Paragraph(
            f"本报告由 PCA 过程能力分析系统自动生成 | {datetime.now().strftime('%Y-%m-%d')}",
            styles["ChineseSmall"],
        )
    )

    doc.build(story)

    if output_path is None:
        buf.seek(0)
        return buf.read()
    else:
        return None


# ==================================================
# Box-Cox 变换分析报告
# ==================================================


def generate_boxcox_report(
    result,
    lam,
    p_orig,
    p_trans,
    lsl,
    usl,
    target,
    data_orig,
    data_trans,
    ai_analysis="",
    output_path=None,
    fig_hist=None,
    fig_qq=None,
    fig_prob=None,
    fig_cap=None,
):
    """
    生成 Box-Cox 变换分析 PDF 报告

    Parameters
    ----------
    result : dict
        变换后 calculate_capability 的返回结果
    lam : float
        Box-Cox 最优 λ
    p_orig : float
        原始数据 Shapiro-Wilk P 值
    p_trans : float
        变换后 Shapiro-Wilk P 值
    lsl, usl, target : float
        原始规格限
    data_orig, data_trans : np.ndarray
        原始数据与变换后数据
    ai_analysis : str
        AI 分析文本
    output_path : str, optional

    Returns
    -------
    bytes or None
    """
    import numpy as np

    font_name = _get_font_name()
    styles = _get_styles(font_name)

    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf if output_path is None else output_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    story = []

    # --- 标题 ---
    story.append(Paragraph("Box-Cox 变换分析报告", styles["ChineseTitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#1565C0")))
    story.append(Spacer(1, 6 * mm))

    # --- 基本信息 ---
    info_data = [
        ["报告生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["分析类型", "Box-Cox 变换分析"],
        ["原始规格下限 (LSL)", f"{lsl:.4f}" if lsl is not None else "—"],
        ["原始规格上限 (USL)", f"{usl:.4f}"],
        ["原始目标值 (Target)", f"{target:.4f}"],
        ["最优 λ 值", f"{lam:.4f}"],
    ]

    info_table = Table(info_data, colWidths=[140, 200])
    info_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [white, HexColor("#F5F5F5")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )

    story.append(info_table)

    # --- 正态性前后对比 ---
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("一、正态性检验对比", styles["ChineseH1"]))

    normality_data = [
        ["阶段", "Shapiro-Wilk P 值", "结论"],
        ["变换前", f"{p_orig:.4f}",
         "正态" if p_orig > 0.05 else "非正态"],
        ["变换后", f"{p_trans:.4f}",
         "正态" if p_trans > 0.05 else "非正态"],
    ]

    normality_table = Table(normality_data, colWidths=[80, 140, 120])
    normality_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(normality_table)

    # --- 变换前后统计量 ---
    story.append(Spacer(1, 4 * mm))

    stat_data = [
        ["统计量", "原始数据", "变换后数据"],
        ["均值", f"{np.mean(data_orig):.4f}", f"{np.mean(data_trans):.4f}"],
        ["标准差", f"{np.std(data_orig, ddof=1):.4f}", f"{np.std(data_trans, ddof=1):.4f}"],
        ["最小值", f"{np.min(data_orig):.4f}", f"{np.min(data_trans):.4f}"],
        ["最大值", f"{np.max(data_orig):.4f}", f"{np.max(data_trans):.4f}"],
    ]

    stat_table = Table(stat_data, colWidths=[80, 130, 130])
    stat_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F5F5F5")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(stat_table)

    # --- 变换后能力指标 ---
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("二、变换后能力指标", styles["ChineseH1"]))

    indicator_table = _build_indicator_table(result, font_name)
    story.append(indicator_table)

    # --- 能力评级 ---
    sigma_level = _sigma_level_from_ppm(result.get("ppm", 0))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("三、能力评级", styles["ChineseH1"]))

    grade_table = _build_grade_table(result["cpk"], sigma_level, font_name)
    story.append(grade_table)

    # --- 图形 ---
    figures = [
        ("直方图对比", fig_hist),
        ("Q-Q 图对比", fig_qq),
        ("概率图对比", fig_prob),
        ("变换后能力直方图", fig_cap),
    ]
    if any(f is not None for _, f in figures):
        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph("四、图形分析", styles["ChineseH1"]))
        for title, fig in figures:
            if fig is not None:
                story.append(Spacer(1, 4 * mm))
                story.append(Paragraph(title, styles["ChineseH2"]))
                ib = io.BytesIO()
                fig.savefig(ib, format="png", dpi=150, bbox_inches="tight")
                ib.seek(0)
                story.append(Image(ib, width=160 * mm, height=80 * mm))
        sec = "五"
    else:
        sec = "四"

    # --- AI 分析 ---
    if ai_analysis:
        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph(f"{sec}、AI 智能分析", styles["ChineseH1"]))
        story.append(Paragraph(ai_analysis.replace("\n", "<br/>"), styles["ChineseBody"]))

    # --- 页脚 ---
    story.append(Spacer(1, 12 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=grey))
    story.append(
        Paragraph(
            f"本报告由 PCA 过程能力分析系统自动生成 | {datetime.now().strftime('%Y-%m-%d')}",
            styles["ChineseSmall"],
        )
    )

    doc.build(story)

    if output_path is None:
        buf.seek(0)
        return buf.read()
    else:
        return None


# ==================================================
# 计量控制图报告 (Xbar-R / I-MR)
# ==================================================

def generate_metrology_report(
    result, chart_type, auto_eval="", ai_analysis="",
    fig_chart=None,
):
    font_name = _get_font_name()
    styles = _get_styles(font_name)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=15*mm, bottomMargin=15*mm)
    story = []

    title_text = "常规计量控制图分析报告" if chart_type == "xbar_r" else "单值-移动极差控制图分析报告"
    sub_text = "Xbar-R Control Chart" if chart_type == "xbar_r" else "I-MR Control Chart"

    story.append(Paragraph(title_text, styles["ChineseTitle"]))
    story.append(Paragraph(sub_text, styles["ChineseSmall"]))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("报告生成时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["ChineseSmall"]))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#1565C0")))
    story.append(Spacer(1, 8*mm))

    story.append(Paragraph("一、关键指标", styles["ChineseH1"]))
    if chart_type == "xbar_r":
        data = [
            ["指标", "数值", "指标", "数值"],
            ["Xbarbar", "%.4f" % result["xbarbar"], "Rbar", "%.4f" % result["rbar"]],
            ["Xbar UCL", "%.4f" % result["xbar_ucl"], "Xbar LCL", "%.4f" % result["xbar_lcl"]],
            ["R UCL", "%.4f" % result["r_ucl"], "R LCL", "%.4f" % result["r_lcl"]],
            ["批次", str(result.get("n_subgroups", "")), "子组大小", str(result.get("subgroup_size", ""))],
        ]
    else:
        data = [
            ["指标", "数值", "指标", "数值"],
            ["均值", "%.4f" % result["mean"], "MRbar", "%.4f" % result["mrbar"]],
            ["I UCL", "%.4f" % result["i_ucl"], "I LCL", "%.4f" % result["i_lcl"]],
            ["MR UCL", "%.4f" % result["mr_ucl"], "观测数", str(result.get("n", ""))],
        ]
    table = Table(data, colWidths=[80, 100, 80, 100])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)

    if auto_eval:
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph("二、评价结论", styles["ChineseH1"]))
        clean = auto_eval.replace("##### ", "").replace("**", "")
        story.append(Paragraph(clean.replace("\n", "<br/>"), styles["ChineseBody"]))

    if fig_chart is not None:
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph("三、控制图", styles["ChineseH1"]))
        ib = io.BytesIO()
        fig_chart.savefig(ib, format="png", dpi=150, bbox_inches="tight")
        ib.seek(0)
        story.append(Image(ib, width=170*mm, height=113*mm))

    if ai_analysis:
        sec = "四" if fig_chart is not None else "三"
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph(sec + "、AI 智能分析", styles["ChineseH1"]))
        story.append(Paragraph(ai_analysis.replace("\n", "<br/>"), styles["ChineseBody"]))

    story.append(Spacer(1, 12*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=grey))
    story.append(Paragraph("本报告由 PCA 系统自动生成 | " + datetime.now().strftime("%Y-%m-%d"), styles["ChineseSmall"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ==================================================
# 计数控制图报告 (P / NP / U / C)
# ==================================================

def generate_attributes_report(
    result, chart_type, auto_eval="", ai_analysis="",
    fig_chart=None,
):
    font_name = _get_font_name()
    styles = _get_styles(font_name)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=15*mm, bottomMargin=15*mm)
    story = []

    type_names = {"np": "NP 控制图 (不合格品数)", "p": "P 控制图 (不合格率)", "c": "C 控制图 (缺陷数)", "u": "U 控制图 (单位缺陷数)"}
    title_text = type_names.get(chart_type, "计数控制图分析报告")

    story.append(Paragraph(title_text, styles["ChineseTitle"]))
    story.append(Paragraph("Attributes Control Chart", styles["ChineseSmall"]))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("报告生成时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["ChineseSmall"]))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#1565C0")))
    story.append(Spacer(1, 8*mm))

    story.append(Paragraph("一、关键指标", styles["ChineseH1"]))
    if chart_type == "np":
        row_data = [
            ["指标", "数值", "指标", "数值"],
            ["平均NP", "%.2f" % result["np_bar"], "UCL", "%.2f" % result["ucl"][0]],
            ["LCL", "%.2f" % result["lcl"][0], "p-bar", "%.4f%%" % (result.get("p_bar", 0)*100)],
            ["子组大小", str(result.get("n_size", "")), "批次数", str(result.get("n", ""))],
        ]
    elif chart_type == "p":
        row_data = [
            ["指标", "数值"],
            ["p-bar (平均不合格率)", "%.4f%%" % (result["p_bar"]*100)],
            ["UCL范围", "%.4f ~ %.4f" % (result["ucl"].min(), result["ucl"].max())],
            ["LCL范围", "%.4f ~ %.4f" % (result["lcl"].min(), result["lcl"].max())],
            ["批次数", str(result.get("n", ""))],
        ]
    elif chart_type == "c":
        row_data = [
            ["指标", "数值", "指标", "数值"],
            ["c-bar", "%.2f" % result["c_bar"], "UCL", "%.2f" % result["ucl"]],
            ["LCL", "%.2f" % result["lcl"], "批次数", str(result.get("n", ""))],
        ]
    else:
        row_data = [
            ["指标", "数值"],
            ["u-bar (平均单位缺陷)", "%.4f" % result["u_bar"]],
            ["UCL范围", "%.4f ~ %.4f" % (result["ucl"].min(), result["ucl"].max())],
            ["LCL范围", "%.4f ~ %.4f" % (result["lcl"].min(), result["lcl"].max())],
            ["批次数", str(result.get("n", ""))],
        ]

    col_w = [100, 120, 100, 120] if len(row_data[0]) == 4 else [140, 200]
    table = Table(row_data, colWidths=col_w)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)

    if auto_eval:
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph("二、评价结论", styles["ChineseH1"]))
        story.append(Paragraph(auto_eval.replace("##### ", "").replace("**", "").replace("\n", "<br/>"), styles["ChineseBody"]))

    if fig_chart is not None:
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph("三、控制图", styles["ChineseH1"]))
        ib = io.BytesIO()
        fig_chart.savefig(ib, format="png", dpi=150, bbox_inches="tight")
        ib.seek(0)
        story.append(Image(ib, width=170*mm, height=80*mm))

    if ai_analysis:
        sec = "四" if fig_chart is not None else "三"
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph(sec + "、AI 智能分析", styles["ChineseH1"]))
        story.append(Paragraph(ai_analysis.replace("\n", "<br/>"), styles["ChineseBody"]))

    story.append(Spacer(1, 12*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=grey))
    story.append(Paragraph("本报告由 PCA 系统自动生成 | " + datetime.now().strftime("%Y-%m-%d"), styles["ChineseSmall"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ==================================================
# 小批量控制图报告
# ==================================================

def generate_small_batch_report(
    chart_type, auto_eval="", ai_analysis="",
    fig_chart=None, fig_chart2=None,
):
    font_name = _get_font_name()
    styles = _get_styles(font_name)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=15*mm, bottomMargin=15*mm)
    story = []

    type_names = {"target": "目标控制图", "ratio": "比例控制图", "standardized": "标准变换控制图"}
    story.append(Paragraph(type_names.get(chart_type, "小批量控制图分析报告"), styles["ChineseTitle"]))
    story.append(Paragraph("Small Batch SPC", styles["ChineseSmall"]))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("报告生成时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["ChineseSmall"]))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#1565C0")))
    story.append(Spacer(1, 8*mm))

    if auto_eval:
        story.append(Paragraph("一、评价结论", styles["ChineseH1"]))
        story.append(Paragraph(auto_eval.replace("##### ", "").replace("**", "").replace("\n", "<br/>"), styles["ChineseBody"]))

    if fig_chart is not None:
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph("二、控制图", styles["ChineseH1"]))
        ib = io.BytesIO()
        fig_chart.savefig(ib, format="png", dpi=150, bbox_inches="tight")
        ib.seek(0)
        story.append(Image(ib, width=170*mm, height=100*mm))

    if fig_chart2 is not None:
        story.append(Spacer(1, 4*mm))
        ib2 = io.BytesIO()
        fig_chart2.savefig(ib2, format="png", dpi=150, bbox_inches="tight")
        ib2.seek(0)
        story.append(Image(ib2, width=170*mm, height=100*mm))

    if ai_analysis:
        sec = "三"
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph(sec + "、AI 智能分析", styles["ChineseH1"]))
        story.append(Paragraph(ai_analysis.replace("\n", "<br/>"), styles["ChineseBody"]))

    story.append(Spacer(1, 12*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=grey))
    story.append(Paragraph("本报告由 PCA 系统自动生成 | " + datetime.now().strftime("%Y-%m-%d"), styles["ChineseSmall"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ==================================================
# 小波动控制图报告 (CUSUM / EWMA)
# ==================================================

def generate_small_shift_report(
    result, chart_type, auto_eval="", ai_analysis="",
    fig_chart=None, fig_imr=None,
):
    font_name = _get_font_name()
    styles = _get_styles(font_name)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=15*mm, bottomMargin=15*mm)
    story = []

    title_text = "CUSUM 累积和控制图分析报告" if chart_type == "cusum" else "EWMA 控制图分析报告"
    sub_text = "Cumulative Sum Control Chart" if chart_type == "cusum" else "Exponentially Weighted Moving Average"

    story.append(Paragraph(title_text, styles["ChineseTitle"]))
    story.append(Paragraph(sub_text, styles["ChineseSmall"]))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("报告生成时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["ChineseSmall"]))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#1565C0")))
    story.append(Spacer(1, 8*mm))

    story.append(Paragraph("一、关键指标", styles["ChineseH1"]))
    if chart_type == "cusum":
        row_data = [
            ["指标", "数值", "指标", "数值"],
            ["目标值", "%.4f" % result["target"], "k值", str(result.get("k", "-"))],
            ["h值", str(result.get("h", "-")), "Max C+", "%.4f" % result["c_plus"].max()],
            ["Max C-", "%.4f" % result["c_minus"].max(), "观测数", str(result.get("n", ""))],
        ]
    else:
        # Use Unicode lambda
        lam = "λ"
        sig = "σ"
        row_data = [
            ["指标", "数值", "指标", "数值"],
            ["目标值", "%.4f" % result["target"], lam, str(result.get("lam", "-"))],
            [sig, "%.4f" % result.get("sigma", 0), "最终UCL", "%.4f" % result["ucl"][-1]],
            ["最终LCL", "%.4f" % result["lcl"][-1], "观测数", str(result.get("n", ""))],
        ]
    table = Table(row_data, colWidths=[80, 100, 80, 100])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1565C0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)

    if auto_eval:
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph("二、评价结论", styles["ChineseH1"]))
        story.append(Paragraph(auto_eval.replace("##### ", "").replace("**", "").replace("\n", "<br/>"), styles["ChineseBody"]))

    if fig_imr is not None:
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph("三、I-MR 控制图", styles["ChineseH1"]))
        ib = io.BytesIO()
        fig_imr.savefig(ib, format="png", dpi=150, bbox_inches="tight")
        ib.seek(0)
        story.append(Image(ib, width=170*mm, height=100*mm))

    if fig_chart is not None:
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph("四、小波动检测控制图", styles["ChineseH1"]))
        ib2 = io.BytesIO()
        fig_chart.savefig(ib2, format="png", dpi=150, bbox_inches="tight")
        ib2.seek(0)
        story.append(Image(ib2, width=170*mm, height=100*mm))

    if ai_analysis:
        sec = "五"
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph(sec + "、AI 智能分析", styles["ChineseH1"]))
        story.append(Paragraph(ai_analysis.replace("\n", "<br/>"), styles["ChineseBody"]))

    story.append(Spacer(1, 12*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=grey))
    story.append(Paragraph("本报告由 PCA 系统自动生成 | " + datetime.now().strftime("%Y-%m-%d"), styles["ChineseSmall"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()
