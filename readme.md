# PCA System — 过程能力分析平台

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于 Python + Streamlit 的 **统计过程控制（SPC）全流程分析平台**，完整复现 Minitab 核心质量控制功能，集成 AI 智能解读。

> 南京航空航天大学 · 经济与管理学院 · 统计过程控制课程实验项目

## 功能模块

| 模块 | 功能 | 数据类型 |
|------|------|----------|
| **正态过程能力分析** | Cp / Cpk / Pp / Ppk / Cpm / Sigma 水平 | 计量型（正态） |
| **二项过程能力分析** | 不合格率 / PPM / DPMO / P 控制图 | 计数型（合格/不合格） |
| **Box-Cox 变换分析** | 最优 λ 求解、变换前后对比 | 计量型（非正态） |
| **常规计量控制图** | Xbar-R / I-MR + 判异准则 | 计量型子组/单值 |
| **常规计数控制图** | P / NP / U / C 四种控制图 | 计数型 |
| **小批量控制图** | 目标图 / 比例图 / 标准变换 | 多品种小批量 |
| **小波动控制图** | CUSUM / EWMA 微小偏移检测 | 计量型单值 |
| **全流程分析智能体** | 自动识别数据结构 → 推荐方法 → 一键分析 → AI 解读 | 通用 |

## 界面预览

详见 `界面/` 目录下的截图（首页、正态分析主页、分析结果等）。

## 项目结构

```
PCA_System/
├── app.py                      # Streamlit 首页入口
├── config.py                   # 全局配置
│
├── pages/
│   ├── 00_AI_Agent.py          # 全流程分析智能体
│   ├── 1_📊_正态分析.py         # 正态过程能力分析
│   ├── 2_📐_二项分析.py         # 二项过程能力分析
│   ├── 3_📈_BoxCox分析.py      # Box-Cox 变换分析
│   ├── 5_📋_计量控制图.py       # 常规计量控制图
│   ├── 6_🔢_计数控制图.py       # 常规计数控制图
│   ├── 7_📐_小批量控制图.py     # 小批量控制图
│   └── 8_📉_小波动控制图.py     # 小波动控制图
│
├── utils/
│   ├── agent_engine.py         # 智能体引擎（数据分类 + 全自动工作流）
│   ├── ai_assistant.py         # AI API 调用（DeepSeek）
│   ├── binomial_capability.py  # 二项过程能力计算
│   ├── boxcox_utils.py         # Box-Cox 变换工具
│   ├── capability.py           # 正态过程能力核心计算
│   ├── data_loader.py          # 文件解析（CSV / Excel）
│   ├── normality_test.py       # 正态性检验（Shapiro-Wilk）
│   ├── plotting.py             # 图表绘制（Matplotlib）
│   ├── report_generator.py     # PDF 报告生成（ReportLab）
│   ├── theme.py                # 企业级 UI 主题
│   └── pca_theme.css           # 双主题 CSS
│
├── data/                       # 示例数据
├── 界面/                       # 界面截图
└── requirements.txt
```

## 快速开始

### 环境要求
- Python 3.9+
- Windows / macOS / Linux

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/PCA_System.git
cd PCA_System

# 安装依赖
pip install -r requirements.txt
```

### 运行

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`，从左侧导航栏选择分析模块开始使用。

### 使用示例数据

每个分析模块均内置了与《统计过程控制实验指导书》完全一致的示例数据，直接选择"使用示例数据"即可体验完整分析流程。

## 技术栈

| 层级 | 技术 |
|------|------|
| Web 框架 | Streamlit |
| 数值计算 | NumPy / SciPy / Pandas |
| 图表绘制 | Matplotlib |
| PDF 报告 | ReportLab |
| AI 集成 | DeepSeek API（OpenAI 兼容） |
| 文档处理 | python-docx |

## 全流程分析智能体

系统的核心亮点。用户只需上传数据文件，无需任何统计学知识即可完成全流程分析：

```
上传数据 → 数据画像 → 结构识别 → 自动分析 → AI 解读 → PDF 导出
```

- **三层数据分类引擎**：自动识别计数型/宽格式子组/单列连续数据
- **二级缓存加速**：同一文件再次上传秒级加载
- **规格限自动推算**：未设置时自动取 ±3σ 范围
- **AI 双视角解读**：管理层摘要 + 质量工程师建议

## 开发

本项目全部代码借助 **OpenAI Codex CLI** 辅助开发，历经八个迭代阶段完成。

