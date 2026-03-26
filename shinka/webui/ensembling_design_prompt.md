# Ensembling Tab 重设计任务

## 上下文

你在为 ShinkaEvolve（一个进化代码优化框架）的 WebUI 设计 Ensembling tab。
该 tab 展示多模型集成（UCB bandit）的数据：每个 LLM 模型的后验概率变化、程序生成数量、成功率等。

## 设计语言参考

已有的 Path tab 使用三合一布局，风格是：
- 深色主题 (#0D0D0D 背景, #1A1A1A 面板, #242424 卡片)
- 紫蓝渐变强调色 (linear-gradient #4285F4 → #8A60F5)
- Google Sans 字体 + Roboto Mono 数据字体
- section title: 12px uppercase, #666 (light) / #9AA0A6 (dark), border-bottom 分隔
- 数据高亮用渐变 clip text
- 交互：hover 发光边框, click 高亮联动

## 数据结构

每个 program 节点有:
```json
{
  "generation": 3,
  "metadata": {
    "model_name": "doubao-seed-2.0-code",
    "llm_result": {
      "model_posteriors": {
        "doubao-seed-2.0-code": 0.18,
        "glm-4.7": 0.15,
        "minimax-m2.5": 0.12,
        "deepseek-v3.2": 0.25,
        ...
      }
    }
  },
  "combined_score": 0.834,
  "correct": true
}
```

## 需要展示的信息

1. **模型概览** — 每个模型的总程序数、成功率、平均分、当前后验概率
2. **后验概率随代数变化** — 折线图，展示 UCB bandit 如何学习偏好不同模型
3. **累计程序数** — 堆叠面积图或折线图，展示每个模型随时间贡献的程序数
4. **模型对比表** — 详细数据表格

## 要求

1. 输出完整的 HTML+CSS+JS 原型（自包含，用硬编码 mock 数据）
2. 三合一布局：顶部概览卡片 → 中间两个图表 → 底部数据表
3. 风格必须跟 Path tab 一致（深色主题、渐变、字体）
4. 支持 dark mode class（body.dark）
5. 图表用纯 SVG 手写（不依赖 Plotly/D3），保持轻量
6. 每个模型用不同颜色，提供 8 色调色板
7. 概览卡片需展示：模型名、程序数、成功率、平均分、后验概率 bar
8. 后验曲线需要 tooltip 悬停显示具体数值
9. 表格支持按列排序（点击表头）
10. 所有文案用中英双语注释标注（方便后续 i18n 接入）

输出一个完整可运行的 HTML 文件，不需要任何外部依赖（除了 Google Sans 字体 CDN）。
