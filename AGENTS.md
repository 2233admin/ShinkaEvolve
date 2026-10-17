# AGENTS.md（fork：2233admin/ShinkaEvolve）

本 fork 在上游 SakanaAI/ShinkaEvolve 之上开发 **Shinka Research**（自动研究层）。
先读 [docs/research/design.md](docs/research/design.md) 和 [docs/research/roadmap.md](docs/research/roadmap.md)。

## 代码边界
- 新代码只放在：`shinka/research/`、`skills/shinka-research/`、`examples/quant_research/`、`tests/research/`、`docs/research/`。
- 不改上游核心代码（`shinka/` 下除 `research/` 以外）。确需改动时单独提 PR，并在描述里写明原因，方便同步上游。
- 必要的接入点（`pyproject.toml` 的 CLI 入口、package-data）可以改，改动保持最小。
- 不从 AI-Scientist-v2 复制代码或提示词原文（许可证不兼容），只参考其方法。

## 设计原则
- 不维护向后兼容；废弃路径直接删。
- 满足需求的最简单实现；不加没有需求依据的抽象、配置项和间接层。
- 优先复用 Shinka 已有能力和现有依赖（pandas、numpy、scipy、pydantic），先查源码再决定自己写。
- 评估器归人：任何让 LLM 能改动评估逻辑、数据切分或看到样本外指标的设计都是 bug。

## 工作流程
- 每项工作对应一个 GitHub issue 和一个 PR，分支名 `research/<简短描述>`。
- 提交信息用 Conventional Commits（`feat(research): …`、`docs(research): …`）。
- 合并条件：本地门禁全绿、评审意见已处理。满足即可直接合并。

## 本地门禁（提 PR 前必须全绿）
```bash
uv run ruff check shinka/research tests/research
uv run ruff format --check shinka/research tests/research
uv run pytest -q tests/research
uv run pytest -q -m "not requires_secrets"   # 确认没有破坏上游测试
```

## 文档语言
- `docs/research/` 内部文档用中文；面向用户的 README / 示例说明用英文。
