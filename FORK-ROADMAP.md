# ShinkaEvolve Fork Roadmap

> Fork: `2233admin/ShinkaEvolve` ← upstream: `SakanaAI/ShinkaEvolve`
> 审计日期: 2026-03-26 | 硬件: RTX 5090 + 9800X3D + 32GB

## 总览矩阵

| # | 问题 | 优先级 | Phase | 状态 |
|---|------|--------|-------|------|
| 1 | 零 GPU 加速 | P0 | 2 | 待开始 |
| 2 | async_runner.py 3940 行上帝类 | P0 | 3 | 待开始 |
| 3 | defaults.py 幻觉模型列表 | P1 | 1 | ~~跳过~~ (已有pricing) |
| 4 | 无 context length 校验 | P1 | 1 | ✅ 完成 |
| 5 | 嵌入全走付费 API | P1 | 1+2 | ✅ 完成 (local/) |
| 6 | async/sync 混用 | P2 | 3 | 待开始 |
| 7 | SQLite 无显式事务 | P2 | 3 | 待开始 |
| 8 | 测试覆盖 ~70% | P2 | 3 | 待开始 |
| 9 | RichTeeConsole 全局锁 | P3 | 3 | 待开始 |

## 保留不动的（审计评分 8+/10）

- LLM 集成: 8 provider + `local/model@url` + UCB bandit — 不碰
- 评估系统: aggregate_metrics_fn / validate_fn / plotting_fn — 不碰
- Island 模型 + MAP-Elites 进化引擎 — 不碰
- Hydra 配置系统 — 不碰
- 错误处理: retry + exponential backoff — 不碰

---

## Phase 1 — 基础整顿

目标：低风险快速收益，修掉开箱即炸的问题。

### 1.1 Git remote 切换

- **任务**: origin 指向 fork，添加 upstream 指向 SakanaAI
- **涉及文件**: `.git/config`
- **改动要点**:
  ```bash
  git remote rename origin upstream
  git remote add origin https://github.com/2233admin/ShinkaEvolve.git
  git fetch --all
  ```
- **验收标准**: `git remote -v` 显示 origin=2233admin, upstream=SakanaAI
- **风险**: 无

### 1.2 修复幻觉模型列表

- **任务**: 替换 defaults.py 中不存在的模型名为实际可用模型
- **涉及文件**: `shinka/defaults.py`
- **改动要点**:
  - `gpt-5-mini` → `gpt-4.1-mini`
  - `gpt-5.4` → `gpt-4.1`
  - `gemini-3-flash-preview` → `gemini-2.5-flash`
  - `gemini-3.1-pro-preview` → `gemini-2.5-pro`
  - 添加 `local/qwen2.5:3b@http://localhost:11434/v1` 作为免费备选
- **验收标准**: 默认配置直接能跑，不报 model not found
- **风险**: 上游模型名可能是 Sakana 内部预览版，需确认 pricing.py 的 provider 映射

### 1.3 本地嵌入支持（Ollama）

- **任务**: embed client 支持 OpenAI-compatible 本地端点
- **涉及文件**: `shinka/embed/client.py`, `shinka/embed/embedding.py`
- **改动要点**:
  - 检测 `embedding_model` 格式：`local/model@base_url` 时走本地
  - 复用已有的 `local_openai.py` provider 模式
  - 添加 `SHINKA_EMBEDDING_BASE_URL` 环境变量覆盖
  - 默认 fallback: 先试本地 `http://localhost:11434/v1`，失败再走 API
- **验收标准**: `embedding_model="local/nomic-embed-text@http://localhost:11434/v1"` 能跑通 novelty check
- **风险**: Ollama embedding 维度可能与 OpenAI text-embedding-3-small (1536) 不同，需要在 novelty judge 中做维度适配或归一化

### 1.4 Context length 守卫

- **任务**: prompt 组装时计算 token 数，超限时截断 meta-recommendations
- **涉及文件**: `shinka/core/async_runner.py`, `shinka/prompts/`
- **改动要点**:
  - 添加 `shinka/utils/token_counter.py`:
    - 优先用 `tiktoken`（OpenAI 模型）
    - Anthropic 模型用 `anthropic.count_tokens()` 或估算（chars/3.5）
    - 其他模型用 chars/4 估算
  - 在 prompt 组装点（`_build_system_prompt` 或等效位置）插入守卫
  - 配置项: `max_prompt_tokens` (默认模型 context 的 60%)
  - 超限策略: 先截断 meta-recommendations → 再截断 inspirations → 最后截断 task description（报警）
- **验收标准**: 连续跑 100 代不因 token limit 报错; 日志中能看到截断警告
- **风险**: 不同模型 context window 差异大（4K~200K），需要从 model_resolver 拿到模型的 max_tokens

---

## Phase 2 — CUDA 加速

目标：RTX 5090 利用率从 0% 提升到有意义的水平。

### 2.1 本地 GPU 嵌入

- **任务**: sentence-transformers + CUDA 替代 API 嵌入
- **涉及文件**: `shinka/embed/client.py`, `shinka/embed/embedding.py`, 新增 `shinka/embed/local_gpu.py`
- **改动要点**:
  - 新增 `LocalGPUEmbeddingClient`:
    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
    ```
  - 配置项: `embedding_backend = "gpu" | "api" | "ollama"`
  - batch 化: 攒够 N 个再一次性 encode，减少 GPU kernel launch 开销
  - 模型选择: `all-MiniLM-L6-v2` (384维, 快) 或 `bge-large-zh-v1.5` (1024维, 中文好)
- **验收标准**:
  - 嵌入速度: 1000 段代码 < 5 秒 (GPU) vs 之前 API 的 30+ 秒
  - GPU 显存占用 < 2GB
  - novelty check 结果与 API 嵌入质量一致（相关性 > 0.9）
- **风险**: 嵌入维度不同需要重建已有数据库的嵌入列；sentence-transformers 依赖可能与现有 torch 版本冲突

### 2.2 faiss-gpu 相似度检索

- **任务**: 用 faiss-gpu 替代 NumPy cosine similarity
- **涉及文件**: `shinka/core/async_novelty_judge.py`, `shinka/database/dbase.py`, 新增 `shinka/embed/faiss_index.py`
- **改动要点**:
  - 新增 `FaissIndex` 封装:
    ```python
    import faiss
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, dim)  # Inner Product = cosine sim on normalized vectors
    index.add(normalized_embeddings)
    D, I = index.search(query, k=10)
    ```
  - 程序入库时自动 add 到 faiss index
  - novelty check 走 faiss 而非遍历比对
  - index 持久化: 保存/加载 faiss index 到 results_dir
- **验收标准**:
  - 10000 个程序的 novelty check < 10ms (GPU) vs 之前 ~500ms (CPU)
  - 内存占用增量 < 500MB
- **风险**: faiss-gpu 安装依赖 CUDA toolkit 版本匹配; Windows 上 faiss-gpu wheel 可能需要手动编译

### 2.3 评估函数 batch 化支持

- **任务**: 为用户评分函数提供 GPU batch 化工具集
- **涉及文件**: `shinka/core/wrap_eval.py`, 新增 `shinka/utils/gpu_batch.py`
- **改动要点**:
  - 提供 `@gpu_batch` decorator 让用户评分函数可选 GPU 加速
  - 内置 tensor 化工具: NumPy array → torch.Tensor (cuda) 的自动转换
  - 多 run 并行: 用 CUDA streams 并行执行多次评估
  - 保持向后兼容: 不用 decorator 的评分函数行为不变
- **验收标准**:
  - 量化选股评分函数（四层评分）GPU 版比 CPU 版快 5x+
  - 不影响现有非 GPU 用户
- **风险**: 用户评分函数多样性大，不是所有计算都适合 GPU; 需要文档说明适用场景

---

## Phase 3 — 架构治理

目标：长期可维护性，降低改动成本。

### 3.1 拆分 async_runner.py

- **任务**: 3940 行上帝类拆为 4 个模块
- **涉及文件**: `shinka/core/async_runner.py` → 拆为:
  - `shinka/core/orchestrator.py` — 主循环 + 生命周期管理
  - `shinka/core/proposal_engine.py` — 程序生成 + LLM 调用
  - `shinka/core/job_monitor.py` — 评估任务调度 + 结果收集
  - `shinka/core/state_manager.py` — 状态机 + 进度追踪 + 恢复
- **改动要点**:
  - 先画依赖图（哪些方法互相调用）
  - 提取公共状态到 `EvolutionState` dataclass
  - 通过 asyncio.Queue / Event 解耦模块间通信
  - 保持 `ShinkaEvolveRunner` 作为 facade，对外接口不变
- **验收标准**:
  - 每个文件 < 1000 行
  - 现有测试全部通过
  - `ShinkaEvolveRunner` 的公共 API 零变化
- **风险**: 这是最大的改动，需要逐步重构而非一次性重写; 异步状态共享容易引入竞态

### 3.2 统一 async

- **任务**: 将 PromptSampler、MetaSummarizer 等同步组件改为 async
- **涉及文件**: `shinka/core/sampler.py`, `shinka/core/async_summarizer.py`, `shinka/core/summarizer.py`
- **改动要点**:
  - `PromptSampler.sample()` → `async sample()`
  - `MetaSummarizer.summarize()` → `async summarize()`
  - 删除 `asyncio.to_thread()` 桥接调用
  - LLM 调用统一走 `query_async()`
- **验收标准**: 代码中零 `asyncio.to_thread()` 调用; 性能不退化
- **风险**: 低——这些组件内部逻辑简单，主要是 LLM API 调用

### 3.3 SQLite 事务管理

- **任务**: 添加显式事务和 checkpoint 策略
- **涉及文件**: `shinka/database/dbase.py`, `shinka/database/async_dbase.py`
- **改动要点**:
  - 写操作包裹在 `BEGIN IMMEDIATE ... COMMIT` 中
  - 添加 savepoint 支持: 批量写入可回滚
  - 定期 WAL checkpoint (`PRAGMA wal_checkpoint(TRUNCATE)`)
  - crash recovery: 启动时检查 WAL 完整性
- **验收标准**: kill -9 后重启不丢已 commit 的数据; WAL 文件不无限增长
- **风险**: 低——SQLite 本身支持良好，只是代码层没显式用

### 3.4 补充测试

- **任务**: 覆盖率从 ~70% 提升到 85%+
- **涉及文件**: `tests/` 全目录
- **改动要点**:
  - 添加 chaos testing: 随机 kill 评估进程, 模拟 API 超时/错误
  - 添加竞态测试: 多 island 并发写入 DB
  - 添加大规模测试: 1000+ 程序的 novelty check 性能回归
  - 添加集成测试: 端到端跑 5 代进化（用 mock LLM）
  - CI: GitHub Actions + pytest-cov + 覆盖率门槛
- **验收标准**: `pytest --cov=shinka --cov-fail-under=85` 通过
- **风险**: mock LLM 的响应质量影响测试有效性

### 3.5 RichTeeConsole 锁优化

- **任务**: 替换全局线程锁为更细粒度的方案
- **涉及文件**: `shinka/core/async_runner.py` (拆分后在 orchestrator.py)
- **改动要点**:
  - 方案 A: 用 `asyncio.Lock` 替代 `threading.Lock`
  - 方案 B: 日志走 `logging` + queue handler，不锁 console
  - 方案 C: Rich 的 `Console(force_terminal=True)` + 去掉手动 tee
- **验收标准**: 高并发 (8+ 并行评估) 时日志输出无明显卡顿
- **风险**: 极低

---

## 上游同步策略

### 定期 merge

```bash
# 每月一次（或按需）
git fetch upstream
git log upstream/main --oneline -20   # 看看有啥新的
git diff main..upstream/main --stat   # 改了哪些文件

# cherry-pick 有价值的（别全 merge，我们改了很多）
git cherry-pick <commit-hash>

# 或者对特定文件做 merge
git checkout upstream/main -- shinka/llm/providers/some_new_provider.py
```

### 同步原则

1. **新 provider 支持**: 直接拿（他们加了新 LLM provider 我们就 cherry-pick）
2. **bug fixes**: 逐个评估，我们改过的文件要手动 merge
3. **进化算法改进**: 仔细审查后决定是否采纳
4. **不同步的**: defaults.py、embed/、我们重构过的 core/

### 冲突预防

- Phase 3 拆分 async_runner.py 后，上游对该文件的改动需要手动映射到我们的新模块
- 维护 `UPSTREAM-MAPPING.md` 记录上游文件 → 我们文件的映射关系

---

## 依赖变更

Phase 2 新增:
```toml
[project.optional-dependencies]
gpu = [
    "torch>=2.5.0",
    "sentence-transformers>=3.0.0",
    "faiss-gpu-cu12>=1.9.0",
]
```

Phase 1 新增:
```toml
[project.optional-dependencies]
local = [
    "tiktoken>=0.8.0",  # token counting
]
```

安装: `pip install -e ".[gpu,local]"`
