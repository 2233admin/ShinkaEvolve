"""
ShinkaEvolve Circle Packing Demo - 最小化体验版
用 gpt-4.1-mini 跑 5 代进化，观察 LLM 如何自动改进圆填充算法
"""
import asyncio
from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# 评估器：本地运行 evaluate.py
job_conf = LocalJobConfig(
    eval_program_path="examples/circle_packing/evaluate.py"
)

# 数据库：单岛屿，小规模
db_conf = DatabaseConfig(
    db_path="demo_circle_db.sqlite",
    num_islands=1,
    archive_size=10,
    elite_selection_ratio=0.3,
    num_archive_inspirations=1,
    num_top_k_inspirations=1,
)

# 进化配置：5 代，单模型，便宜快速
evo_conf = EvolutionConfig(
    init_program_path="examples/circle_packing/initial.py",
    task_sys_msg=(
        "You are an expert mathematician specializing in circle packing problems. "
        "The best known result for the sum of radii when packing 26 circles in a "
        "unit square is 2.635. Be creative and try to find a new solution."
    ),
    patch_types=["diff", "full"],
    patch_type_probs=[0.5, 0.5],
    num_generations=5,
    max_proposal_jobs=1,
    max_patch_attempts=3,
    llm_models=["gpt-4.1-mini"],
    llm_dynamic_selection=None,      # 单模型不需要 UCB
    llm_kwargs={"temperatures": [0.7], "max_tokens": 8192},
    embedding_model=None,            # 跳过 embedding 省钱
    meta_rec_interval=None,          # 跳过 meta summarizer
    results_dir="demo_results",
    language="python",
)

runner = ShinkaEvolveRunner(
    evo_config=evo_conf,
    job_config=job_conf,
    db_config=db_conf,
    max_evaluation_jobs=1,
)

print("=" * 60)
print("ShinkaEvolve Circle Packing Demo")
print("Model: gpt-4.1-mini | Generations: 5 | Islands: 1")
print("=" * 60)

runner.run()
