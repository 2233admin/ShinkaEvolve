// Writes the design preview's fixture from a running `shinka_research ui` on the demo workspace that
// `shinka/research/webui/demo/build_demo_workspace.py` generates (deterministic, labelled 演示数据), so the
// preview draws real component output on demo data:
//   node src/preview/fixture.mjs http://127.0.0.1:8765
// Trimmed to stay small: the first 300 trials of d1-momentum-regimes and the programs of its h001/e01 run up
// to generation 10, without the long text fields the preview never shows.
import { writeFileSync } from 'node:fs'

const base = process.argv[2] ?? 'http://127.0.0.1:8765'
const RUN = 'workspace/d1-momentum-regimes/experiments/h001/e01/results/programs.sqlite'
const LONG = ['llm_result', 'thought', 'code_analysis_metrics', 'stdout_log', 'stderr_log', 'novelty_explanation', 'patch_description']

async function json(path) {
  const response = await fetch(base + path)
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`)
  return response.json()
}

const snap = await json('/api/snapshot?direction=d1-momentum-regimes')
snap.trial_points = snap.trial_points.slice(0, 300)
delete snap.library
const q = new URLSearchParams({ db_path: RUN })
const programs = (await json(`/get_programs_summary?${q}`))
  .filter((p) => (p.generation ?? 0) <= 10)
  .map(({ text_feedback, embedding_pca_2d, embedding_pca_3d, embedding_cluster_id, ...p }) => {
    const metadata = { ...p.metadata }
    for (const key of LONG) delete metadata[key]
    return { ...p, metadata }
  })
const stats = await json(`/get_database_stats?${q}`)
const prompts = stats.has_prompt_evo ? await json(`/get_system_prompts?${q}`) : []
writeFileSync(new URL('./fixture.json', import.meta.url), JSON.stringify({ source: 'demo workspace (演示数据)', run: RUN, snap, programs, stats, prompts }))
console.log(`trials ${snap.trial_points.length}, programs ${programs.length}, prompts ${prompts.length}`)
