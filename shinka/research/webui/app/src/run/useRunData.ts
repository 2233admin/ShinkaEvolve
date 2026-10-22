import { useEffect, useState } from 'react'
import type { RunStats } from '../runs/data'
import type { RunEntry } from '../types'
import type { MetaFile, Prompt, RunProgram } from './data'

/** One run's analysis data; prompts and meta files are optional parts that fail on their own. */
export type RunData = {
  programs: RunProgram[]
  stats: RunStats
  prompts: Prompt[]
  /** Why the prompt versions could not be read (L02-E1); null when they were, or the run has none. */
  promptsError: string | null
  metaFiles: MetaFile[]
  metaError: string | null
}

const message = (error: unknown): string => (error instanceof Error ? error.message : String(error))

async function readJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { cache: 'no-store', signal })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return (await res.json()) as T
}

export const runQuery = (run: string, extra: Record<string, string> = {}) => new URLSearchParams({ db_path: run, ...extra }).toString()

type Read = { key: string; data: RunData | null; error: string | null }

/**
 * A run's programs, statistics, prompt versions and meta-analysis files, read again whenever the shell
 * lists the runs again, so the page follows the automatic refresh and its pause. A failed read keeps
 * the data shown before and says why.
 */
export function useRunData(run: string, listing: RunEntry[] | null): { data: RunData | null; error: string | null } {
  const [read, setRead] = useState<Read>({ key: run, data: null, error: null })
  useEffect(() => {
    if (listing === null) return
    const controller = new AbortController()
    const { signal } = controller
    const q = runQuery(run)
    const optional = async <T>(url: string): Promise<[T, string | null]> => {
      try {
        return [await readJson<T>(url, signal), null]
      } catch (error) {
        if (signal.aborted) throw error
        return [[] as T, message(error)]
      }
    }
    ;(async () => {
      const [programs, stats] = await Promise.all([readJson<RunProgram[]>(`/get_programs_summary?${q}`, signal), readJson<RunStats>(`/get_database_stats?${q}`, signal)])
      const [[prompts, promptsError], [metaFiles, metaError]] = await Promise.all([
        stats.has_prompt_evo ? optional<Prompt[]>(`/get_system_prompts?${q}`) : Promise.resolve<[Prompt[], null]>([[], null]),
        optional<MetaFile[]>(`/get_meta_files?${q}`),
      ])
      return { programs, stats, prompts, promptsError, metaFiles, metaError }
    })().then(
      (data) => setRead({ key: run, data, error: null }),
      (error: unknown) => {
        if (signal.aborted) return
        setRead((previous) => ({ key: run, data: previous.key === run ? previous.data : null, error: message(error) }))
      },
    )
    return () => controller.abort()
  }, [run, listing])
  return read.key === run ? read : { data: null, error: null }
}

// Program details, meta files and plot lists never change once written: one read per URL serves every view.
const cache = new Map<string, unknown>()
const CACHE_LIMIT = 300

function remember(url: string, value: unknown) {
  cache.delete(url)
  cache.set(url, value)
  if (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value as string)
}

export type Fetched<T> = { data: T | null; error: string | null; loading: boolean }

/** One read-only JSON resource, cached; a failure is read again when `again` changes (each shell refresh). */
export function useJson<T>(url: string | null, again: unknown): Fetched<T> {
  const [state, setState] = useState<{ url: string | null; data: T | null; error: string | null }>({ url: null, data: null, error: null })
  useEffect(() => {
    if (url === null || cache.has(url)) return
    const controller = new AbortController()
    readJson<T>(url, controller.signal).then(
      (data) => {
        remember(url, data)
        setState({ url, data, error: null })
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setState({ url, data: null, error: message(error) })
      },
    )
    return () => controller.abort()
  }, [url, again])
  if (url === null) return { data: null, error: null, loading: false }
  if (cache.has(url)) return { data: cache.get(url) as T, error: null, loading: false }
  return state.url === url ? { data: state.data, error: state.error, loading: state.error === null } : { data: null, error: null, loading: true }
}

/** Reads a cached resource outside a component (the meta pane searches the files one by one). */
export async function fetchJson<T>(url: string): Promise<T> {
  if (cache.has(url)) return cache.get(url) as T
  const data = await readJson<T>(url)
  remember(url, data)
  return data
}

/**
 * The text a reader sees in an element, for a plain-text download of a note (OD-12): `innerText`
 * leaves out the numbers and bullets the browser draws for list items, so they are added for the read.
 */
export function shownText(el: HTMLElement): string {
  const marks = [...el.querySelectorAll('li')].map((li) => {
    const list = li.parentElement
    const mark = document.createElement('span')
    mark.textContent = list instanceof HTMLOListElement ? `${list.start + [...list.children].indexOf(li)}. ` : '- '
    li.prepend(mark)
    return mark
  })
  try {
    return `${el.innerText.trim()}\n`
  } finally {
    for (const mark of marks) mark.remove()
  }
}

/** Saves text as a file the way the restored page did: a Blob behind a temporary link. */
export function saveText(text: string, filename: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }))
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}
