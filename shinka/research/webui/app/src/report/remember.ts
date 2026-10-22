import { useState } from 'react'

const memory = new Map<string, unknown>()

/**
 * State that outlives the page for the rest of the visit, so Back from an experiment's analysis finds
 * the report as it was left (spec › 往返保留上下文). A reload starts afresh.
 */
export function useRemembered<T>(key: string, initial: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => (memory.has(key) ? (memory.get(key) as T) : initial))
  const remember = (next: T) => {
    memory.set(key, next)
    setValue(next)
  }
  return [value, remember]
}
