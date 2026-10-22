// The demo generator (webui/demo) prefixes every title with this label. The shell shows the label
// once, as a DEMO mark beside the workspace, and the titles without it (approved high-fidelity
// package, hifi/extract.mjs › clean).
export const DEMO_TAG = /\[?DEMO\s*·\s*演示数据，非真实研究\]?\s*[—–:：·.。,，-]?\s*/

const EVERY_TAG = new RegExp(DEMO_TAG.source, 'g')

/** A title without the demo label; other titles are returned as written. */
export function untagged(text: string): string {
  return text.replace(EVERY_TAG, '').trim()
}
