export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return '';
  const seconds = Math.round((Date.now() - parseDate(iso).getTime()) / 1000);
  if (seconds < 45) return 'just now';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/** The API returns naive UTC timestamps from SQLite; treat them as UTC. */
export function parseDate(iso: string): Date {
  return new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`);
}

export function formatMs(ms: number | null | undefined): string {
  if (ms == null) return '–';
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

export function formatBytes(n: number): string {
  return n < 1024 * 1024 ? `${Math.round(n / 1024)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function pct(v: number | undefined): string {
  return v == null ? '–' : `${(v * 100).toFixed(1)}%`;
}

export function shortName(filename: string): string {
  return filename.replace(/\.pdf$/i, '').replace(/[_-]+/g, ' ');
}

/** Answer text split into plain text and citation markers, so the template renders both safely. */
export type AnswerPart = { text: string } | { cite: number };

export function answerParts(text: string): AnswerPart[] {
  const parts: AnswerPart[] = [];
  const re = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let last = 0;
  for (let m: RegExpExecArray | null; (m = re.exec(text)); ) {
    if (m.index > last) parts.push({ text: text.slice(last, m.index) });
    for (const n of m[1].split(',')) parts.push({ cite: Number(n.trim()) });
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push({ text: text.slice(last) });
  return parts;
}

export function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}

const STOP = new Set(
  'a an and are as at be by can do does for from has have how i if in is it its of on or the this to was what when where which who why will with my me our we you your there their they much many'.split(
    ' ',
  ),
);

export function queryTerms(question: string): string[] {
  return Array.from(
    new Set(
      question
        .toLowerCase()
        .split(/[^a-z0-9.-]+/)
        .map((t) => t.replace(/^[.-]+|[.-]+$/g, ''))
        .filter((t) => t.length > 1 && !STOP.has(t)),
    ),
  );
}

/** Escapes the text, then wraps query terms in <mark>. Safe for [innerHTML]. */
export function highlight(text: string, terms: string[]): string {
  const html = escapeHtml(text);
  if (!terms.length) return html;
  const pattern = terms
    .map((t) => escapeHtml(t).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .sort((a, b) => b.length - a.length)
    .join('|');
  return html.replace(new RegExp(`\\b(${pattern})\\b`, 'gi'), '<mark>$1</mark>');
}

/** Joins lines a PDF wrapped mid-sentence, for display (same rule as the backend's extractive reflow). */
export function reflow(text: string): string {
  const lines = text
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean);
  if (!lines.length) return '';
  const width = Math.max(...lines.map((l) => l.length));
  const out = [lines[0]];
  for (let i = 1; i < lines.length; i++) {
    const prev = lines[i - 1];
    const open = width >= 60 && !/[.!?:;]$/.test(prev);
    const lower = /^[a-z]/.test(lines[i]);
    if (open && (prev.length >= 0.85 * width || (prev.length >= 0.7 * width && lower))) out[out.length - 1] += ' ' + lines[i];
    else out.push(lines[i]);
  }
  return out.join('\n');
}
