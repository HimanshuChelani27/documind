export type Strategy = 'dense' | 'bm25' | 'hybrid' | 'hybrid_rerank';
export type Scope = 'library' | 'workspace';

export const STRATEGY_LABELS: Record<Strategy, string> = {
  dense: 'Dense',
  bm25: 'BM25',
  hybrid: 'Hybrid',
  hybrid_rerank: 'Hybrid + rerank',
};
export const STRATEGY_HINTS: Record<Strategy, string> = {
  dense: 'FAISS over bge-small embeddings: matches meaning, even with different words',
  bm25: 'Sparse keyword scoring: exact terms, codes and numbers',
  hybrid: 'Dense + BM25 merged with reciprocal-rank fusion',
  hybrid_rerank: 'Hybrid candidates re-scored by a cross-encoder (MiniLM)',
};
export const STRATEGIES = Object.keys(STRATEGY_LABELS) as Strategy[];

export interface DocumentInfo {
  id: string;
  collection: string;
  scope: Scope;
  filename: string;
  title: string;
  pages: number;
  chunks: number;
  size_bytes: number;
  status: 'processing' | 'ready' | 'failed';
  error: string | null;
  created_at: string;
}

export interface ChunkInfo {
  id: string;
  doc_id: string;
  ordinal: number;
  page: number;
  heading: string | null;
  char_start: number;
  text: string;
}

export interface Hit {
  chunk_id: string;
  doc_id: string;
  doc_title: string;
  filename: string;
  page: number;
  heading: string | null;
  text: string;
  score: number;
  rank: number;
}

export interface StageHit {
  chunk_id: string;
  page: number;
  filename: string;
  score: number;
}

export interface RetrievalEvent {
  type: 'retrieval';
  strategy: Strategy;
  hits: Hit[];
  stages: Partial<Record<'dense' | 'bm25' | 'fused' | 'reranked', StageHit[]>>;
  timings_ms: Record<string, number>;
}

export interface Source extends Omit<Hit, 'rank'> {
  n: number;
  cited: boolean;
}

export interface DoneEvent {
  type: 'done';
  mode: 'live' | 'extractive';
  provider: string;
  model: string | null;
  fallback_reason: string | null;
  usage: { input_tokens: number; output_tokens: number };
  timings_ms: Record<string, number | null>;
}

export type ChatEvent =
  | RetrievalEvent
  | { type: 'token'; text: string }
  | { type: 'citations'; answer: string; sources: Source[] }
  | DoneEvent;

export interface LlmProvider {
  provider: string;
  model: string;
  mode: 'live' | 'extractive';
}

export interface Dataset {
  id: string;
  name: string;
  collection: string;
  scope: Scope;
  owner: string | null;
  builtin: boolean;
  items: number;
}

export interface EvalItem {
  id: string;
  dataset_id: string;
  question: string;
  answer: string | null;
  doc: string;
  gold_pages: number[];
  type: string | null;
}

export type Metrics = Record<string, number>;

export interface EvalRun {
  id: string;
  dataset_id: string;
  dataset_name?: string;
  shared?: boolean;
  collection: string;
  strategies: Strategy[];
  k: number;
  items: number;
  with_answers: boolean;
  status: 'running' | 'done' | 'failed';
  mode: 'live' | 'extractive';
  judge: string | null;
  summary: { strategies?: Record<string, Metrics>; by_type?: Record<string, Record<string, Metrics>> };
  error: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface RetrievedRef {
  chunk_id: string;
  filename: string;
  page: number;
  score: number;
  relevant: boolean;
}

export interface EvalRunDetail extends EvalRun {
  results: {
    item_id: string;
    question: string;
    reference: string | null;
    doc: string | null;
    gold_pages: number[];
    type: string | null;
    strategies: Record<string, { metrics: Metrics; retrieved: RetrievedRef[]; answer: string | null }>;
  }[];
}

export type EvalEvent =
  | { type: 'progress'; done: number; total: number; item_id: string; strategy: Strategy; metrics: Metrics }
  | { type: 'done'; status: string; summary: EvalRun['summary']; error: string | null };

export type IngestEvent =
  | { type: 'progress'; stage: string; progress: number }
  | { type: 'done'; document: DocumentInfo | null; error?: string };
