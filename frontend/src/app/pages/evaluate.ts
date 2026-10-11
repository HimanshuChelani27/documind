import { httpResource } from '@angular/common/http';
import { ChangeDetectionStrategy, Component, OnDestroy, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ApiService, api, errorMessage } from '../core/api';
import { formatMs, pct, shortName, timeAgo } from '../core/format';
import { LibraryStore } from '../core/library';
import {
  Dataset,
  EvalEvent,
  EvalItem,
  EvalRun,
  EvalRunDetail,
  Metrics,
  STRATEGIES,
  STRATEGY_HINTS,
  STRATEGY_LABELS,
  Scope,
  Strategy,
} from '../core/models';
import { Icon } from '../layout/icon';

/** Fixed per strategy (color follows the entity). Validated for adjacent bars on the dark surface. */
const SERIES_COLORS: Record<Strategy, string> = {
  dense: '#3987e5',
  bm25: '#d95926',
  hybrid: '#199e70',
  hybrid_rerank: '#c98500',
};

const RETRIEVAL_METRICS = [
  { key: 'hit', label: 'Hit@k', hint: 'Share of questions with at least one gold page in the top k' },
  { key: 'recall', label: 'Recall@k', hint: 'Share of each question’s gold pages found in the top k' },
  { key: 'mrr', label: 'MRR', hint: 'Mean reciprocal rank of the first gold-page chunk' },
  { key: 'ndcg', label: 'nDCG@k', hint: 'Rank-weighted gain: gold pages near the top score more' },
  { key: 'context_precision', label: 'Ctx precision', hint: 'Share of the top-k chunks that come from a gold page' },
];
const JUDGE_METRICS = [
  { key: 'faithfulness', label: 'Faithfulness', hint: 'LLM judge: share of answer claims supported by the retrieved text' },
  { key: 'answer_relevance', label: 'Answer relevance', hint: 'LLM judge: how directly the answer addresses the question' },
];
const PROXY_METRICS = [
  { key: 'supported_share', label: 'Supported (proxy)', hint: 'Keyless proxy: answer sentences closely matched (cosine ≥ 0.8) by a retrieved sentence' },
  { key: 'qa_similarity', label: 'Q–A similarity (proxy)', hint: 'Keyless proxy: embedding similarity between question and answer' },
];

const CHART = { w: 640, h: 250, left: 36, right: 8, top: 12, bottom: 28 };

@Component({
  selector: 'app-evaluate',
  imports: [FormsModule, Icon],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <h1 class="page-title">Evaluate retrieval</h1>
    <p class="page-sub max-w-3xl">
      Run a question set through each strategy and score it against gold pages. Retrieval metrics are always real.
      Answer metrics use an LLM judge when a key is configured, otherwise clearly labelled embedding proxies.
    </p>

    <!-- Run form -->
    <section class="card mt-6 p-4 sm:p-5">
      <div class="grid gap-4 md:grid-cols-[minmax(0,1.2fr)_minmax(0,2fr)_auto] md:items-end">
        <div>
          <label class="label" for="ds">Question set</label>
          <select id="ds" class="input" [ngModel]="datasetId()" (ngModelChange)="pickedDataset.set($event)">
            @for (d of datasets.value(); track d.id) {
              <option [value]="d.id">{{ d.name }} ({{ d.items }})</option>
            }
          </select>
        </div>
        <div>
          <p class="label">Strategies</p>
          <div class="flex flex-wrap gap-1.5">
            @for (s of strategies; track s) {
              <button
                class="inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs transition"
                [class]="picked().has(s) ? 'border-accent/60 bg-accent/10 text-text-primary' : 'border-border-soft text-text-muted hover:text-text-primary'"
                [attr.aria-pressed]="picked().has(s)"
                [title]="hints[s]"
                (click)="toggleStrategy(s)"
              >
                <span class="h-2 w-2 rounded-sm" [style.background]="colors[s]"></span>{{ labels[s] }}
              </button>
            }
          </div>
        </div>
        <div class="flex items-end gap-2">
          <div>
            <label class="label" for="k">k</label>
            <select id="k" class="input !w-20" [ngModel]="k()" (ngModelChange)="k.set(+$event)">
              @for (n of [1, 3, 5, 10]; track n) {
                <option [value]="n">{{ n }}</option>
              }
            </select>
          </div>
          <button class="btn-primary h-[38px]" [disabled]="running() || !picked().size || !datasetId()" (click)="run()">
            <app-icon name="play" /> Run
          </button>
        </div>
      </div>
      <label class="mt-4 flex items-start gap-2 text-xs text-text-secondary">
        <input type="checkbox" class="mt-0.5 accent-[var(--color-accent)]" [ngModel]="withAnswers()" (ngModelChange)="withAnswers.set($event)" />
        <span>
          Also generate and grade answers (first {{ answerLimit }} questions).
          <span class="text-text-muted">With an LLM key this uses an LLM judge; without one, keyless proxies.</span>
        </span>
      </label>
      @if (progress(); as p) {
        <div class="mt-4" aria-live="polite">
          <div class="h-1.5 overflow-hidden rounded-full bg-surface-2">
            <div class="h-full rounded-full bg-accent transition-all" [style.width.%]="(p.done / p.total) * 100"></div>
          </div>
          <p class="mt-1.5 font-mono text-[11px] text-text-muted">{{ p.done }} / {{ p.total }} question × strategy runs</p>
        </div>
      }
      @if (error()) {
        <p class="mt-3 text-sm text-deny">{{ error() }}</p>
      }
    </section>

    <!-- Run picker -->
    <div class="mt-8 flex flex-wrap items-center justify-between gap-3">
      <h2 class="text-lg font-semibold">Results</h2>
      @if (runs.value().length) {
        <select class="input !w-auto max-w-full" [ngModel]="activeRunId()" (ngModelChange)="selectedRun.set($event)" aria-label="Choose a run">
          @for (r of runs.value(); track r.id) {
            <option [value]="r.id">
              {{ r.shared ? 'Baseline' : 'Your run' }} · {{ r.dataset_name }} · k={{ r.k }} · {{ ago(r.created_at) }}{{ r.status !== 'done' ? ' · ' + r.status : '' }}
            </option>
          }
        </select>
      }
    </div>

    @if (detail.value(); as d) {
      @if (d.status === 'failed') {
        <p class="mt-4 text-sm text-deny">This run failed: {{ d.error }}</p>
      } @else if (d.status === 'running') {
        <p class="mt-4 text-sm text-text-muted">Still running…</p>
      } @else {
        <p class="mt-1 text-xs text-text-muted">
          {{ d.items }} questions · k = {{ d.k }} ·
          {{ d.with_answers ? (d.mode === 'live' ? 'answers graded by ' + d.judge : 'answers scored with keyless proxies') : 'retrieval only' }}
        </p>

        <!-- Summary table -->
        <section class="card mt-4 overflow-hidden">
          <div class="overflow-x-auto">
            <table class="w-full min-w-[720px]">
              <thead>
                <tr>
                  <th class="table-head">Strategy</th>
                  @for (m of columns(); track m.key) {
                    <th class="table-head text-right" [title]="m.hint">{{ m.label }}</th>
                  }
                  <th class="table-head text-right" title="Mean retrieval time per question (embedding + search + fusion + rerank)">Latency</th>
                </tr>
              </thead>
              <tbody>
                @for (s of d.strategies; track s) {
                  <tr>
                    <td class="table-cell whitespace-nowrap text-text-primary">
                      <span class="mr-2 inline-block h-2.5 w-2.5 rounded-sm align-middle" [style.background]="colors[s]"></span>{{ labels[s] }}
                    </td>
                    @for (m of columns(); track m.key) {
                      <td class="table-cell text-right tabular-nums" [class]="isBest(d, s, m.key) ? 'font-semibold text-text-primary' : ''">
                        {{ fmt(metric(d, s, m.key)) }}
                        @if (isBest(d, s, m.key)) {
                          <span class="sr-only">(best)</span><span aria-hidden="true" class="ml-1 text-accent">▲</span>
                        }
                      </td>
                    }
                    <td class="table-cell text-right tabular-nums text-text-muted">{{ ms(metric(d, s, 'latency_ms')) }}</td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        </section>

        <!-- Chart -->
        <section class="card mt-4 p-4 sm:p-5">
          <div class="flex flex-wrap items-center justify-between gap-3">
            <h3 class="text-sm font-medium">Retrieval metrics by strategy</h3>
            <ul class="flex flex-wrap gap-3 text-xs text-text-secondary" aria-label="Legend">
              @for (s of d.strategies; track s) {
                <li class="flex items-center gap-1.5"><span class="h-2.5 w-2.5 rounded-sm" [style.background]="colors[s]"></span>{{ labels[s] }}</li>
              }
            </ul>
          </div>
          <div class="-mx-4 mt-3 overflow-x-auto px-4 sm:mx-0 sm:px-0">
          <div class="relative min-w-[560px]">
            <svg [attr.viewBox]="'0 0 ' + chart.w + ' ' + chart.h" class="w-full" role="img" [attr.aria-label]="chartLabel()">
              @for (t of ticks; track t) {
                <line [attr.x1]="chart.left" [attr.x2]="chart.w - chart.right" [attr.y1]="y(t)" [attr.y2]="y(t)" stroke="#29292f" stroke-width="1" [attr.stroke-dasharray]="t === 0 ? null : '2 4'" />
                <text [attr.x]="chart.left - 6" [attr.y]="y(t) + 3" text-anchor="end" fill="#85858f" font-size="10" font-family="Geist Mono, monospace">{{ t * 100 }}%</text>
              }
              @for (g of bars(); track g.metric) {
                <text [attr.x]="g.center" [attr.y]="chart.h - 8" text-anchor="middle" fill="#b9b9c1" font-size="11">{{ g.label }}</text>
                @for (b of g.bars; track b.strategy) {
                  <path
                    [attr.d]="b.path"
                    [attr.fill]="colors[b.strategy]"
                    [attr.opacity]="hover() && hover()!.key !== g.metric + b.strategy ? 0.45 : 1"
                  />
                  <rect
                    [attr.x]="b.x - 1"
                    [attr.y]="chart.top"
                    [attr.width]="b.w + 2"
                    [attr.height]="chart.h - chart.top - chart.bottom"
                    fill="transparent"
                    (mouseenter)="hover.set({ key: g.metric + b.strategy, x: b.x + b.w / 2, y: b.y, text: labels[b.strategy] + ' · ' + g.label + ': ' + fmt(b.value) })"
                    (mouseleave)="hover.set(null)"
                  />
                  @if (b.best) {
                    <text [attr.x]="b.x + b.w / 2" [attr.y]="b.y - 4" text-anchor="middle" fill="#f2f1ee" font-size="9" font-family="Geist Mono, monospace">{{ short(b.value) }}</text>
                  }
                }
              }
            </svg>
            @if (hover(); as h) {
              <div
                class="pointer-events-none absolute -translate-x-1/2 -translate-y-full whitespace-nowrap rounded-lg border border-border bg-bg-raised px-2.5 py-1.5 text-xs text-text-primary shadow-lg"
                [style.left.%]="(h.x / chart.w) * 100"
                [style.top.%]="(h.y / chart.h) * 100"
              >
                {{ h.text }}
              </div>
            }
          </div>
          </div>
          <p class="mt-2 text-[11px] text-text-muted">The value of the best strategy is labelled on each metric; hover any bar for its value. The table above has every number.</p>
        </section>

        <!-- By question type -->
        @if (types().length > 1) {
          <section class="card mt-4 overflow-hidden">
            <div class="border-b border-border-soft px-4 py-3">
              <h3 class="text-sm font-medium">Recall@{{ d.k }} and MRR by question type</h3>
              <p class="mt-0.5 text-xs text-text-muted">
                Keyword questions quote exact terms (codes, figures, names); paraphrased ones use different words than the document.
              </p>
            </div>
            <div class="overflow-x-auto">
              <table class="w-full min-w-[560px]">
                <thead>
                  <tr>
                    <th class="table-head">Strategy</th>
                    @for (t of types(); track t) {
                      <th class="table-head text-right">{{ t }} recall</th>
                      <th class="table-head text-right">{{ t }} MRR</th>
                    }
                  </tr>
                </thead>
                <tbody>
                  @for (s of d.strategies; track s) {
                    <tr>
                      <td class="table-cell whitespace-nowrap text-text-primary">
                        <span class="mr-2 inline-block h-2.5 w-2.5 rounded-sm align-middle" [style.background]="colors[s]"></span>{{ labels[s] }}
                      </td>
                      @for (t of types(); track t) {
                        <td class="table-cell text-right tabular-nums">{{ fmt(d.summary.by_type?.[s]?.[t]?.['recall']) }}</td>
                        <td class="table-cell text-right tabular-nums">{{ fmt(d.summary.by_type?.[s]?.[t]?.['mrr']) }}</td>
                      }
                    </tr>
                  }
                </tbody>
              </table>
            </div>
          </section>
        }

        <!-- Per question -->
        <section class="card mt-4 overflow-hidden">
          <div class="flex flex-wrap items-center justify-between gap-2 border-b border-border-soft px-4 py-3">
            <h3 class="text-sm font-medium">Per question</h3>
            <label class="flex items-center gap-2 text-xs text-text-secondary">
              <input type="checkbox" class="accent-[var(--color-accent)]" [ngModel]="onlyDisagree()" (ngModelChange)="onlyDisagree.set($event)" />
              Only where strategies disagree
            </label>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full min-w-[720px]">
              <thead>
                <tr>
                  <th class="table-head">Question</th>
                  @for (s of d.strategies; track s) {
                    <th class="table-head text-center" [title]="'Rank of the first gold-page chunk for ' + labels[s]">{{ labels[s] }}</th>
                  }
                </tr>
              </thead>
              <tbody>
                @for (row of rows(); track row.item_id) {
                  <tr class="cursor-pointer transition hover:bg-surface-2" (click)="toggleRow(row.item_id)" [attr.aria-expanded]="openRow() === row.item_id">
                    <td class="table-cell">
                      <span class="text-text-primary">{{ row.question }}</span>
                      <span class="mt-0.5 block font-mono text-[11px] text-text-muted">
                        {{ row.type ?? '' }} · {{ docName(row.doc) }} p.{{ row.gold_pages.join(', ') }}
                      </span>
                    </td>
                    @for (s of d.strategies; track s) {
                      <td class="table-cell text-center font-mono text-xs">
                        @if (firstRank(row.strategies[s]?.retrieved); as r) {
                          <span class="rounded-full bg-ok/10 px-2 py-0.5 text-ok" [title]="'First gold-page chunk at rank ' + r">#{{ r }}</span>
                        } @else {
                          <span class="rounded-full bg-deny/10 px-2 py-0.5 text-deny" title="No gold page in the top k">miss</span>
                        }
                      </td>
                    }
                  </tr>
                  @if (openRow() === row.item_id) {
                    <tr>
                      <td [attr.colspan]="d.strategies.length + 1" class="border-b border-border-soft bg-bg-raised px-4 py-4">
                        @if (row.reference) {
                          <p class="mb-3 text-xs"><span class="text-text-muted">Reference answer:</span> <span class="text-text-secondary">{{ row.reference }}</span></p>
                        }
                        <div class="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                          @for (s of d.strategies; track s) {
                            <div class="rounded-xl border border-border-soft p-3">
                              <p class="flex items-center gap-1.5 text-xs font-medium text-text-primary">
                                <span class="h-2 w-2 rounded-sm" [style.background]="colors[s]"></span>{{ labels[s] }}
                              </p>
                              <ol class="mt-2 space-y-1">
                                @for (h of row.strategies[s]?.retrieved ?? []; track $index; let i = $index) {
                                  <li class="flex items-center gap-2 font-mono text-[11px]" [class]="h.relevant ? 'text-ok' : 'text-text-muted'">
                                    <span class="w-4 text-right">{{ i + 1 }}</span>
                                    <span class="min-w-0 flex-1 truncate">{{ docName(h.filename) }} · p{{ h.page }}</span>
                                    @if (h.relevant) {
                                      <app-icon name="check" cls="h-3 w-3" />
                                    }
                                  </li>
                                }
                              </ol>
                              @if (row.strategies[s]?.answer; as a) {
                                <p class="mt-2 border-t border-border-soft pt-2 text-xs text-text-secondary">{{ a }}</p>
                              }
                            </div>
                          }
                        </div>
                      </td>
                    </tr>
                  }
                } @empty {
                  <tr><td [attr.colspan]="d.strategies.length + 1" class="table-cell py-8 text-center text-text-muted">Every strategy agrees on every question.</td></tr>
                }
              </tbody>
            </table>
          </div>
        </section>
      }
    } @else if (runs.isLoading() || detail.isLoading()) {
      <p class="mt-4 text-sm text-text-muted">Loading…</p>
    } @else {
      <p class="mt-4 text-sm text-text-muted">No runs yet. The baseline run appears a few seconds after the server starts.</p>
    }

    <!-- Custom question sets -->
    <section class="card mt-10 p-4 sm:p-5">
      <h2 class="text-lg font-semibold">Your question sets</h2>
      <p class="mt-1 text-sm text-text-muted">
        Write questions about the sample library or your uploads, mark the page(s) that answer each one, then run them
        above. Private to this browser.
      </p>
      <div class="mt-4 flex flex-wrap items-end gap-2">
        <div class="min-w-48 flex-1">
          <label class="label" for="dsname">New set</label>
          <input id="dsname" class="input" placeholder="e.g. Contract questions" maxlength="80" [ngModel]="newName()" (ngModelChange)="newName.set($event)" />
        </div>
        <select class="input !w-auto" [ngModel]="newScope()" (ngModelChange)="newScope.set($event)" aria-label="Documents">
          <option value="library">about the sample library</option>
          <option value="workspace">about my uploads</option>
        </select>
        <button class="btn-ghost" [disabled]="!newName().trim()" (click)="createDataset()"><app-icon name="plus" /> Create</button>
      </div>

      @for (ds of mine(); track ds.id) {
        <div class="mt-5 rounded-xl border border-border-soft p-4">
          <div class="flex items-center justify-between gap-2">
            <p class="font-medium text-text-primary">{{ ds.name }} <span class="ml-1 text-xs font-normal text-text-muted">{{ ds.items }} questions · {{ ds.scope === 'library' ? 'sample library' : 'your uploads' }}</span></p>
            <div class="flex gap-1">
              <button class="btn-ghost !px-3 !py-1 !text-xs" (click)="editing.set(editing() === ds.id ? null : ds.id)">{{ editing() === ds.id ? 'Close' : 'Edit' }}</button>
              <button class="grid h-8 w-8 place-items-center rounded-full text-text-muted hover:bg-surface hover:text-deny" (click)="deleteDataset(ds)" [attr.aria-label]="'Delete ' + ds.name"><app-icon name="trash" /></button>
            </div>
          </div>
          @if (editing() === ds.id) {
            <ul class="mt-3 space-y-1.5">
              @for (it of items.value(); track it.id) {
                <li class="flex items-start gap-2 text-sm">
                  <span class="min-w-0 flex-1">{{ it.question }} <span class="font-mono text-[11px] text-text-muted">{{ docName(it.doc) }} p.{{ it.gold_pages.join(', ') }}</span></span>
                  <button class="text-text-muted hover:text-deny" (click)="deleteItem(ds, it)" aria-label="Delete question"><app-icon name="x" cls="h-3.5 w-3.5" /></button>
                </li>
              }
            </ul>
            <div class="mt-3 grid gap-2 sm:grid-cols-[minmax(0,2fr)_minmax(0,1.2fr)_6rem_auto]">
              <input class="input" placeholder="Question" maxlength="300" [ngModel]="q()" (ngModelChange)="q.set($event)" aria-label="Question" />
              <select class="input" [ngModel]="qDoc()" (ngModelChange)="qDoc.set($event)" aria-label="Document">
                <option value="">Document…</option>
                @for (doc of docsFor(ds.scope); track doc.id) {
                  <option [value]="doc.id">{{ doc.title }}</option>
                }
              </select>
              <input class="input" placeholder="Pages: 3, 4" [ngModel]="qPages()" (ngModelChange)="qPages.set($event)" aria-label="Gold pages" />
              <button class="btn-primary" [disabled]="!q().trim() || !qDoc() || !parsePages(qPages()).length" (click)="addItem(ds)">Add</button>
            </div>
          }
        </div>
      }
    </section>
  `,
})
export class Evaluate implements OnDestroy {
  private readonly api = inject(ApiService);
  private readonly store = inject(LibraryStore);

  protected readonly strategies = STRATEGIES;
  protected readonly labels = STRATEGY_LABELS;
  protected readonly hints = STRATEGY_HINTS;
  protected readonly colors = SERIES_COLORS;
  protected readonly chart = CHART;
  protected readonly ticks = [0, 0.25, 0.5, 0.75, 1];
  protected readonly answerLimit = 12;
  protected readonly ms = formatMs;
  protected readonly ago = timeAgo;

  protected readonly datasets = httpResource<Dataset[]>(() => api('/eval/datasets'), { defaultValue: [] });
  protected readonly runs = httpResource<EvalRun[]>(() => api('/eval/runs'), { defaultValue: [] });
  protected readonly selectedRun = signal<string | null>(null);
  protected readonly activeRunId = computed(
    () => this.selectedRun() ?? this.runs.value().find((r) => r.status === 'done')?.id ?? this.runs.value()[0]?.id,
  );
  protected readonly detail = httpResource<EvalRunDetail>(() => {
    const id = this.activeRunId();
    return id ? api(`/eval/runs/${id}`) : undefined;
  });

  protected readonly pickedDataset = signal<string | null>(null);
  protected readonly datasetId = computed(() => this.pickedDataset() ?? this.datasets.value()[0]?.id ?? '');
  protected readonly picked = signal(new Set<Strategy>(STRATEGIES));
  protected readonly k = signal(5);
  protected readonly withAnswers = signal(false);
  protected readonly running = signal(false);
  protected readonly progress = signal<{ done: number; total: number } | null>(null);
  protected readonly error = signal<string | null>(null);
  protected readonly hover = signal<{ key: string; x: number; y: number; text: string } | null>(null);
  protected readonly openRow = signal<string | null>(null);
  protected readonly onlyDisagree = signal(false);

  protected readonly newName = signal('');
  protected readonly newScope = signal<Scope>('library');
  protected readonly editing = signal<string | null>(null);
  protected readonly q = signal('');
  protected readonly qDoc = signal('');
  protected readonly qPages = signal('');
  protected readonly items = httpResource<EvalItem[]>(
    () => {
      const id = this.editing();
      return id ? api(`/eval/datasets/${id}/items`) : undefined;
    },
    { defaultValue: [] },
  );
  protected readonly mine = computed(() => this.datasets.value().filter((d) => !d.builtin));

  private stopStream: (() => void) | null = null;

  protected readonly columns = computed(() => {
    const d = this.detail.value();
    const first = d ? Object.values(d.summary.strategies ?? {})[0] ?? {} : {};
    return [
      ...RETRIEVAL_METRICS,
      ...JUDGE_METRICS.filter((m) => m.key in first),
      ...PROXY_METRICS.filter((m) => m.key in first),
    ];
  });

  protected readonly types = computed(() => {
    const d = this.detail.value();
    const first = d ? Object.values(d.summary.by_type ?? {})[0] : undefined;
    return first ? Object.keys(first).sort() : [];
  });

  protected readonly bars = computed(() => {
    const d = this.detail.value();
    if (!d?.summary.strategies) return [];
    const strategies = d.strategies;
    const plotW = CHART.w - CHART.left - CHART.right;
    const groupW = plotW / RETRIEVAL_METRICS.length;
    const gap = 2;
    const barW = Math.min(22, (groupW * 0.72 - gap * (strategies.length - 1)) / strategies.length);
    return RETRIEVAL_METRICS.map((m, gi) => {
      const values = strategies.map((s) => this.metric(d, s, m.key) ?? 0);
      const best = Math.max(...values);
      const total = barW * strategies.length + gap * (strategies.length - 1);
      const start = CHART.left + gi * groupW + (groupW - total) / 2;
      return {
        metric: m.key,
        label: m.label,
        center: CHART.left + gi * groupW + groupW / 2,
        bars: strategies.map((s, i) => {
          const value = values[i];
          const x = start + i * (barW + gap);
          const y = this.y(value);
          return { strategy: s, value, x, y, w: barW, path: barPath(x, y, barW, this.y(0)), best: value === best };
        }),
      };
    });
  });

  protected readonly rows = computed(() => {
    const d = this.detail.value();
    if (!d) return [];
    if (!this.onlyDisagree()) return d.results;
    return d.results.filter((r) => {
      const ranks = d.strategies.map((s) => this.firstRank(r.strategies[s]?.retrieved));
      return new Set(ranks).size > 1;
    });
  });

  protected readonly chartLabel = computed(() => {
    const d = this.detail.value();
    if (!d) return '';
    const parts = d.strategies.map((s) => `${STRATEGY_LABELS[s]} recall ${pct(this.metric(d, s, 'recall'))}`);
    return `Grouped bar chart of retrieval metrics. ${parts.join(', ')}.`;
  });

  ngOnDestroy(): void {
    this.stopStream?.();
  }

  protected y(v: number): number {
    return CHART.top + (1 - Math.max(0, Math.min(1, v))) * (CHART.h - CHART.top - CHART.bottom);
  }

  protected metric(d: EvalRun, s: string, key: string): number | undefined {
    return (d.summary.strategies?.[s] as Metrics | undefined)?.[key];
  }

  protected isBest(d: EvalRun, s: string, key: string): boolean {
    const v = this.metric(d, s, key);
    if (v == null || d.strategies.length < 2) return false;
    return v === Math.max(...d.strategies.map((x) => this.metric(d, x, key) ?? -1));
  }

  protected fmt(v: number | undefined): string {
    return pct(v);
  }

  protected short(v: number): string {
    return `${Math.round(v * 100)}%`;
  }

  protected docName(filename: string | null): string {
    return filename ? shortName(filename) : '';
  }

  protected firstRank(retrieved: { relevant: boolean }[] | undefined): number | null {
    const i = (retrieved ?? []).findIndex((h) => h.relevant);
    return i >= 0 ? i + 1 : null;
  }

  protected toggleRow(id: string): void {
    this.openRow.set(this.openRow() === id ? null : id);
  }

  protected toggleStrategy(s: Strategy): void {
    this.picked.update((set) => {
      const next = new Set(set);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  }

  protected async run(): Promise<void> {
    this.error.set(null);
    this.running.set(true);
    try {
      const run = await this.api.startRun({
        dataset_id: this.datasetId(),
        strategies: STRATEGIES.filter((s) => this.picked().has(s)),
        k: this.k(),
        with_answers: this.withAnswers(),
      });
      this.progress.set({ done: 0, total: run.items * run.strategies.length });
      this.stopStream = this.api.stream<EvalEvent>(
        `/eval/runs/${run.id}/events`,
        (e) => {
          if (e.type === 'progress') this.progress.set({ done: e.done, total: e.total });
          else if (e.status === 'failed') this.error.set(e.error ?? 'The run failed');
        },
        (err) => {
          if (err) this.error.set(err);
          this.running.set(false);
          this.progress.set(null);
          this.selectedRun.set(run.id);
          this.runs.reload();
        },
      );
    } catch (err) {
      this.error.set(errorMessage(err));
      this.running.set(false);
    }
  }

  protected docsFor(scope: Scope) {
    return (scope === 'library' ? this.store.library() : this.store.uploads()).filter((d) => d.status === 'ready');
  }

  protected parsePages(raw: string): number[] {
    return raw
      .split(/[,\s]+/)
      .map((p) => Number(p))
      .filter((n) => Number.isInteger(n) && n > 0);
  }

  protected async createDataset(): Promise<void> {
    try {
      this.error.set(null);
      const ds = await this.api.createDataset(this.newName().trim(), this.newScope());
      this.newName.set('');
      this.datasets.reload();
      this.editing.set(ds.id);
    } catch (err) {
      this.error.set(errorMessage(err));
    }
  }

  protected async deleteDataset(ds: Dataset): Promise<void> {
    if (!confirm(`Delete "${ds.name}" and its questions?`)) return;
    await this.api.deleteDataset(ds.id);
    if (this.pickedDataset() === ds.id) this.pickedDataset.set(null);
    this.datasets.reload();
  }

  protected async addItem(ds: Dataset): Promise<void> {
    try {
      this.error.set(null);
      await this.api.addItem(ds.id, { question: this.q().trim(), doc_id: this.qDoc(), gold_pages: this.parsePages(this.qPages()) });
      this.q.set('');
      this.qPages.set('');
      this.items.reload();
      this.datasets.reload();
    } catch (err) {
      this.error.set(errorMessage(err));
    }
  }

  protected async deleteItem(ds: Dataset, it: EvalItem): Promise<void> {
    await this.api.deleteItem(ds.id, it.id);
    this.items.reload();
    this.datasets.reload();
  }
}

/** A bar with 4px rounded top corners, square at the baseline. */
function barPath(x: number, y: number, w: number, base: number): string {
  const h = base - y;
  if (h <= 0) return '';
  const r = Math.min(4, w / 2, h);
  return `M${x},${base}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${base}Z`;
}
