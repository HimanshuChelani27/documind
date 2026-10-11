import { DecimalPipe } from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnDestroy,
  computed,
  inject,
  signal,
  viewChild,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';

import { ApiService, api } from '../core/api';
import { answerParts, formatMs, highlight, queryTerms, reflow, shortName } from '../core/format';
import { LibraryStore } from '../core/library';
import {
  ChatEvent,
  DoneEvent,
  RetrievalEvent,
  STRATEGIES,
  STRATEGY_HINTS,
  STRATEGY_LABELS,
  Scope,
  Source,
  StageHit,
  Strategy,
} from '../core/models';
import { WORKSPACE_ID } from '../core/workspace';
import { Icon } from '../layout/icon';

interface Turn {
  id: number;
  question: string;
  strategy: Strategy;
  answer: string;
  retrieval: RetrievalEvent | null;
  sources: Source[] | null;
  done: DoneEvent | null;
  error: string | null;
  streaming: boolean;
  showRetrieval: boolean;
}

const SUGGESTIONS: Record<Scope, string[]> = {
  library: [
    'What is the per diem for international travel?',
    'What does error code E03 mean?',
    'The motor is running but nothing comes out of the hose. What should I check?',
    'Why was the second quarter difficult?',
    'Can I roll unused holiday into the next year?',
    'Which part of the business grew the fastest?',
  ],
  workspace: ['Summarise the main points of this document', 'What are the key numbers mentioned?'],
};

type StageKey = 'dense' | 'bm25' | 'fused' | 'reranked';

const STAGE_TITLES: Record<StageKey, string> = {
  dense: 'Dense (FAISS)',
  bm25: 'BM25',
  fused: 'Fused (RRF)',
  reranked: 'Reranked (cross-encoder)',
};

@Component({
  selector: 'app-chat',
  imports: [FormsModule, RouterLink, Icon, DecimalPipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="grid grid-cols-1 gap-6 lg:grid-cols-[17rem_minmax(0,1fr)]">
      <!-- Controls -->
      <aside class="min-w-0 space-y-4 lg:sticky lg:top-20 lg:self-start">
        <section class="card p-4">
          <div class="flex rounded-full border border-border-soft bg-bg-raised p-1" role="tablist">
            <button
              role="tab"
              class="seg flex-1"
              [class]="scope() === 'library' ? 'bg-surface-2 text-text-primary' : 'text-text-muted hover:text-text-primary'"
              [attr.aria-selected]="scope() === 'library'"
              (click)="setScope('library')"
            >
              Sample library
            </button>
            <button
              role="tab"
              class="seg flex-1"
              [class]="scope() === 'workspace' ? 'bg-surface-2 text-text-primary' : 'text-text-muted hover:text-text-primary'"
              [attr.aria-selected]="scope() === 'workspace'"
              (click)="setScope('workspace')"
            >
              My uploads ({{ store.uploads().length }})
            </button>
          </div>

          <ul class="mt-3 space-y-1">
            @for (d of scopeDocs(); track d.id) {
              <li>
                <label
                  class="flex cursor-pointer items-start gap-2.5 rounded-xl px-2 py-1.5 text-sm hover:bg-surface-2"
                  [class.opacity-50]="d.status !== 'ready'"
                >
                  <input
                    type="checkbox"
                    class="mt-0.5 accent-[var(--color-accent)]"
                    [checked]="isSelected(d.id)"
                    [disabled]="d.status !== 'ready'"
                    (change)="toggleDoc(d.id)"
                  />
                  <span class="min-w-0">
                    <span class="block truncate text-text-primary" [title]="d.title">{{ d.title }}</span>
                    <span class="font-mono text-[11px] text-text-muted">
                      @if (d.status === 'ready') {
                        {{ d.pages }} pages · {{ d.chunks }} chunks
                      } @else {
                        {{ d.status }}
                      }
                    </span>
                  </span>
                </label>
              </li>
            } @empty {
              <li class="px-2 py-3 text-xs text-text-muted">
                @if (scope() === 'workspace') {
                  No uploads yet. Add a PDF below; it stays private to this browser.
                } @else {
                  Loading the sample library…
                }
              </li>
            }
          </ul>
          <p class="mt-2 px-2 text-[11px] text-text-muted">
            {{ selected().size ? selected().size + ' selected' : 'Searching all documents' }}
          </p>

          @if (scope() === 'workspace') {
            <label
              class="mt-3 flex cursor-pointer flex-col items-center gap-1.5 rounded-xl border border-dashed border-border px-3 py-4 text-center text-xs text-text-muted transition hover:border-accent/60 hover:text-text-primary"
              [class.border-accent]="dragging()"
              (dragover)="$event.preventDefault(); dragging.set(true)"
              (dragleave)="dragging.set(false)"
              (drop)="onDrop($event)"
            >
              <app-icon name="upload" cls="h-5 w-5" />
              <span><span class="text-accent-soft">Choose a PDF</span> or drop it here</span>
              <span class="text-[11px]">Up to 5 MB, 40 pages, 5 files</span>
              <input type="file" accept="application/pdf,.pdf" class="sr-only" (change)="onFile($event)" />
            </label>
            @if (store.upload(); as u) {
              <div class="mt-3 rounded-xl border border-border-soft bg-bg-raised p-3 text-xs" aria-live="polite">
                <p class="truncate text-text-primary">{{ u.filename }}</p>
                <div class="mt-2 h-1.5 overflow-hidden rounded-full bg-surface-2">
                  <div
                    class="h-full rounded-full transition-all"
                    [class]="u.error ? 'bg-deny' : 'bg-accent'"
                    [style.width.%]="u.progress * 100"
                  ></div>
                </div>
                <p class="mt-1.5" [class]="u.error ? 'text-deny' : 'text-text-muted'">
                  {{ u.error ?? store.stageLabel(u.stage) }}
                </p>
              </div>
            }
          }
        </section>

        <section class="card p-4">
          <p class="label">Retrieval strategy</p>
          <div class="grid grid-cols-2 gap-1.5">
            @for (s of strategies; track s) {
              <button
                class="rounded-xl border px-2.5 py-1.5 text-left text-xs transition"
                [class]="
                  strategy() === s
                    ? 'border-accent/60 bg-accent/10 text-text-primary'
                    : 'border-border-soft text-text-secondary hover:border-border hover:text-text-primary'
                "
                (click)="strategy.set(s)"
              >
                {{ labels[s] }}
              </button>
            }
          </div>
          <p class="mt-2 text-[11px] leading-relaxed text-text-muted">{{ hints[strategy()] }}</p>
          <label class="label mt-4" for="topk">Sources per answer</label>
          <select id="topk" class="input" [ngModel]="topK()" (ngModelChange)="topK.set(+$event)">
            @for (k of [3, 5, 8]; track k) {
              <option [value]="k">{{ k }}</option>
            }
          </select>
        </section>
      </aside>

      <!-- Conversation -->
      <section class="flex min-h-[70vh] min-w-0 flex-col">
        @if (!turns().length) {
          <div class="mb-6">
            <h1 class="page-title">Ask your documents</h1>
            <p class="page-sub max-w-2xl">
              Every answer cites the page it came from. Open
              <span class="text-text-secondary">How this was retrieved</span> to see what dense search, BM25,
              fusion and the reranker each found, or switch strategy and ask again.
            </p>
            <div class="mt-5 flex flex-wrap gap-2">
              @for (q of suggestions(); track q) {
                <button class="btn-ghost !py-1.5 text-left !text-xs" (click)="ask(q)">{{ q }}</button>
              }
            </div>
          </div>
        }

        <div class="flex-1 space-y-6">
          @for (t of turns(); track t.id) {
            <article class="space-y-3">
              <div class="flex justify-end">
                <p class="max-w-[85%] rounded-2xl rounded-br-md bg-surface-2 px-4 py-2.5 text-sm text-text-primary">
                  {{ t.question }}
                </p>
              </div>

              <div class="card p-4 sm:p-5">
                <div class="mb-2 flex flex-wrap items-center gap-2 text-[11px]">
                  <span class="chip">{{ labels[t.strategy] }}</span>
                  @if (t.done; as d) {
                    <span class="chip" [class]="d.mode === 'live' ? '!text-ok' : ''">
                      {{ d.mode === 'live' ? d.provider + ' · ' + d.model : 'Extractive (no LLM)' }}
                    </span>
                    <span class="font-mono text-text-muted">
                      retrieval {{ ms(d.timings_ms['retrieval_total']) }} · total {{ ms(d.timings_ms['total']) }}
                    </span>
                  } @else if (t.streaming) {
                    <span class="font-mono text-text-muted">{{ t.retrieval ? 'answering…' : 'retrieving…' }}</span>
                  }
                </div>

                @if (t.error) {
                  <p class="text-sm text-deny">{{ t.error }}</p>
                } @else if (!t.answer && t.streaming) {
                  <div class="space-y-2" aria-hidden="true">
                    <div class="h-3 w-4/5 animate-pulse rounded bg-surface-2"></div>
                    <div class="h-3 w-3/5 animate-pulse rounded bg-surface-2"></div>
                  </div>
                } @else {
                  <p class="whitespace-pre-line text-[15px] leading-relaxed text-text-primary">
                    @for (part of parts(t.answer); track $index) {
                      @if (isCite(part)) {
                        <button
                          class="cite"
                          (click)="openSource(t, part.cite)"
                          [attr.aria-label]="'Show source ' + part.cite"
                        >
                          {{ part.cite }}
                        </button>
                      } @else {
                        {{ part.text }}
                      }
                    }
                    @if (t.streaming) {
                      <span class="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-accent align-middle"></span>
                    }
                  </p>
                }
                @if (t.done?.fallback_reason) {
                  <p class="mt-2 text-xs text-warn">{{ t.done?.fallback_reason }}</p>
                }

                @if (t.sources?.length) {
                  <div class="mt-4 grid gap-2 sm:grid-cols-2">
                    @for (s of t.sources; track s.n) {
                      <button
                        class="rounded-xl border p-3 text-left transition hover:border-accent/50 hover:bg-surface-2"
                        [class]="s.cited ? 'border-accent/30' : 'border-border-soft opacity-70'"
                        (click)="openSource(t, s.n)"
                      >
                        <div class="flex items-center gap-2 text-xs">
                          <span class="cite !mx-0">{{ s.n }}</span>
                          <span class="truncate text-text-primary">{{ s.doc_title }}</span>
                          <span class="ml-auto shrink-0 font-mono text-text-muted">p. {{ s.page }}</span>
                        </div>
                        <p class="mt-1.5 line-clamp-2 text-xs text-text-muted">{{ s.text }}</p>
                      </button>
                    }
                  </div>
                }

                @if (t.retrieval; as r) {
                  <button
                    class="mt-4 inline-flex items-center gap-1.5 text-xs text-text-secondary hover:text-text-primary"
                    (click)="toggleRetrieval(t.id)"
                    [attr.aria-expanded]="t.showRetrieval"
                  >
                    <app-icon name="chevron" cls="h-3.5 w-3.5 transition-transform" [class.rotate-90]="t.showRetrieval" />
                    How this was retrieved
                  </button>
                  @if (t.showRetrieval) {
                    <div class="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                      @for (stage of stageKeys(r); track stage) {
                        <div class="rounded-xl border border-border-soft bg-bg-raised p-3">
                          <p class="flex items-baseline justify-between text-xs font-medium text-text-primary">
                            {{ stageTitle(stage) }}
                            <span class="font-mono text-[10px] font-normal text-text-muted">
                              {{ ms(r.timings_ms[stage === 'fused' ? 'fusion' : stage === 'reranked' ? 'rerank' : stage]) }}
                            </span>
                          </p>
                          <ol class="mt-2 space-y-1">
                            @for (h of r.stages[stage]!.slice(0, 8); track h.chunk_id; let i = $index) {
                              <li
                                class="flex items-center gap-2 rounded-lg px-1.5 py-1 font-mono text-[11px]"
                                [class]="finalIds(r).has(h.chunk_id) ? 'bg-accent/10 text-text-primary' : 'text-text-muted'"
                                [title]="h.filename + ' · page ' + h.page"
                              >
                                <span class="w-4 text-right">{{ i + 1 }}</span>
                                <span class="min-w-0 flex-1 truncate">{{ short(h) }}</span>
                                <span class="tabular-nums">{{ h.score | number: '1.2-3' }}</span>
                              </li>
                            }
                          </ol>
                        </div>
                      }
                    </div>
                    <p class="mt-2 text-[11px] text-text-muted">
                      Highlighted rows made it into the final top {{ r.hits.length }}. RRF scores use only ranks
                      (1/(60+rank) per list), so dense and BM25 scores never need calibrating.
                    </p>
                  }
                }
              </div>
            </article>
          }
        </div>

        <form
          class="sticky bottom-0 mt-6 bg-gradient-to-t from-bg via-bg to-transparent pb-2 pt-4"
          (submit)="$event.preventDefault(); ask(draft())"
        >
          <div class="card flex items-end gap-2 p-2">
            <textarea
              #box
              rows="1"
              class="max-h-40 min-h-10 flex-1 resize-none bg-transparent px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none"
              placeholder="Ask a question about {{ scope() === 'library' ? 'the sample documents' : 'your uploads' }}…"
              [ngModel]="draft()"
              (ngModelChange)="draft.set($event)"
              name="question"
              maxlength="500"
              (keydown.enter)="onEnter($event)"
              aria-label="Question"
            ></textarea>
            <button class="btn-primary h-10 w-10 !p-0" [disabled]="busy() || draft().trim().length < 2" aria-label="Ask">
              <app-icon name="send" />
            </button>
          </div>
        </form>
      </section>
    </div>

    <!-- Source viewer -->
    @if (viewing(); as v) {
      <div class="fixed inset-0 z-50 flex items-end justify-center bg-black/60 p-0 sm:items-center sm:p-6" (click)="viewing.set(null)">
        <div
          class="card max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-b-none p-5 sm:rounded-2xl"
          role="dialog"
          aria-modal="true"
          [attr.aria-label]="'Source ' + v.source.n"
          (click)="$event.stopPropagation()"
        >
          <div class="flex items-start gap-3">
            <span class="cite !mx-0 mt-1">{{ v.source.n }}</span>
            <div class="min-w-0 flex-1">
              <p class="font-medium text-text-primary">{{ v.source.doc_title }}</p>
              <p class="mt-0.5 font-mono text-xs text-text-muted">
                page {{ v.source.page }}{{ v.source.heading ? ' · ' + v.source.heading : '' }} · score
                {{ v.source.score | number: '1.2-3' }}
              </p>
            </div>
            <button class="grid h-8 w-8 place-items-center rounded-full hover:bg-surface-2" (click)="viewing.set(null)" aria-label="Close">
              <app-icon name="x" />
            </button>
          </div>
          <p
            class="mt-4 whitespace-pre-line rounded-xl border border-border-soft bg-bg-raised p-4 text-sm leading-relaxed text-text-secondary"
            [innerHTML]="v.html"
          ></p>
          <div class="mt-4 flex flex-wrap gap-2">
            <a class="btn-ghost !text-xs" [href]="pdfUrl(v.source)" target="_blank" rel="noopener">
              <app-icon name="external" cls="h-3.5 w-3.5" /> Open PDF at page {{ v.source.page }}
            </a>
            <a class="btn-ghost !text-xs" [routerLink]="['/documents', v.source.doc_id]" [queryParams]="{ chunk: v.source.chunk_id }">
              <app-icon name="layers" cls="h-3.5 w-3.5" /> See all chunks
            </a>
          </div>
        </div>
      </div>
    }
  `,
})
export class Chat implements OnDestroy {
  protected readonly store = inject(LibraryStore);
  private readonly api = inject(ApiService);
  private readonly box = viewChild<ElementRef<HTMLTextAreaElement>>('box');

  protected readonly strategies = STRATEGIES;
  protected readonly labels = STRATEGY_LABELS;
  protected readonly hints = STRATEGY_HINTS;
  protected readonly ms = formatMs;

  protected readonly scope = signal<Scope>('library');
  protected readonly strategy = signal<Strategy>('hybrid_rerank');
  protected readonly topK = signal(5);
  protected readonly selected = signal<Set<string>>(new Set());
  protected readonly draft = signal('');
  protected readonly turns = signal<Turn[]>([]);
  protected readonly dragging = signal(false);
  protected readonly viewing = signal<{ source: Source; html: string } | null>(null);
  protected readonly busy = computed(() => this.turns().some((t) => t.streaming));
  protected readonly scopeDocs = computed(() =>
    this.scope() === 'library' ? this.store.library() : this.store.uploads(),
  );
  protected readonly suggestions = computed(() => SUGGESTIONS[this.scope()]);

  private nextId = 1;
  private stop: (() => void) | null = null;

  ngOnDestroy(): void {
    this.stop?.();
  }

  protected setScope(scope: Scope): void {
    this.scope.set(scope);
    this.selected.set(new Set());
  }

  protected isSelected(id: string): boolean {
    return this.selected().has(id);
  }

  protected toggleDoc(id: string): void {
    this.selected.update((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  protected onEnter(event: Event): void {
    const e = event as KeyboardEvent;
    if (e.shiftKey || e.isComposing) return;
    e.preventDefault();
    this.ask(this.draft());
  }

  protected async onFile(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (file) await this.uploadAndSelect(file);
  }

  protected async onDrop(event: DragEvent): Promise<void> {
    event.preventDefault();
    this.dragging.set(false);
    const file = event.dataTransfer?.files?.[0];
    if (file) await this.uploadAndSelect(file);
  }

  private async uploadAndSelect(file: File): Promise<void> {
    const doc = await this.store.uploadFile(file);
    if (doc?.status === 'ready') this.selected.set(new Set([doc.id]));
  }

  protected ask(raw: string): void {
    const question = raw.trim();
    if (question.length < 2 || this.busy()) return;
    this.draft.set('');
    const id = this.nextId++;
    const turn: Turn = {
      id,
      question,
      strategy: this.strategy(),
      answer: '',
      retrieval: null,
      sources: null,
      done: null,
      error: null,
      streaming: true,
      showRetrieval: false,
    };
    this.turns.update((ts) => [...ts, turn]);
    setTimeout(() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }));
    const body = {
      question,
      scope: this.scope(),
      strategy: this.strategy(),
      top_k: this.topK(),
      doc_ids: this.selected().size ? [...this.selected()] : null,
    };
    this.stop = this.api.stream<ChatEvent>(
      '/chat',
      (e) => {
        switch (e.type) {
          case 'retrieval':
            return this.patch(id, { retrieval: e });
          case 'token':
            return this.patch(id, (t) => ({ answer: t.answer + e.text }));
          case 'citations':
            return this.patch(id, { answer: e.answer, sources: e.sources });
          case 'done':
            return this.patch(id, { done: e });
        }
      },
      (error) => {
        this.patch(id, { streaming: false, ...(error ? { error } : {}) });
        this.box()?.nativeElement.focus();
      },
      body,
    );
  }

  private patch(id: number, change: Partial<Turn> | ((t: Turn) => Partial<Turn>)): void {
    this.turns.update((ts) =>
      ts.map((t) => (t.id === id ? { ...t, ...(typeof change === 'function' ? change(t) : change) } : t)),
    );
  }

  protected toggleRetrieval(id: number): void {
    this.patch(id, (t) => ({ showRetrieval: !t.showRetrieval }));
  }

  protected openSource(t: Turn, n: number): void {
    const source = t.sources?.find((s) => s.n === n);
    if (!source) return;
    this.viewing.set({ source, html: highlight(reflow(source.text), queryTerms(t.question)) });
  }

  protected pdfUrl(s: Source): string {
    return api(`/documents/${s.doc_id}/file?workspace=${WORKSPACE_ID}#page=${s.page}`);
  }

  protected parts = answerParts;

  protected isCite(part: { text: string } | { cite: number }): part is { cite: number } {
    return 'cite' in part;
  }

  protected stageKeys(r: RetrievalEvent): StageKey[] {
    return (['dense', 'bm25', 'fused', 'reranked'] as const).filter((k) => r.stages[k]?.length);
  }

  protected stageTitle(stage: StageKey): string {
    return STAGE_TITLES[stage];
  }

  protected finalIds(r: RetrievalEvent): Set<string> {
    return new Set(r.hits.map((h) => h.chunk_id));
  }

  protected short(h: StageHit): string {
    return `${shortName(h.filename)} · p${h.page}`;
  }
}
