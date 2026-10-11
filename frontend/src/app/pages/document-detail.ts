import { httpResource } from '@angular/common/http';
import { ChangeDetectionStrategy, Component, computed, effect, input } from '@angular/core';
import { RouterLink } from '@angular/router';

import { api } from '../core/api';
import { reflow } from '../core/format';
import { ChunkInfo, DocumentInfo } from '../core/models';
import { WORKSPACE_ID } from '../core/workspace';
import { Icon } from '../layout/icon';

@Component({
  selector: 'app-document-detail',
  imports: [RouterLink, Icon],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <a routerLink="/documents" class="inline-flex items-center gap-1 text-xs text-text-muted hover:text-text-primary">
      <app-icon name="chevron" cls="h-3.5 w-3.5 rotate-180" /> Documents
    </a>
    @if (doc.value(); as d) {
      <div class="mt-3 flex flex-wrap items-end justify-between gap-4">
        <div class="min-w-0">
          <h1 class="page-title truncate">{{ d.title }}</h1>
          <p class="page-sub">
            {{ d.pages }} pages · {{ d.chunks }} chunks · {{ avg() }} characters per chunk on average
          </p>
        </div>
        <a class="btn-ghost" [href]="pdf()" target="_blank" rel="noopener"><app-icon name="external" /> Open PDF</a>
      </div>
      <p class="mt-4 max-w-3xl text-sm text-text-muted">
        Chunks follow the document's structure: they break at headings first, then paragraph and sentence ends, and
        never cross a page, so every citation points at one page. Consecutive chunks in a section overlap slightly so
        a fact cut at a boundary still appears whole in one of them.
      </p>
    } @else if (doc.error()) {
      <p class="mt-6 text-sm text-deny">Document not found.</p>
    }

    <div class="mt-6 space-y-8">
      @for (group of pages(); track group.page) {
        <section>
          <h2 class="mb-3 font-mono text-xs uppercase tracking-wider text-text-muted">Page {{ group.page }}</h2>
          <div class="space-y-3">
            @for (c of group.chunks; track c.id) {
              <article
                [id]="'chunk-' + c.id"
                class="card p-4"
                [class]="c.id === chunk() ? '!border-accent/60 ring-1 ring-accent/30' : ''"
              >
                <div class="flex flex-wrap items-center gap-2 text-[11px]">
                  <span class="chip">#{{ c.ordinal }}</span>
                  @if (c.heading) {
                    <span class="chip !text-accent-soft">{{ c.heading }}</span>
                  }
                  <span class="ml-auto font-mono text-text-muted">chars {{ c.char_start }}–{{ c.char_start + c.text.length }} · {{ c.text.length }}</span>
                </div>
                <p class="mt-2 whitespace-pre-line text-sm leading-relaxed">{{ flow(c.text) }}</p>
              </article>
            }
          </div>
        </section>
      } @empty {
        @if (chunks.isLoading()) {
          <p class="text-sm text-text-muted">Loading chunks…</p>
        }
      }
    </div>
  `,
})
export class DocumentDetail {
  readonly id = input.required<string>();
  readonly chunk = input<string>();

  protected readonly doc = httpResource<DocumentInfo>(() => api(`/documents/${this.id()}`));
  protected readonly chunks = httpResource<ChunkInfo[]>(() => api(`/documents/${this.id()}/chunks`), {
    defaultValue: [],
  });
  protected readonly flow = reflow;

  protected readonly pages = computed(() => {
    const groups = new Map<number, ChunkInfo[]>();
    for (const c of this.chunks.value()) groups.set(c.page, [...(groups.get(c.page) ?? []), c]);
    return [...groups.entries()].map(([page, chunks]) => ({ page, chunks }));
  });
  protected readonly avg = computed(() => {
    const cs = this.chunks.value();
    return cs.length ? Math.round(cs.reduce((a, c) => a + c.text.length, 0) / cs.length) : 0;
  });
  protected readonly pdf = computed(() => api(`/documents/${this.id()}/file?workspace=${WORKSPACE_ID}`));

  constructor() {
    // Scroll to the chunk a citation linked to, once the chunks have rendered.
    effect(() => {
      const target = this.chunk();
      if (target && this.chunks.value().length) {
        setTimeout(() => document.getElementById('chunk-' + target)?.scrollIntoView({ block: 'center' }));
      }
    });
  }
}
