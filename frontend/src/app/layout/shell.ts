import { httpResource } from '@angular/common/http';
import { ChangeDetectionStrategy, Component, signal } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

import { api } from '../core/api';
import { REPO_URL } from '../core/config';
import { LlmProvider } from '../core/models';
import { Icon } from './icon';

const NAV = [
  { path: '/chat', label: 'Ask', icon: 'message' },
  { path: '/documents', label: 'Documents', icon: 'file' },
  { path: '/evaluate', label: 'Evaluate', icon: 'chart' },
];

@Component({
  selector: 'app-shell',
  imports: [RouterOutlet, RouterLink, RouterLinkActive, Icon],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="flex min-h-screen">
      <aside
        class="fixed inset-y-0 left-0 z-40 w-56 -translate-x-full border-r border-border-soft bg-bg-raised transition-transform md:translate-x-0"
        [class.translate-x-0]="menuOpen()"
      >
        <div class="flex h-16 items-center gap-2.5 px-5">
          <img src="favicon.svg" alt="" class="h-8 w-8" />
          <div>
            <p class="text-sm font-semibold text-text-primary">DocuMind</p>
            <p class="text-[11px] text-text-muted">Hybrid RAG + evals</p>
          </div>
        </div>
        <nav class="mt-2 space-y-0.5 px-3" aria-label="Main">
          @for (item of nav; track item.path) {
            <a
              [routerLink]="item.path"
              routerLinkActive="!bg-surface-2 !text-text-primary [&>app-icon]:!text-accent"
              (click)="menuOpen.set(false)"
              class="flex items-center gap-3 rounded-xl px-3 py-2 text-sm text-text-secondary transition hover:bg-surface hover:text-text-primary"
            >
              <app-icon [name]="item.icon" class="text-text-muted" />
              {{ item.label }}
            </a>
          }
        </nav>
        <div class="absolute inset-x-0 bottom-0 space-y-2 border-t border-border-soft p-4 text-xs text-text-muted">
          <p class="leading-relaxed">
            Uploads are private to this browser and deleted after 24 hours.
          </p>
          <a [href]="repo" target="_blank" rel="noopener" class="flex items-center gap-2 hover:text-text-primary">
            <app-icon name="github" cls="h-3.5 w-3.5" /> Source on GitHub
          </a>
        </div>
      </aside>
      @if (menuOpen()) {
        <div class="fixed inset-0 z-30 bg-black/60 md:hidden" (click)="menuOpen.set(false)"></div>
      }

      <div class="min-w-0 flex-1 md:pl-56">
        <header
          class="sticky top-0 z-20 flex h-14 items-center justify-between gap-3 border-b border-border-soft bg-bg/80 px-4 backdrop-blur sm:px-6"
        >
          <button
            class="grid h-9 w-9 place-items-center rounded-full border border-border md:hidden"
            (click)="menuOpen.set(true)"
            aria-label="Open menu"
          >
            <app-icon name="menu" />
          </button>
          <p class="hidden truncate text-xs text-text-muted lg:block">
            Most RAG demos stop at "it answered something". DocuMind shows which retrieval strategy actually works, and ties every answer to a page.
          </p>
          @if (provider.value(); as p) {
            <span
              class="inline-flex shrink-0 items-center gap-2 rounded-full border px-3 py-1 font-mono text-[11px]"
              [class]="p.mode === 'live' ? 'border-ok/40 text-ok' : 'border-border text-text-muted'"
              [title]="
                p.mode === 'live'
                  ? 'Answers are written by an LLM from the retrieved passages'
                  : 'No LLM key configured: answers are the best-matching source sentences. Retrieval, reranking and citations are real.'
              "
            >
              <span class="h-1.5 w-1.5 rounded-full" [class]="p.mode === 'live' ? 'bg-ok' : 'bg-text-muted'"></span>
              {{ p.mode === 'live' ? p.provider + ' · ' + p.model : 'Extractive mode (no LLM key)' }}
            </span>
          } @else if (provider.error()) {
            <span class="rounded-full border border-warn/40 px-3 py-1 font-mono text-[11px] text-warn">API offline</span>
          }
        </header>
        <main class="mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8">
          <router-outlet />
        </main>
      </div>
    </div>
  `,
})
export class Shell {
  protected readonly nav = NAV;
  protected readonly repo = REPO_URL;
  protected readonly menuOpen = signal(false);
  protected readonly provider = httpResource<LlmProvider>(() => api('/llm/provider'));
}
