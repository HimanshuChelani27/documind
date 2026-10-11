import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';

import { api, errorMessage } from '../core/api';
import { formatBytes, timeAgo } from '../core/format';
import { LibraryStore } from '../core/library';
import { DocumentInfo } from '../core/models';
import { WORKSPACE_ID } from '../core/workspace';
import { Icon } from '../layout/icon';

@Component({
  selector: 'app-documents',
  imports: [RouterLink, Icon],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 class="page-title">Documents</h1>
        <p class="page-sub max-w-2xl">
          The shared sample library, plus anything you upload (private to this browser, deleted after 24 hours).
          Open a document to see exactly how it was chunked.
        </p>
      </div>
      <label class="btn-primary cursor-pointer">
        <app-icon name="upload" /> Upload PDF
        <input type="file" accept="application/pdf,.pdf" class="sr-only" (change)="onFile($event)" />
      </label>
    </div>

    @if (store.upload(); as u) {
      <div class="card mt-4 flex items-center gap-4 p-4 text-sm" aria-live="polite">
        <div class="min-w-0 flex-1">
          <p class="truncate text-text-primary">{{ u.filename }}</p>
          <div class="mt-2 h-1.5 overflow-hidden rounded-full bg-surface-2">
            <div class="h-full rounded-full transition-all" [class]="u.error ? 'bg-deny' : 'bg-accent'" [style.width.%]="u.progress * 100"></div>
          </div>
        </div>
        <p class="shrink-0 text-xs" [class]="u.error ? 'text-deny' : 'text-text-muted'">{{ u.error ?? store.stageLabel(u.stage) }}</p>
      </div>
    }
    @if (error()) {
      <p class="mt-4 text-sm text-deny">{{ error() }}</p>
    }

    <section class="card mt-6 overflow-hidden">
      <div class="overflow-x-auto">
        <table class="w-full min-w-[640px]">
          <thead>
            <tr>
              <th class="table-head">Document</th>
              <th class="table-head">Where</th>
              <th class="table-head text-right">Pages</th>
              <th class="table-head text-right">Chunks</th>
              <th class="table-head text-right">Size</th>
              <th class="table-head">Added</th>
              <th class="table-head"><span class="sr-only">Actions</span></th>
            </tr>
          </thead>
          <tbody>
            @for (d of store.docs.value(); track d.id) {
              <tr class="transition hover:bg-surface-2">
                <td class="table-cell max-w-xs">
                  <a [routerLink]="['/documents', d.id]" class="block truncate text-text-primary hover:text-accent-soft">{{ d.title }}</a>
                  <span class="font-mono text-[11px] text-text-muted">{{ d.filename }}</span>
                  @if (d.status !== 'ready') {
                    <span class="mt-1 block text-xs" [class]="d.status === 'failed' ? 'text-deny' : 'text-warn'">
                      {{ d.status === 'failed' ? d.error || 'Failed' : 'Processing…' }}
                    </span>
                  }
                </td>
                <td class="table-cell">
                  @if (d.scope === 'library') {
                    <span class="chip">sample library</span>
                  } @else {
                    <span class="chip !text-accent-soft"><app-icon name="lock" cls="mr-1 h-3 w-3" /> your upload</span>
                  }
                </td>
                <td class="table-cell text-right tabular-nums">{{ d.pages }}</td>
                <td class="table-cell text-right tabular-nums">{{ d.chunks }}</td>
                <td class="table-cell text-right tabular-nums text-text-muted">{{ bytes(d.size_bytes) }}</td>
                <td class="table-cell whitespace-nowrap text-text-muted">{{ ago(d.created_at) }}</td>
                <td class="table-cell">
                  <div class="flex justify-end gap-1">
                    <a class="grid h-8 w-8 place-items-center rounded-full text-text-muted hover:bg-surface hover:text-text-primary" [href]="pdf(d)" target="_blank" rel="noopener" [attr.aria-label]="'Open ' + d.filename">
                      <app-icon name="external" />
                    </a>
                    @if (d.scope === 'workspace' && d.status !== 'processing') {
                      <button class="grid h-8 w-8 place-items-center rounded-full text-text-muted hover:bg-surface hover:text-deny" (click)="remove(d)" [attr.aria-label]="'Delete ' + d.filename">
                        <app-icon name="trash" />
                      </button>
                    }
                  </div>
                </td>
              </tr>
            } @empty {
              <tr>
                <td colspan="7" class="table-cell py-10 text-center text-text-muted">
                  {{ store.docs.isLoading() ? 'Loading…' : store.docs.error() ? 'Could not reach the API.' : 'No documents.' }}
                </td>
              </tr>
            }
          </tbody>
        </table>
      </div>
    </section>
  `,
})
export class Documents {
  protected readonly store = inject(LibraryStore);
  protected readonly error = signal<string | null>(null);
  protected readonly ago = timeAgo;
  protected readonly bytes = formatBytes;

  constructor() {
    this.store.docs.reload();
  }

  protected async onFile(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (file) await this.store.uploadFile(file);
  }

  protected async remove(d: DocumentInfo): Promise<void> {
    if (!confirm(`Delete ${d.filename}? This removes it and its index.`)) return;
    try {
      this.error.set(null);
      await this.store.remove(d);
    } catch (err) {
      this.error.set(errorMessage(err));
    }
  }

  protected pdf(d: DocumentInfo): string {
    return api(`/documents/${d.id}/file?workspace=${WORKSPACE_ID}`);
  }
}
