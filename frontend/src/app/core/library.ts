import { httpResource } from '@angular/common/http';
import { Injectable, computed, inject, signal } from '@angular/core';

import { ApiService, api, errorMessage } from './api';
import { DocumentInfo, IngestEvent } from './models';

export interface UploadState {
  filename: string;
  stage: string;
  progress: number;
  error: string | null;
}

const STAGE_LABELS: Record<string, string> = {
  upload: 'Uploading',
  parse: 'Parsing pages',
  chunk: 'Chunking',
  embed: 'Embedding',
  index: 'Indexing',
  done: 'Ready',
  failed: 'Failed',
};

/** Documents visible to this browser (sample library + private uploads) and the upload flow. */
@Injectable({ providedIn: 'root' })
export class LibraryStore {
  private readonly api = inject(ApiService);
  readonly docs = httpResource<DocumentInfo[]>(() => api('/documents'), { defaultValue: [] });
  readonly library = computed(() => this.docs.value().filter((d) => d.scope === 'library'));
  readonly uploads = computed(() => this.docs.value().filter((d) => d.scope === 'workspace'));
  readonly upload = signal<UploadState | null>(null);

  stageLabel(stage: string): string {
    return STAGE_LABELS[stage] ?? stage;
  }

  async uploadFile(file: File): Promise<DocumentInfo | null> {
    if (!/\.pdf$/i.test(file.name) && file.type !== 'application/pdf') {
      this.upload.set({ filename: file.name, stage: 'failed', progress: 1, error: 'Only PDF files are supported' });
      return null;
    }
    if (file.size > 5 * 1024 * 1024) {
      this.upload.set({ filename: file.name, stage: 'failed', progress: 1, error: 'The limit is 5 MB per file' });
      return null;
    }
    this.upload.set({ filename: file.name, stage: 'upload', progress: 0.02, error: null });
    let job: string;
    try {
      job = (await this.api.upload(file)).job;
    } catch (err) {
      this.upload.set({ filename: file.name, stage: 'failed', progress: 1, error: errorMessage(err) });
      return null;
    }
    this.docs.reload();
    return new Promise((resolve) => {
      let result: DocumentInfo | null = null;
      this.api.stream<IngestEvent>(
        `/ingest/${job}/events`,
        (e) => {
          if (e.type === 'progress') {
            this.upload.update((u) => u && { ...u, stage: e.stage, progress: Math.max(u.progress, e.progress) });
          } else {
            result = e.document;
            const failed = !e.document || e.document.status === 'failed';
            this.upload.set({
              filename: file.name,
              stage: failed ? 'failed' : 'done',
              progress: 1,
              error: failed ? (e.document?.error ?? e.error ?? 'Ingestion failed') : null,
            });
          }
        },
        (error) => {
          if (error) this.upload.update((u) => u && { ...u, stage: 'failed', error });
          this.docs.reload();
          resolve(result);
        },
      );
    });
  }

  async remove(doc: DocumentInfo): Promise<void> {
    await this.api.deleteDocument(doc.id);
    this.docs.reload();
  }
}
