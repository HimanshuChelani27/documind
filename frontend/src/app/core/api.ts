import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';

import { API_URL } from './config';
import { Dataset, DocumentInfo, EvalItem, EvalRun, Scope, Strategy } from './models';
import { WORKSPACE_ID } from './workspace';

export const api = (path: string) => `${API_URL}${path}`;

/** Pulls a readable message out of an HttpErrorResponse / fetch failure. */
export function errorMessage(err: unknown): string {
  const e = err as { error?: { detail?: unknown }; status?: number; message?: string };
  const detail = e?.error?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d: { msg?: string }) => d.msg).join('; ');
  if (e?.status === 0) return 'Could not reach the DocuMind API. It may be waking up; try again in a moment.';
  return e?.message ?? 'Something went wrong';
}

/** Write operations and streams. Reads use httpResource() directly in components. */
@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);

  upload(file: File): Promise<{ job: string; document: DocumentInfo }> {
    const form = new FormData();
    form.append('file', file);
    return firstValueFrom(this.http.post<{ job: string; document: DocumentInfo }>(api('/documents'), form));
  }

  deleteDocument(id: string): Promise<unknown> {
    return firstValueFrom(this.http.delete(api(`/documents/${id}`)));
  }

  createDataset(name: string, scope: Scope): Promise<Dataset> {
    return firstValueFrom(this.http.post<Dataset>(api('/eval/datasets'), { name, scope }));
  }

  deleteDataset(id: string): Promise<unknown> {
    return firstValueFrom(this.http.delete(api(`/eval/datasets/${id}`)));
  }

  addItem(datasetId: string, body: { question: string; doc_id: string; gold_pages: number[]; answer?: string }) {
    return firstValueFrom(this.http.post<EvalItem>(api(`/eval/datasets/${datasetId}/items`), body));
  }

  deleteItem(datasetId: string, itemId: string): Promise<unknown> {
    return firstValueFrom(this.http.delete(api(`/eval/datasets/${datasetId}/items/${itemId}`)));
  }

  startRun(body: { dataset_id: string; strategies: Strategy[]; k: number; with_answers: boolean }) {
    return firstValueFrom(this.http.post<EvalRun>(api('/eval/runs'), body));
  }

  /**
   * Server-Sent Events over fetch (so it can POST a body and send the workspace header).
   * Returns a function that stops the stream.
   */
  stream<T>(
    path: string,
    onEvent: (e: T) => void,
    onDone: (error?: string) => void,
    body?: unknown,
  ): () => void {
    const controller = new AbortController();
    (async () => {
      try {
        const res = await fetch(api(path), {
          method: body === undefined ? 'GET' : 'POST',
          headers: {
            'X-Workspace': WORKSPACE_ID,
            ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
          },
          body: body === undefined ? undefined : JSON.stringify(body),
          signal: controller.signal,
        });
        if (!res.ok || !res.body) {
          let detail = `HTTP ${res.status}`;
          try {
            const data = await res.json();
            if (typeof data?.detail === 'string') detail = data.detail;
          } catch {
            /* not JSON */
          }
          throw new Error(detail);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let idx: number;
          while ((idx = buffer.indexOf('\n\n')) >= 0) {
            const chunk = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 2);
            const data = chunk
              .split('\n')
              .filter((l) => l.startsWith('data: '))
              .map((l) => l.slice(6))
              .join('\n');
            if (data) onEvent(JSON.parse(data) as T);
          }
        }
        onDone();
      } catch (e) {
        if (!controller.signal.aborted) {
          const msg = e instanceof Error ? e.message : 'Stream failed';
          onDone(msg === 'Failed to fetch' ? 'Could not reach the DocuMind API. It may be waking up; try again.' : msg);
        }
      }
    })();
    return () => controller.abort();
  }
}
