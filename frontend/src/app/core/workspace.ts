import { HttpInterceptorFn } from '@angular/common/http';

import { API_URL } from './config';

const KEY = 'documind.workspace';

function randomId(): string {
  const bytes = new Uint8Array(12);
  crypto.getRandomValues(bytes);
  return 'ws-' + Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * The browser's private upload workspace. No accounts: a random id lives in local storage and
 * goes to the API as `X-Workspace`. If storage is unavailable it lasts for this tab only.
 */
export const WORKSPACE_ID: string = (() => {
  try {
    const existing = localStorage.getItem(KEY);
    if (existing && /^[A-Za-z0-9_-]{8,64}$/.test(existing)) return existing;
    const id = randomId();
    localStorage.setItem(KEY, id);
    return id;
  } catch {
    return randomId();
  }
})();

export const workspaceInterceptor: HttpInterceptorFn = (req, next) =>
  req.url.startsWith(API_URL) ? next(req.clone({ setHeaders: { 'X-Workspace': WORKSPACE_ID } })) : next(req);
