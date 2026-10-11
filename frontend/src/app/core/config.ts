declare global {
  interface Window {
    __DOCUMIND__?: { apiUrl?: string };
  }
}

/** Backend base URL, from public/env.js (written at deploy time). */
export const API_URL = (window.__DOCUMIND__?.apiUrl ?? 'http://localhost:8002').replace(/\/$/, '');

export const REPO_URL = 'https://github.com/HimanshuChelani27/documind';
