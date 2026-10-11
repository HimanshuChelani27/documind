# Plan: DocuMind, hybrid RAG with built-in evaluation (portfolio project #2)

> **On approval, step 0:** create `~/Desktop/documind/` and save this plan there as `PLAN.md`. The user will run the build in a separate Claude Code session started in that folder, so this plan is written to be self-contained.

## Context
Himanshu Chelani is building a dark-theme portfolio (Astro, in `~/Desktop/himanshu website/site`, its own git repo) for recruiters and freelance clients.

Its personal-projects section has four cards:
- **AgentHub** (`~/Desktop/agenthub`): built.
- **DocuMind**: this plan.
- **EvalForge** and **KisanVoice**: later.

The DocuMind card already promises the following (from `site/src/data/projects.ts`, slug `documind`, status "In Progress", repo `github.com/HimanshuChelani27/documind`):
- PDF ingestion with semantic chunking and metadata.
- Hybrid retrieval: FAISS dense + BM25 sparse → reciprocal-rank fusion → cross-encoder rerank.
- Answers stream with page-level citations.
- An eval page that compares retrieval strategies side by side (faithfulness, relevance, context recall).
- Pluggable LLM: Claude, OpenAI, Azure OpenAI or local Ollama.
- Stack: Python, FastAPI, FAISS, BM25, sentence-transformers*, Angular, Docker. *See the embeddings decision below.

**Pitch:** "Most RAG demos stop at 'it answered something'. DocuMind shows which retrieval strategy actually works on your documents and keeps every answer tied to a source."

**Success means:**
- A public repo with a clear README.
- A live demo that works **with no API key**.
- It runs locally with `uv` + `npm` (Docker optional; Docker isn't installed on this Mac).
- CI is green.
- The portfolio card is flipped to "Live".

**Machine facts:**
- macOS. System Python is 3.9, so use **Python 3.12 via `uv`**.
- Node 22, Angular 22 via `npx @angular/cli`.
- No Docker, no `gh` CLI.
- Port 8000 is used by another local project ("sapna shringar"), so **run DocuMind's backend on 8002** locally. AgentHub uses 8001.

---

## 1. Key decisions (and why)

| Decision | Choice | Why |
|---|---|---|
| Embeddings | **fastembed** (ONNX) with `BAAI/bge-small-en-v1.5` (384-d) | Free hosting has 512 MB RAM. torch + sentence-transformers won't fit; fastembed is light and uses the same model family. Mention "sentence-transformers-compatible models via ONNX" in the README, and update the portfolio stack tag to "fastembed". |
| Reranker | fastembed `TextCrossEncoder` with `Xenova/ms-marco-MiniLM-L-6-v2` | A real cross-encoder at small size. |
| Dense index | `faiss-cpu`, `IndexFlatIP` on normalised vectors, one index per collection, saved to disk | Exact search is plenty at demo scale and keeps it simple. |
| Sparse | `bm25s` (fast, pure numpy), with simple tokenisation, lowercase and stopwords | |
| PDF parsing | **pypdf** (BSD) | Avoids PyMuPDF's AGPL licence. |
| Chunking | Page-aware recursive chunks (~800 chars, 120 overlap), split on headings and paragraphs first, then sentences | Every chunk keeps `doc_id`, `page`, `heading` and `char_start`, so citations are page-exact. "Semantic" here means structure-aware boundaries: be honest about that in the README. |
| Fusion | Reciprocal-rank fusion, k=60 | |
| LLM | Same provider resolution as AgentHub: anthropic → google → groq → openai/azure → ollama (only when forced) → **extractive** | Reuse the pattern from `~/Desktop/agenthub/backend/app/gateway/llm.py` (see section 7). |
| Keyless mode | **Extractive answers** | With no key, the answer is built from the top reranked sentences, each with its citation, and the UI labels it "Extractive mode (no LLM key)". Retrieval, rerank, citations and retrieval evals are all real. Honest, and it never breaks on quota. |
| Storage | SQLite (SQLModel) for documents, chunks and eval runs; files under `data/` | Zero setup. |
| Auth | **No login.** The demo library is shared and read-only. Uploads go to a private **workspace** keyed by a random id the browser keeps (`X-Workspace` header). | Simpler than AgentHub, and right for a "try it" tool. |

---

## 2. What a visitor sees
1. **Chat page**, with a library sidebar:
   - The sample library is preloaded with 3 fictional PDFs (below).
   - Pick documents, ask a question, and the answer **streams** with inline citations like `[1]` and `[2]`.
   - Clicking a citation opens the source passage with the page number and the matched text highlighted.
   - A collapsible **"How this was retrieved"** panel lists the dense hits, BM25 hits, fused order and reranked order, with scores. This is the differentiator: it makes retrieval visible.
   - A strategy switcher (Dense / BM25 / Hybrid / Hybrid + rerank) lets visitors compare live.
2. **Upload:** drag in a PDF (≤ 5 MB, ≤ 40 pages). You see an ingestion progress bar (parse → chunk → embed → index), then you can chat with it. It is private to that browser and deleted after 24 h.
3. **Evaluate page:**
   - Pick a dataset (the built-in gold Q&A set, or questions you add) and strategies, then run.
   - Get a side-by-side table and bar chart:
     - **Retrieval metrics (always real):** hit@k, recall@k, MRR, nDCG@k against gold pages.
     - **Answer metrics (when an LLM key exists):** faithfulness and answer relevance via LLM-as-judge, plus context precision.
     - **Keyless proxies** (clearly labelled): the share of answer sentences supported by a cited chunk (embedding similarity ≥ threshold), and question–answer embedding similarity.
   - Drill into any question to see per-strategy retrieved chunks and whether the gold page was hit.
   - Results are stored, so you can compare to previous runs.
4. **Documents page:** a list of documents with page and chunk counts and a chunk viewer, so you can see exactly how a PDF was chunked.

**Sample corpus** (fictional, written by us, so no copyright issues). Generate the PDFs with a script from markdown sources so they're reproducible:
- `northwind_employee_handbook.pdf` (~12 pages): leave, travel policy, expenses, security.
- `aquaflow_x200_user_manual.pdf` (~10 pages): a pump product with specs tables, troubleshooting and error codes. Good for exact BM25 matches.
- `northwind_fy25_annual_report.pdf` (~8 pages): numbers and narrative.
- `eval/gold.jsonl`: ~40 questions, each with `answer`, `doc` and `gold_pages`, mixing keyword-style questions (where BM25 wins) with paraphrased ones (where dense wins). This is what makes the comparison interesting.

---

## 3. Architecture
```
Angular app ──REST + SSE──► FastAPI (DocuMind)
                             ├─ ingest/: pypdf → page text → chunker → fastembed → FAISS + bm25s (per collection)
                             ├─ retrieval/: dense, sparse, rrf, rerank, strategies registry
                             ├─ answer/: prompt builder (numbered sources), LLM providers, extractive fallback, citation parser
                             ├─ eval/: datasets, retrieval metrics, judge metrics, runner (background task + SSE progress)
                             ├─ workspaces/: header-based isolation, TTL cleanup, upload limits
                             └─ SQLite (SQLModel) + data/indexes/{collection}/
```
**Collections:** `library` (the seeded sample docs, read-only) and `ws_<id>` (one per visitor workspace).

---

## 4. Repo layout (`~/Desktop/documind/`, its own git repo)
```
documind/
  backend/            uv project, Python 3.12, package = false
    pyproject.toml    fastapi[standard], sqlmodel, pydantic-settings, pypdf, fastembed, faiss-cpu, bm25s,
                      numpy, anthropic, openai, python-multipart; dev: pytest, pytest-asyncio, ruff
    app/
      main.py         lifespan: init DB, load models (warm), ingest sample library if missing, start TTL cleaner
      config.py       Settings (env prefix DOCUMIND_; provider keys via aliases ANTHROPIC_API_KEY etc.)
      db.py, models.py   Document, Chunk, EvalDataset, EvalItem, EvalRun, EvalResult
      ingest/  pdf.py, chunker.py, embed.py (fastembed singleton), indexer.py
      retrieval/  dense.py, sparse.py, fusion.py, rerank.py, strategies.py
      answer/  prompt.py, llm.py (providers + extractive), citations.py, routes.py (POST /chat → SSE)
      eval/  metrics.py, judge.py, runner.py, routes.py
      workspaces.py   dependency resolving the collection from the X-Workspace header; limits; cleanup
      documents/routes.py   upload, list, delete, chunks, page text
    tests/
  samples/  src/*.md, build_pdfs.py (fpdf2 or reportlab), *.pdf (committed), eval/gold.jsonl
  frontend/  Angular 22 (chat, documents, evaluate pages)
  docker-compose.yml, backend/Dockerfile, frontend/Dockerfile (nginx)
  render.yaml, frontend/vercel.json, .github/workflows/ci.yml
  README.md, LICENSE (MIT, Himanshu Chelani 2026), .env.example, .gitignore
```

---

## 5. API (main endpoints)
- `GET /health`, `GET /llm/provider` → `{provider, model, mode: live|extractive}`
- `GET /documents`, `POST /documents` (multipart upload → 202 + ingestion job id), `GET /documents/{id}`, `GET /documents/{id}/chunks`, `DELETE /documents/{id}` (workspace docs only)
- `GET /ingest/{job}/events`: SSE with stage and progress
- `POST /chat` `{question, doc_ids?, strategy, top_k}` → SSE events:
  1. `retrieval`: every stage's hits with scores
  2. `token`: answer deltas
  3. `citations`: `[n] → chunk id, doc, page, text`
  4. `done`: latency per stage, tokens, mode
- `POST /retrieve`: retrieval only, no LLM (used by the eval and debugging)
- `GET /eval/datasets`, `POST /eval/datasets/{id}/items`, `POST /eval/runs` `{dataset, strategies[], k}` → run id; `GET /eval/runs/{id}/events` (SSE progress); `GET /eval/runs`, `GET /eval/runs/{id}` (aggregate + per-item)

---

## 6. Safety and limits for the public demo
- **Upload checks:** the file must be a PDF by magic bytes, ≤ 5 MB and ≤ 40 pages. Strip nothing executable; never shell out.
- **Workspace limits:**
  - At most 5 documents per workspace, and a 24 h TTL cleanup (files, rows and index).
  - The library collection is immutable via the API.
- **Rate limits:** per IP, an in-memory sliding window (chat 20/min, upload 5/hour, eval runs 3/hour).
- **Prompt-injection hygiene:**
  - Retrieved text goes inside clearly delimited source blocks.
  - The system prompt says sources are data, not instructions.
  - The answer must cite only provided source numbers. The citation parser drops unknown numbers.
- **Rendering:** the frontend renders answers as escaped text with citation chips. Never use `innerHTML` with raw model output (reuse the escape-first renderer idea from AgentHub's `format.ts`).
- **CORS:** only the deployed frontend origin plus localhost.

---

## 7. Reuse from AgentHub (copy and adapt; don't import across repos)
- **LLM providers:** `~/Desktop/agenthub/backend/app/gateway/llm.py`.
  - `resolve_provider` order: anthropic → google → groq → openai → fallback.
  - `DEFAULT_MODELS`:
    - anthropic `claude-haiku-5-5`
    - google `gemini-2.5-flash`
    - groq `llama-3.3-70b-versatile`
    - openai `gpt-4o-mini`
    - ollama `llama3.2`
  - `BASE_URLS` for Gemini and Groq's OpenAI-compatible endpoints.
  - Anthropic SDK 1.13: `messages.create(..., output_config={"effort": "low"})`. Pass `system` via an extras dict only when set. For DocuMind use **streaming** (`client.messages.stream(...)`), with prompt caching on the system prompt.
  - OpenAI SDK 3.28: `chat.completions.create(stream=True)`.
  - Add **Azure OpenAI** (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`) since the portfolio card promises it.
  - Fallback on any provider error: the extractive answer.
- **Backend conventions:** `~/Desktop/agenthub/backend/pyproject.toml`.
  - Ruff (line 110; E, F, I, UP, B, SIM; ignore B008) and pytest `asyncio_mode=auto`.
  - `config.py` uses pydantic-settings with `env_file=".env"` and key aliases.
  - `db.py` `make_engine` uses StaticPool for in-memory test DBs.
- **SSE:** the AgentHub `runs/routes.py` pattern (async generator, keep-alive comments) on the server; `ApiService.streamRun` in `frontend/src/app/core/api.ts` (fetch-based SSE parser with abort) on the client. Use a POST variant for `/chat`.
- **Frontend shell and theme:** copy from `~/Desktop/agenthub/frontend/`:
  - `src/styles.css`: tokens bg #0c0c0f, surface #17171b, border #29292f, accent **(DocuMind uses #4fc3a1 teal as its accent** to match its portfolio card; keep violet #7d6bf5 as secondary), plus the Geist fonts and component classes card, chip, btn-primary, btn-ghost, input, table-head and table-cell.
  - `layout/shell.ts`, `layout/icon.ts`.
  - `core/format.ts` (escapeHtml, formatMs, timeAgo).
  - `public/env.js` + `scripts/write-env.mjs` (runtime API URL; env var `DOCUMIND_API_URL`).
  - `.postcssrc.json` (Tailwind v4 via `@tailwindcss/postcss`).
  - Angular 22 style: standalone, zoneless, signals, `httpResource`, `@if`/`@for`, `input()`, OnPush, files without the `.component` suffix.
  - Signal gotcha seen in AgentHub: inside an `effect` that triggers a load, wrap the call in `untracked()`.

---

## 8. Milestones
| # | Milestone | Done when |
|---|---|---|
| 0 | Save this plan to `~/Desktop/documind/PLAN.md`, `git init`, scaffold the uv backend + `ng new frontend` | Both run with hello-world endpoints |
| 1 | Sample corpus | 3 PDFs built from markdown + `gold.jsonl` (40 Q) committed |
| 2 | Ingestion | pypdf → chunker → fastembed → FAISS + bm25s persisted; library auto-ingested at startup; tests on chunk boundaries and page metadata |
| 3 | Retrieval | dense, sparse, RRF, rerank, strategy registry, `/retrieve`; tests check hybrid ≥ the single methods on gold |
| 4 | Answering | prompt + providers + extractive fallback + citation parser + SSE `/chat`; tests (scripted provider stub) |
| 5 | Workspaces + upload | limits, TTL cleanup, rate limits, ingestion SSE; tests (oversize/non-PDF rejected, isolation between workspaces) |
| 6 | Evaluation | metrics (hit@k, recall@k, MRR, nDCG), judge (LLM) and proxies (keyless), runner + SSE, stored runs; unit tests for the metrics with hand-computed values |
| 7 | Frontend | chat (streaming, citation popover, retrieval panel, strategy switch), documents and chunk viewer, upload, evaluate (table + bar chart + drill-down); mobile-checked |
| 8 | Packaging | Dockerfiles, compose, CI (ruff, pytest, ng build), `render.yaml` (pre-download models at build, `DOCUMIND_CORS_ORIGINS`), `vercel.json` (`node scripts/write-env.mjs && ng build`) |
| 9 | Docs | README: pitch, screenshots/GIF, mermaid diagram, "which strategy wins on which question type" findings from the eval, quickstart, deploy steps |
| 10 | Portfolio update (in `~/Desktop/himanshu website/site`) | `projects.ts` documind → status "Live", demo URL, stack tag fastembed; FAQ projects answer updated |

Show progress to the user after milestones 3, 6 and 7.

**Memory note:** fastembed downloads models to a cache on first use (~150 MB in total). Render's build step should run a warm-up script so cold starts don't download. Keep RAM under about 450 MB: load the embedder and reranker once as singletons.

---

## 9. Verification
- **pytest:**
  - The chunker keeps page numbers and the overlap.
  - RRF ordering is correct on a toy example.
  - Metrics match hand-computed values.
  - On `gold.jsonl`, hybrid + rerank recall@5 ≥ dense and ≥ BM25 (assert, and print the table).
  - Upload rejects non-PDF and oversize files.
  - Workspace A can't see workspace B's docs.
  - `/chat` in extractive mode returns citations that all point at retrieved chunks.
- **Manual / headless Chrome** (CDP screenshot script, like the one used for AgentHub, at 1440 and 390 width):
  - Ask "What is the per-diem for international travel?" and get a streamed answer with a page citation that opens the right passage.
  - Upload a PDF and chat with it.
  - Run the eval across 4 strategies and see the table and chart.
  - No console errors and no horizontal overflow.
- **Keyless check:** with no keys set, everything works and the badge shows "Extractive mode". With `GROQ_API_KEY` or `GOOGLE_API_KEY` in `backend/.env`, answers are generated and judge metrics appear.
- **CI** passes on GitHub. A fresh clone works by following the README quickstart.

## 10. Needed from the user
- A GitHub repo `documind` (the user pushes; `gh` isn't installed).
- Render and Vercel deploys using the provided configs.
- Optional: a free Gemini or Groq key for live answers and judge metrics.
