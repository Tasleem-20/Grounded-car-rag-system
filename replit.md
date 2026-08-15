# Grounded RAG

A modular Streamlit Basic Retrieval-Augmented Generation app that indexes PDF/TXT
documents with OpenAI embeddings and answers questions using a local FAISS index.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `streamlit run app.py` — run the Basic RAG interface locally
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env for RAG: `OPENAI_API_KEY` — OpenAI embeddings and generation credential
- Required env for the preconfigured API server: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `app.py` — Streamlit interface and application orchestration
- `src/document_loader.py` — PDF/TXT loading and text cleaning
- `src/chunker.py` — overlapping chunks with source offsets
- `src/embeddings.py` — OpenAI embedding client
- `src/vector_store.py` — persisted local FAISS index and metadata
- `src/retriever.py` — question embedding and top-k retrieval
- `src/generator.py` — grounded OpenAI answer generation
- `data/documents/` — locally saved uploaded documents
- `indexes/` — generated FAISS index and traceability metadata

## Architecture decisions

- Retrieval uses normalized FAISS inner-product search, which produces cosine-similarity scores.
- Chunk metadata is persisted beside the index so every result can be traced to a file and character range.
- The app intentionally stops at basic retrieval and grounded generation; adaptive/self-checking research modules are deferred.

## Product

Users can upload PDF/TXT references, process them into a local searchable index,
ask questions, and inspect the exact retrieved evidence and similarity scores
used for each answer.

## User preferences

- Keep the Basic RAG foundation simple and modular so later research components can be added independently.

## Gotchas

- A valid OpenAI API key is required before document processing or question answering.
- Scanned PDFs without an OCR text layer are rejected as empty documents.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
