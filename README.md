# RAG (Retrieval-Augmented Generation) — Complete Guide with Node.js & Express

> Primary focus: how RAG works internally, every component explained from first principles, then a full production Express implementation. Not just "call an API" — understand what happens inside every step.

---

## Table of Contents

1. [What is RAG and Why Does it Exist?](#1-what-is-rag-and-why-does-it-exist)
2. [The Problem RAG Solves](#2-the-problem-rag-solves)
3. [RAG Architecture — The Full Picture](#3-rag-architecture--the-full-picture)
4. [Component 1 — Document Loading & Parsing](#4-component-1--document-loading--parsing)
5. [Component 2 — Chunking (Text Splitting)](#5-component-2--chunking-text-splitting)
6. [Component 3 — Embeddings](#6-component-3--embeddings)
7. [Component 4 — Vector Database](#7-component-4--vector-database)
8. [Component 5 — Retrieval](#8-component-5--retrieval)
9. [Component 6 — Reranking](#9-component-6--reranking)
10. [Component 7 — The Prompt (Context Injection)](#10-component-7--the-prompt-context-injection)
11. [Component 8 — The LLM (Generation)](#11-component-8--the-llm-generation)
12. [How Similarity Search Works Internally](#12-how-similarity-search-works-internally)
13. [RAG Failure Modes & How to Fix Them](#13-rag-failure-modes--how-to-fix-them)
14. [Advanced RAG Patterns](#14-advanced-rag-patterns)
15. [Project Setup — Express + RAG](#15-project-setup--express--rag)
16. [Ingestion Pipeline — Node.js](#16-ingestion-pipeline--nodejs)
17. [Query Pipeline — Express API](#17-query-pipeline--express-api)
18. [Full REST API Implementation](#18-full-rest-api-implementation)
19. [Streaming Responses](#19-streaming-responses)
20. [Evaluation — How to Know if Your RAG is Good](#20-evaluation--how-to-know-if-your-rag-is-good)
21. [Production Checklist](#21-production-checklist)

---

## 1. What is RAG and Why Does it Exist?

**RAG** (Retrieval-Augmented Generation) is a pattern that augments an LLM's answer with relevant documents retrieved from a knowledge base at query time.

A Large Language Model is trained on data up to a cutoff date. It has no knowledge of:
- Your company's private documents
- Real-time data (prices, inventory, news)
- Documents created after its training cutoff
- Anything not in its training set

RAG solves this by **retrieving** the relevant information from *your* data store and **stuffing it into the prompt** before asking the LLM to answer.

```
WITHOUT RAG:
  User: "What is our refund policy for orders over $200?"
  LLM:  "I don't have access to your specific refund policy..." ❌

WITH RAG:
  User: "What is our refund policy for orders over $200?"
  System: [searches your policy docs] → finds the relevant section
  LLM:  "According to your policy document: orders over $200 are eligible
          for a full refund within 30 days with original receipt." ✅
```

---

## 2. The Problem RAG Solves

### Why not just fine-tune the LLM on your data?

| Approach | Fine-tuning | RAG |
|---|---|---|
| Data freshness | Stale — retrain needed for updates | Live — update the vector DB |
| Cost | Very expensive ($$$) | Cheap — just embed new docs |
| Transparency | Black box | Citable sources |
| Data security | Training data may leak | Data stays in your DB |
| Setup time | Days/weeks | Hours |
| Works for private docs | ❌ (data goes to model provider) | ✅ (docs stay in your infra) |

### Why not just pass all documents in the prompt?

LLMs have a **context window limit** (e.g. 128k tokens for GPT-4o). More importantly:
- Passing 1,000 documents = slow and expensive
- LLMs suffer from "lost in the middle" — they ignore content in the middle of long prompts
- You only need the 3–5 most relevant chunks, not everything

---

## 3. RAG Architecture — The Full Picture

RAG has two completely separate pipelines:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INGESTION PIPELINE  (runs once, or on document updates)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Raw Documents
  (PDF, DOCX, HTML, DB rows, CSV...)
        │
        ▼
  ┌─────────────┐
  │   Loader    │  Reads & parses raw files → plain text
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   Chunker   │  Splits text into overlapping chunks
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │ Embedding Model  │  Converts each chunk → vector [0.12, -0.87, ...]
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Vector Database │  Stores vectors + original text + metadata
  └──────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 QUERY PIPELINE  (runs on every user question)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  User Question: "What is the refund policy?"
        │
        ▼
  ┌──────────────────┐
  │ Embedding Model  │  Same model → query vector [0.15, -0.91, ...]
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Vector Database │  Similarity search → top-k most similar chunks
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │    Reranker      │  (optional) Re-scores retrieved chunks for precision
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────────────────────────┐
  │  Prompt Builder                      │
  │  "Context: <chunk1> <chunk2>...      │
  │   Question: What is the refund...?"  │
  └────────┬─────────────────────────────┘
           │
           ▼
  ┌──────────────────┐
  │       LLM        │  Reads context + question → generates answer
  └────────┬─────────┘
           │
           ▼
  Answer + Sources → User
```

---

## 4. Component 1 — Document Loading & Parsing

The first step is turning raw files into plain text. This sounds trivial but is where most RAG systems first fail — garbage in, garbage out.

### What loaders do

A **loader** reads a file format and extracts clean, structured text, preserving as much semantic meaning as possible.

```typescript
// src/ingestion/loaders/types.ts

export interface LoadedDocument {
  content:  string;                     // the extracted plain text
  metadata: DocumentMetadata;           // who, what, when — used for filtering later
}

export interface DocumentMetadata {
  source:      string;                  // file path, URL, or DB row ID
  type:        'pdf' | 'html' | 'txt' | 'docx' | 'csv' | 'markdown';
  title?:      string;
  author?:     string;
  createdAt?:  string;
  pageNumber?: number;                  // useful for multi-page PDFs
  url?:        string;
  tags?:       string[];
}
```

```typescript
// src/ingestion/loaders/pdf-loader.ts
/**
 * PDF LOADER — Why PDFs are hard:
 *
 * PDFs don't store text as a flat string. They store drawing commands:
 * "draw glyph 'H' at position (72, 680)". A PDF parser must reconstruct
 * reading order from spatial positions — and often gets it wrong for
 * multi-column layouts, tables, or scanned images (which have NO text at all).
 *
 * For scanned PDFs, you need OCR (Optical Character Recognition) before loading.
 */
import pdfParse from 'pdf-parse';
import fs from 'fs/promises';
import path from 'path';
import type { LoadedDocument } from './types';

export async function loadPDF(filePath: string): Promise<LoadedDocument[]> {
  const buffer = await fs.readFile(filePath);
  const data   = await pdfParse(buffer);

  // pdf-parse gives us all text concatenated; split by page for better metadata
  const pages  = data.text.split(/\f/).filter((p) => p.trim().length > 0);

  return pages.map((pageText, index) => ({
    content:  cleanText(pageText),
    metadata: {
      source:     filePath,
      type:       'pdf',
      title:      path.basename(filePath, '.pdf'),
      pageNumber: index + 1,
    },
  }));
}

// src/ingestion/loaders/html-loader.ts
/**
 * HTML LOADER — Why HTML is hard:
 *
 * HTML mixes actual content with nav bars, footers, ads, cookie banners.
 * Passing raw HTML to the embedding model confuses it — "Accept Cookies" is
 * not part of your article content.
 *
 * JSDOM + custom selectors extracts only the article body.
 */
import { JSDOM } from 'jsdom';
import type { LoadedDocument } from './types';

export async function loadHTML(url: string): Promise<LoadedDocument> {
  const response = await fetch(url);
  const html     = await response.text();
  const dom      = new JSDOM(html);
  const document  = dom.window.document;

  // Remove noise: nav, footer, ads, scripts, styles
  const noise = ['nav', 'footer', 'header', 'aside', 'script', 'style', '.ad', '.cookie-banner'];
  noise.forEach((selector) => {
    document.querySelectorAll(selector).forEach((el) => el.remove());
  });

  // Extract main content — try common content container selectors
  const main = document.querySelector('main, article, [role="main"], .content, .post-body')
            ?? document.body;

  return {
    content:  cleanText(main.textContent ?? ''),
    metadata: {
      source: url,
      type:   'html',
      title:  document.querySelector('title')?.textContent?.trim(),
      url,
    },
  };
}

// src/ingestion/loaders/csv-loader.ts
/**
 * CSV LOADER — Structured data needs a different approach:
 *
 * Don't just join all cells into one blob. Convert each row into a
 * human-readable sentence so the embedding model understands the relationships.
 *
 * Row: { product: "Widget A", price: 29.99, stock: 142 }
 * Text: "Widget A has a price of $29.99 and a stock of 142 units."
 *
 * This is called "row-to-text templating" — the LLM can now answer
 * "what is the price of Widget A" by finding this chunk semantically.
 */
import { parse } from 'csv-parse/sync';
import fs from 'fs/promises';
import type { LoadedDocument } from './types';

export async function loadCSV(
  filePath: string,
  rowToText: (row: Record<string, string>, index: number) => string
): Promise<LoadedDocument[]> {
  const content = await fs.readFile(filePath, 'utf-8');
  const rows    = parse(content, { columns: true, skip_empty_lines: true });

  return rows.map((row: Record<string, string>, i: number) => ({
    content:  rowToText(row, i),
    metadata: { source: filePath, type: 'csv' },
  }));
}

// ─── Shared text cleaner ───────────────────────────────────────────────────────
function cleanText(text: string): string {
  return text
    .replace(/\r\n/g, '\n')           // normalize line endings
    .replace(/[ \t]+/g, ' ')          // collapse multiple spaces/tabs
    .replace(/\n{3,}/g, '\n\n')       // max two consecutive newlines
    .replace(/[^\x20-\x7E\n]/g, '')   // remove non-printable characters
    .trim();
}
```

---

## 5. Component 2 — Chunking (Text Splitting)

Chunking is the most underestimated and most impactful parameter in RAG. The wrong chunking strategy breaks retrieval more than any other factor.

### Why chunk at all?

1. **Context window limits** — embedding models have token limits (typically 512–8192 tokens). A 50-page PDF can't be embedded as one unit.
2. **Retrieval precision** — if a chunk is too large, one relevant sentence is buried in 2,000 tokens of noise, diluting its semantic signal in the embedding.
3. **Answer quality** — if a chunk is too small, the LLM gets a fragment without enough context to form a useful answer.

### The Chunking Trade-off

```
Chunk too SMALL (e.g. 1 sentence):
  ✅ High retrieval precision — finds the exact sentence
  ❌ Low context — LLM can't answer from just one sentence
  ❌ More chunks = more storage + slower search

Chunk too LARGE (e.g. 5 pages):
  ✅ Rich context — LLM has everything it needs
  ❌ Low retrieval precision — relevant info drowns in noise
  ❌ Few chunks = good info missed if it's on page 3 of the chunk

Sweet spot: 256–512 tokens with 10–20% overlap
```

### Why Overlap Matters

```
Document text (simplified):
"...Refunds are processed within 5 business days.
 For orders over $200, a manager must approve the refund.
 Contact support@example.com to initiate..."

Chunk 1 (tokens 0-256):   "...Refunds are processed within 5 business days.
                             For orders over $200, a manager must approve..."
Chunk 2 (tokens 206-462): "For orders over $200, a manager must approve the refund.
                             Contact support@example.com to initiate..."

The sentence "For orders over $200, a manager must approve the refund"
appears in BOTH chunks (overlap zone).

Without overlap: a question spanning the chunk boundary misses it entirely.
With overlap: at least one chunk contains the full sentence in context.
```

### Chunking Strategies

```typescript
// src/ingestion/chunkers/chunker.ts

export interface Chunk {
  content:   string;
  metadata:  Record<string, unknown>;
  chunkIndex: number;
  tokenCount: number;
}

// ─── STRATEGY 1: Fixed-Size Chunking (simplest, often good enough) ─────────────
/**
 * Splits text every N characters with M characters of overlap.
 * Problem: can cut in the middle of a sentence.
 * Use when: you need simplicity and consistency over semantic boundaries.
 */
export function fixedSizeChunker(
  content:     string,
  metadata:    Record<string, unknown>,
  chunkSize:   number = 1000,   // characters (≈ 250 tokens for English)
  chunkOverlap: number = 200
): Chunk[] {
  const chunks: Chunk[] = [];
  let start = 0;
  let index = 0;

  while (start < content.length) {
    const end  = Math.min(start + chunkSize, content.length);
    const text = content.slice(start, end);

    chunks.push({
      content:    text,
      metadata:   { ...metadata, chunkIndex: index },
      chunkIndex: index++,
      tokenCount: estimateTokens(text),
    });

    start = end - chunkOverlap;  // move back by overlap amount
    if (start >= content.length) break;
  }

  return chunks;
}

// ─── STRATEGY 2: Recursive Sentence-Aware Chunking (recommended default) ────────
/**
 * Tries to split at paragraph → sentence → word → character boundaries,
 * in that priority order. This preserves semantic units.
 *
 * Algorithm:
 * 1. Try to split by "\n\n" (paragraph). If all pieces fit in chunkSize → done.
 * 2. If a piece is still too big, split it by "\n" (line).
 * 3. If still too big, split by ". " (sentence).
 * 4. If still too big, split by " " (word).
 * 5. If still too big, split character by character (last resort).
 */
export function recursiveChunker(
  content:      string,
  metadata:     Record<string, unknown>,
  chunkSize:    number = 1000,
  chunkOverlap: number = 200
): Chunk[] {
  const separators = ['\n\n', '\n', '. ', ' ', ''];
  const rawChunks  = splitRecursively(content, separators, chunkSize);

  // Merge small chunks and apply overlap
  return mergeAndOverlap(rawChunks, metadata, chunkSize, chunkOverlap);
}

function splitRecursively(
  text:       string,
  separators: string[],
  maxSize:    number
): string[] {
  const [sep, ...remaining] = separators;

  if (sep === undefined || text.length <= maxSize) return [text];

  const pieces = text.split(sep).filter((p) => p.trim().length > 0);
  const result: string[] = [];

  for (const piece of pieces) {
    if (piece.length <= maxSize) {
      result.push(piece);
    } else {
      // This piece is still too big — go deeper with the next separator
      result.push(...splitRecursively(piece, remaining, maxSize));
    }
  }

  return result;
}

function mergeAndOverlap(
  pieces:       string[],
  metadata:     Record<string, unknown>,
  chunkSize:    number,
  chunkOverlap: number
): Chunk[] {
  const chunks:  Chunk[] = [];
  let current    = '';
  let index      = 0;

  for (const piece of pieces) {
    if ((current + piece).length > chunkSize && current.length > 0) {
      chunks.push({ content: current.trim(), metadata: { ...metadata }, chunkIndex: index++, tokenCount: estimateTokens(current) });
      // Start next chunk with overlap from previous
      const words  = current.split(' ');
      const overlap = words.slice(-Math.floor(chunkOverlap / 5)).join(' ');
      current = overlap + ' ' + piece;
    } else {
      current = current ? current + ' ' + piece : piece;
    }
  }

  if (current.trim()) {
    chunks.push({ content: current.trim(), metadata: { ...metadata }, chunkIndex: index, tokenCount: estimateTokens(current) });
  }

  return chunks;
}

// ─── STRATEGY 3: Semantic Chunking (best quality, most expensive) ────────────────
/**
 * Computes embeddings for every sentence, then finds "breakpoints" where
 * semantic similarity drops sharply between adjacent sentences.
 *
 * Intuition: sentences within a paragraph are semantically similar.
 * When the topic changes, similarity between consecutive sentence embeddings drops.
 * Those drop points become chunk boundaries.
 *
 * Cost: requires N embedding calls during ingestion (one per sentence).
 * Benefit: chunks always contain topically coherent content.
 */
export async function semanticChunker(
  content:   string,
  metadata:  Record<string, unknown>,
  embedFn:   (text: string) => Promise<number[]>,
  threshold: number = 0.3   // cosine distance threshold for breakpoints
): Promise<Chunk[]> {
  // Split into sentences
  const sentences = content
    .split(/(?<=[.!?])\s+/)
    .filter((s) => s.trim().length > 20);

  if (sentences.length < 3) {
    return [{ content, metadata, chunkIndex: 0, tokenCount: estimateTokens(content) }];
  }

  // Embed all sentences
  const embeddings = await Promise.all(sentences.map(embedFn));

  // Find breakpoints where similarity drops
  const breakpoints: number[] = [];
  for (let i = 0; i < embeddings.length - 1; i++) {
    const similarity = cosineSimilarity(embeddings[i], embeddings[i + 1]);
    if (similarity < (1 - threshold)) {
      breakpoints.push(i + 1); // break AFTER sentence i
    }
  }

  // Build chunks from breakpoints
  const chunks: Chunk[] = [];
  let start = 0;

  for (const bp of [...breakpoints, sentences.length]) {
    const chunkText = sentences.slice(start, bp).join(' ');
    chunks.push({
      content:    chunkText.trim(),
      metadata:   { ...metadata },
      chunkIndex: chunks.length,
      tokenCount: estimateTokens(chunkText),
    });
    start = bp;
  }

  return chunks;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function estimateTokens(text: string): number {
  // Rough estimate: 1 token ≈ 4 characters in English
  return Math.ceil(text.length / 4);
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dot    = a.reduce((sum, v, i) => sum + v * b[i], 0);
  const normA  = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
  const normB  = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
  return normA && normB ? dot / (normA * normB) : 0;
}
```

---

## 6. Component 3 — Embeddings

An **embedding** is a numerical representation of text meaning. It converts a string of arbitrary length into a fixed-size vector of floating-point numbers (e.g. 1536 numbers for OpenAI's `text-embedding-3-small`).

### What Embeddings Actually Represent

The embedding model is trained to place semantically similar text *near each other* in vector space. The actual numbers are meaningless to humans — only the distances between vectors matter.

```
"I need to cancel my order"      → [0.12, -0.87, 0.34, 0.91, ...]
"How do I return a purchase?"    → [0.14, -0.85, 0.37, 0.89, ...]  ← similar direction
"What is the weather in Paris?"  → [-0.73, 0.21, -0.55, 0.11, ...] ← very different

These three sentences map to points in 1536-dimensional space.
The first two are "close" → high cosine similarity → retrieved together.
The third is "far away" → low similarity → not retrieved for a refund question.
```

### Why the SAME Model Must Be Used for Ingestion and Query

```
If you embed documents with model A (1536 dimensions)
and embed the query with model B (768 dimensions):
  → Dimensions don't match → can't compare → crash

If you embed with model A at ingestion
and model B at query time (same dimensions, different training):
  → Vectors are in completely different "languages"
  → Similarity scores are meaningless → terrible retrieval

Rule: ingestion embedding model = query embedding model. Always.
```

```typescript
// src/embeddings/embedding-service.ts
/**
 * Wraps the embedding API with:
 *  - Batching    — send multiple chunks in one API call (cheaper + faster)
 *  - Caching     — don't re-embed the same text twice
 *  - Rate limiting — OpenAI has requests-per-minute limits
 *  - Retry       — transient network errors
 */
import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// In-memory cache — use Redis in production
const embeddingCache = new Map<string, number[]>();

const EMBEDDING_MODEL = 'text-embedding-3-small';
const EMBEDDING_DIMS  = 1536;
const BATCH_SIZE      = 100;   // OpenAI allows up to 2048 inputs per call

export async function embedText(text: string): Promise<number[]> {
  const cacheKey = `embed:${EMBEDDING_MODEL}:${hashText(text)}`;

  if (embeddingCache.has(cacheKey)) {
    return embeddingCache.get(cacheKey)!;
  }

  const response = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: text.replace(/\n/g, ' '), // newlines degrade embedding quality slightly
  });

  const vector = response.data[0].embedding;
  embeddingCache.set(cacheKey, vector);
  return vector;
}

export async function embedBatch(texts: string[]): Promise<number[][]> {
  const results:   number[][] = new Array(texts.length);
  const toEmbed:   { text: string; index: number }[] = [];

  // Check cache first
  for (let i = 0; i < texts.length; i++) {
    const cached = embeddingCache.get(`embed:${EMBEDDING_MODEL}:${hashText(texts[i])}`);
    if (cached) {
      results[i] = cached;
    } else {
      toEmbed.push({ text: texts[i], index: i });
    }
  }

  // Batch API calls for uncached texts
  for (let i = 0; i < toEmbed.length; i += BATCH_SIZE) {
    const batch    = toEmbed.slice(i, i + BATCH_SIZE);
    const response = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: batch.map((b) => b.text.replace(/\n/g, ' ')),
    });

    for (let j = 0; j < batch.length; j++) {
      const vector = response.data[j].embedding;
      results[batch[j].index] = vector;
      embeddingCache.set(`embed:${EMBEDDING_MODEL}:${hashText(batch[j].text)}`, vector);
    }

    // Rate limit: pause between batches
    if (i + BATCH_SIZE < toEmbed.length) {
      await sleep(200);
    }
  }

  return results;
}

function hashText(text: string): string {
  // Simple hash for cache key — use a proper hash in production
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = (hash * 31 + text.charCodeAt(i)) & 0x7fffffff;
  }
  return hash.toString(36);
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

export { EMBEDDING_MODEL, EMBEDDING_DIMS };
```

---

## 7. Component 4 — Vector Database

A **vector database** stores your chunk embeddings and their associated text/metadata, and provides fast similarity search (finding the closest vectors to a query vector).

### What's Actually Stored

```
Each stored record contains:
┌──────────────────────────────────────────────────────────┐
│  id:        "chunk_8f3a2b"                               │
│  vector:    [0.12, -0.87, 0.34, ...]  (1536 floats)      │
│  content:   "Refunds over $200 require manager approval" │
│  metadata:  {                                            │
│    source:    "refund-policy.pdf",                       │
│    pageNumber: 3,                                        │
│    type:      "pdf",                                     │
│    createdAt: "2024-01-15"                               │
│  }                                                       │
└──────────────────────────────────────────────────────────┘
```

### How Indexes Work Internally (ANN — Approximate Nearest Neighbor)

Finding the exact closest vector in a set of 1 million 1536-dimensional vectors by brute force would take ~1.5 billion multiplications per query. That's too slow.

Vector DBs use **Approximate Nearest Neighbor (ANN)** indexes that trade a tiny bit of accuracy for massive speed gains:

```
HNSW (Hierarchical Navigable Small World) — used by Pinecone, pgvector, Weaviate:

  Imagine a multilevel graph where each node is a vector.
  Level 2 (sparse):   A ─────────── E ─────────── J
  Level 1 (medium):   A ── C ── E ── G ── I ── J
  Level 0 (dense):    A ─ B ─ C ─ D ─ E ─ F ─ G ─ H ─ I ─ J

  Query process:
  1. Enter at Level 2 (few nodes — fast coarse navigation)
  2. Find closest node, move to Level 1 (more nodes — finer search)
  3. Find closest node, move to Level 0 (all nodes — precise local search)

  Result: finds approximate nearest neighbors in O(log N) instead of O(N)
  Speed: 1000x faster than brute force at 99% accuracy

IVF (Inverted File Index) — used by Faiss:
  Groups vectors into clusters (like a k-means). At query time,
  only search the nearest 10-20 clusters instead of all vectors.
```

```typescript
// src/vectordb/pgvector-store.ts
/**
 * USING PGVECTOR — Postgres extension for vector storage.
 *
 * Why pgvector over a dedicated vector DB?
 *   - You probably already have Postgres
 *   - Combine vector search with SQL filters (e.g. "find similar docs from 2024")
 *   - ACID transactions — ingestion is atomic with your business data
 *   - No new infrastructure to operate
 *
 * Tradeoff: not as fast as Pinecone at very large scale (>10M vectors)
 * For most applications (<5M chunks), pgvector is excellent.
 */

import { Pool } from 'pg';
import type { Chunk } from '../ingestion/chunkers/chunker';

const pool = new Pool({ connectionString: process.env.DATABASE_URL });

// ─── Setup ────────────────────────────────────────────────────────────────────
export async function setupVectorTable(): Promise<void> {
  await pool.query(`
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS document_chunks (
      id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      content     TEXT NOT NULL,
      embedding   vector(1536),          -- stores the 1536-float vector
      metadata    JSONB NOT NULL DEFAULT '{}',
      source      TEXT,
      created_at  TIMESTAMPTZ DEFAULT NOW()
    );

    -- HNSW index for fast approximate nearest-neighbor search.
    -- m: number of connections per node (higher = better accuracy, more memory)
    -- ef_construction: search depth during index build (higher = better quality)
    CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
      ON document_chunks
      USING hnsw (embedding vector_cosine_ops)
      WITH (m = 16, ef_construction = 64);

    -- Regular index on source for metadata filtering
    CREATE INDEX IF NOT EXISTS chunks_source_idx ON document_chunks (source);
    CREATE INDEX IF NOT EXISTS chunks_metadata_idx ON document_chunks USING GIN (metadata);
  `);
  console.log('[VectorDB] Table and indexes ready');
}

// ─── Upsert chunks ────────────────────────────────────────────────────────────
export async function upsertChunks(
  chunks:     Chunk[],
  embeddings: number[][]
): Promise<void> {
  const client = await pool.connect();

  try {
    await client.query('BEGIN');

    for (let i = 0; i < chunks.length; i++) {
      const chunk     = chunks[i];
      const embedding = embeddings[i];

      // Convert the JS number[] to pgvector's '[0.12,-0.87,...]' string format
      const vectorStr = `[${embedding.join(',')}]`;

      await client.query(`
        INSERT INTO document_chunks (content, embedding, metadata, source)
        VALUES ($1, $2::vector, $3, $4)
        ON CONFLICT DO NOTHING
      `, [
        chunk.content,
        vectorStr,
        JSON.stringify(chunk.metadata),
        chunk.metadata.source as string,
      ]);
    }

    await client.query('COMMIT');
    console.log(`[VectorDB] Upserted ${chunks.length} chunks`);
  } catch (err) {
    await client.query('ROLLBACK');
    throw err;
  } finally {
    client.release();
  }
}

// ─── Delete by source ─────────────────────────────────────────────────────────
export async function deleteBySource(source: string): Promise<void> {
  await pool.query('DELETE FROM document_chunks WHERE source = $1', [source]);
  console.log(`[VectorDB] Deleted all chunks for source: ${source}`);
}
```

---

## 8. Component 5 — Retrieval

Retrieval is the query-time step: embed the user's question, search the vector DB for the most similar chunks, and return them.

### The Three Retrieval Strategies

```typescript
// src/retrieval/retriever.ts

import { embedText } from '../embeddings/embedding-service';
import { pool } from '../vectordb/pgvector-store';

export interface RetrievedChunk {
  id:         string;
  content:    string;
  metadata:   Record<string, unknown>;
  score:      number;    // cosine similarity — 1.0 = identical, 0.0 = unrelated
  source:     string;
}

// ─── STRATEGY 1: Dense Retrieval (default — semantic similarity) ──────────────
/**
 * Converts the query to a vector, then finds the k chunks with the
 * highest cosine similarity. This is "semantic search" — it finds
 * conceptually similar content even if the exact words differ.
 *
 * Works: "cancel my purchase" finds "return policy" even with no word overlap.
 * Fails: exact product codes, rare technical terms, proper nouns (see BM25 below).
 */
export async function denseRetrieval(
  query:        string,
  topK:         number = 5,
  filter?:      { source?: string; tags?: string[] }
): Promise<RetrievedChunk[]> {
  const queryVector = await embedText(query);
  const vectorStr   = `[${queryVector.join(',')}]`;

  // Build optional WHERE clause from metadata filters
  // This is where pgvector's SQL integration shines — combine vector search with SQL
  let whereClause  = '';
  const params: (string | number)[] = [vectorStr, topK];

  if (filter?.source) {
    params.push(filter.source);
    whereClause += ` AND source = $${params.length}`;
  }
  if (filter?.tags?.length) {
    params.push(JSON.stringify(filter.tags));
    whereClause += ` AND metadata->'tags' ?| $${params.length}::text[]`;
  }

  // <=> is pgvector's cosine DISTANCE operator (0 = identical, 2 = opposite)
  // We convert to similarity: 1 - distance
  const result = await pool.query(`
    SELECT
      id,
      content,
      metadata,
      source,
      1 - (embedding <=> $1::vector) AS score
    FROM document_chunks
    WHERE 1=1 ${whereClause}
    ORDER BY embedding <=> $1::vector
    LIMIT $2
  `, params);

  return result.rows.map((row) => ({
    id:       row.id,
    content:  row.content,
    metadata: row.metadata,
    score:    parseFloat(row.score),
    source:   row.source,
  }));
}

// ─── STRATEGY 2: Sparse Retrieval / BM25 (keyword matching) ──────────────────
/**
 * BM25 is the algorithm behind traditional search engines (Elasticsearch).
 * It counts keyword frequency and inverse document frequency (TF-IDF variant).
 *
 * Works: exact terms, model numbers, names, rare technical jargon.
 * Fails: synonyms, paraphrases ("cancel" vs "terminate").
 *
 * PostgreSQL has built-in full-text search (ts_vector + ts_query) which
 * approximates BM25. For true BM25, use Elasticsearch or the pg_bm25 extension.
 */
export async function sparseRetrieval(
  query: string,
  topK:  number = 5
): Promise<RetrievedChunk[]> {
  const result = await pool.query(`
    SELECT
      id,
      content,
      metadata,
      source,
      ts_rank(to_tsvector('english', content), query) AS score
    FROM document_chunks,
         plainto_tsquery('english', $1) query
    WHERE to_tsvector('english', content) @@ query
    ORDER BY score DESC
    LIMIT $2
  `, [query, topK]);

  return result.rows.map((row) => ({
    id:       row.id,
    content:  row.content,
    metadata: row.metadata,
    score:    parseFloat(row.score),
    source:   row.source,
  }));
}

// ─── STRATEGY 3: Hybrid Retrieval (dense + sparse combined) — BEST QUALITY ────
/**
 * Combines semantic similarity (dense) with keyword matching (sparse).
 * Uses Reciprocal Rank Fusion (RRF) to merge the two ranked lists.
 *
 * RRF formula: score(d) = Σ 1 / (k + rank(d))
 * where k=60 is a smoothing constant and rank is the 1-based position in each list.
 *
 * Why RRF instead of just averaging scores?
 *   Dense scores are cosine similarities (0–1).
 *   BM25 scores are on a completely different scale (0–10+).
 *   You can't average them meaningfully. RRF uses rank (position) which is scale-free.
 */
export async function hybridRetrieval(
  query:  string,
  topK:   number = 5,
  filter?: { source?: string }
): Promise<RetrievedChunk[]> {
  const [denseResults, sparseResults] = await Promise.all([
    denseRetrieval(query,  topK * 2, filter),  // get more than needed, RRF will trim
    sparseRetrieval(query, topK * 2),
  ]);

  // Reciprocal Rank Fusion
  const K = 60;
  const scores = new Map<string, { chunk: RetrievedChunk; score: number }>();

  function addRRFScore(results: RetrievedChunk[]) {
    results.forEach((chunk, rank) => {
      const current = scores.get(chunk.id) ?? { chunk, score: 0 };
      current.score += 1 / (K + rank + 1);
      scores.set(chunk.id, current);
    });
  }

  addRRFScore(denseResults);
  addRRFScore(sparseResults);

  return [...scores.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ chunk, score }) => ({ ...chunk, score }));
}
```

---

## 9. Component 6 — Reranking

Retrieval (vector search) is fast but imprecise. A **reranker** is a slower but much more accurate model that re-scores the retrieved chunks.

### Why Reranking Exists

```
The gap between retrieval and reranking:

Retrieval model:  Trained to produce vectors with fast ANN search.
                  Sees the query and ONE document independently.
                  "Is this document about refunds? → yes/no" (bi-encoder)

Reranker:         A cross-encoder — sees query + document TOGETHER.
                  Can model how well the document ANSWERS the specific query.
                  "Given this query, does this document actually answer it?" (0.0-1.0)

Result: retrieval gets you the right neighborhood,
        reranking finds the exact right house.
```

```typescript
// src/retrieval/reranker.ts
/**
 * Cross-encoder reranker using Cohere's rerank API.
 *
 * Typical pipeline:
 *   1. Dense retrieval: fetch top 20 candidates (fast, ANN)
 *   2. Reranker: score all 20 against the query (slower but precise)
 *   3. Take top 5 reranked results → send to LLM
 *
 * Cost: reranking 20 docs is still much cheaper than embedding 20 docs
 * AND sending all 20 to the LLM.
 */
import { CohereClient } from 'cohere-ai';
import type { RetrievedChunk } from './retriever';

const cohere = new CohereClient({ token: process.env.COHERE_API_KEY! });

export async function rerankChunks(
  query:  string,
  chunks: RetrievedChunk[],
  topN:   number = 5
): Promise<RetrievedChunk[]> {
  if (chunks.length === 0) return [];
  if (chunks.length <= topN) return chunks;

  const response = await cohere.rerank({
    model:     'rerank-english-v3.0',
    query,
    documents: chunks.map((c) => c.content),
    topN,
    returnDocuments: false,   // we already have the content — save bandwidth
  });

  // Map reranked results back to our chunk objects
  return response.results.map((result) => ({
    ...chunks[result.index],
    score: result.relevanceScore,  // 0.0–1.0, higher = more relevant
  }));
}

// ─── Lightweight local reranker (no API cost) ─────────────────────────────────
/**
 * Simple reciprocal rank reranker using query-term overlap.
 * Not as good as a cross-encoder but free and fast.
 * Use this in development or when Cohere is unavailable.
 */
export function localRerank(
  query:  string,
  chunks: RetrievedChunk[],
  topN:   number = 5
): RetrievedChunk[] {
  const queryTerms = new Set(
    query.toLowerCase()
         .split(/\W+/)
         .filter((t) => t.length > 3)   // skip short stop words
  );

  return chunks
    .map((chunk) => {
      const chunkTerms = chunk.content.toLowerCase().split(/\W+/);
      const overlap    = chunkTerms.filter((t) => queryTerms.has(t)).length;
      const coverage   = overlap / queryTerms.size;    // what % of query terms appear in chunk
      return { ...chunk, score: chunk.score * 0.7 + coverage * 0.3 };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, topN);
}
```

---

## 10. Component 7 — The Prompt (Context Injection)

The prompt is where retrieval meets generation. The quality of your prompt template directly determines answer quality.

### The RAG Prompt Anatomy

```
┌─────────────────────────────────────────────────────────┐
│  SYSTEM PROMPT                                          │
│  "You are a helpful assistant. Answer ONLY using the   │
│   provided context. If the answer is not in the        │
│   context, say you don't know."                        │
├─────────────────────────────────────────────────────────┤
│  CONTEXT BLOCK (retrieved chunks, formatted)           │
│  "Context 1 [Source: refund-policy.pdf, page 3]:       │
│   Orders over $200 require manager approval..."        │
│                                                        │
│  "Context 2 [Source: faq.html]:                        │
│   To initiate a refund, contact support@..."           │
├─────────────────────────────────────────────────────────┤
│  CONVERSATION HISTORY (optional, for multi-turn)       │
│  User: "How long does a refund take?"                  │
│  Assistant: "5-7 business days..."                     │
├─────────────────────────────────────────────────────────┤
│  CURRENT USER QUESTION                                 │
│  "What if my order was $250?"                          │
└─────────────────────────────────────────────────────────┘
```

```typescript
// src/prompt/prompt-builder.ts

import type { RetrievedChunk } from '../retrieval/retriever';

export interface ConversationTurn {
  role:    'user' | 'assistant';
  content: string;
}

export interface PromptOptions {
  query:       string;
  chunks:      RetrievedChunk[];
  history?:    ConversationTurn[];
  systemRole?: string;
  maxContextTokens?: number;   // prevent context from exceeding LLM limit
}

/**
 * Builds the final message array for the LLM.
 *
 * Key decisions:
 * 1. Context ordering: most relevant LAST (LLMs attend more to recent tokens)
 * 2. Source attribution: include source name so LLM can cite it
 * 3. Instruction grounding: explicitly tell LLM to use ONLY the context
 * 4. "I don't know" instruction: prevents hallucination on out-of-context questions
 */
export function buildRAGPrompt(options: PromptOptions): {
  systemPrompt: string;
  messages:     ConversationTurn[];
} {
  const { query, chunks, history = [], systemRole, maxContextTokens = 3000 } = options;

  // ── Format context blocks ──────────────────────────────────────────────────
  let totalTokens = 0;
  const contextBlocks: string[] = [];

  // Reverse order: put most relevant chunk LAST (closer to the question)
  // because LLMs have recency bias — they pay more attention to recent tokens.
  const orderedChunks = [...chunks].reverse();

  for (const chunk of orderedChunks) {
    const sourceLabel = [
      chunk.metadata?.title ?? chunk.source,
      chunk.metadata?.pageNumber ? `page ${chunk.metadata.pageNumber}` : null,
    ].filter(Boolean).join(', ');

    const block = `[Source: ${sourceLabel}]\n${chunk.content}`;
    const tokenEstimate = Math.ceil(block.length / 4);

    if (totalTokens + tokenEstimate > maxContextTokens) break; // stay under limit
    contextBlocks.push(block);
    totalTokens += tokenEstimate;
  }

  const contextText = contextBlocks.length > 0
    ? contextBlocks.join('\n\n---\n\n')
    : 'No relevant context found.';

  // ── System prompt ──────────────────────────────────────────────────────────
  const systemPrompt = systemRole ?? `You are a knowledgeable assistant. Answer the user's question based SOLELY on the provided context below.

Rules:
- If the answer is clearly in the context, answer directly and cite the source.
- If the context is partially relevant, use what you can and note limitations.
- If the context does not contain the answer, say exactly: "I don't have enough information in my knowledge base to answer this question."
- Never make up information not present in the context.
- Keep answers concise and structured. Use bullet points when listing multiple items.`;

  // ── Build message array ────────────────────────────────────────────────────
  const contextMessage: ConversationTurn = {
    role:    'user',
    content: `Here is the relevant context:\n\n${contextText}\n\n---\n\nNow answer this question: ${query}`,
  };

  return {
    systemPrompt,
    messages: [
      ...history.slice(-6),   // keep last 3 turns of conversation (6 messages)
      contextMessage,
    ],
  };
}

// ─── Hypothetical Document Embedding (HyDE) — advanced query expansion ─────────
/**
 * HyDE trick: instead of embedding the raw question (which may be short and vague),
 * ask the LLM to generate a HYPOTHETICAL ANSWER first, then embed that answer.
 *
 * Intuition: "What is the refund policy?" is a short question.
 *             A hypothetical answer ("Refunds are processed within 5 business days...")
 *             uses the same vocabulary and phrasing as the actual policy document.
 *             Embedding the hypothetical answer retrieves better results.
 */
export async function generateHypotheticalDocument(
  query:     string,
  llmClient: (prompt: string) => Promise<string>
): Promise<string> {
  const prompt = `Generate a short hypothetical passage that would perfectly answer this question. Write it in the style of a formal document excerpt. Do NOT say you are generating a hypothetical.

Question: ${query}
Hypothetical passage:`;

  return llmClient(prompt);
}
```

---

## 11. Component 8 — The LLM (Generation)

The LLM receives the system prompt + context + question and generates the answer. In RAG, the LLM's job is narrow: read the provided context and synthesize an answer. It should NOT use its parametric memory (training knowledge) — only the context.

```typescript
// src/llm/llm-service.ts
/**
 * LLM service with:
 *  - Streaming support  — start showing tokens before generation completes
 *  - Token counting     — stay within context window limits
 *  - Source extraction  — parse which sources the LLM cited
 *  - Fallback models    — if primary model fails, try a cheaper backup
 */
import OpenAI from 'openai';
import type { ConversationTurn } from '../prompt/prompt-builder';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export interface LLMResponse {
  content: string;
  model:   string;
  usage:   {
    promptTokens:     number;
    completionTokens: number;
    totalTokens:      number;
    estimatedCostUSD: number;
  };
}

export async function generateAnswer(
  systemPrompt: string,
  messages:     ConversationTurn[],
  model:        string = 'gpt-4o-mini'   // cheaper model for RAG — context does the heavy lifting
): Promise<LLMResponse> {
  const response = await openai.chat.completions.create({
    model,
    messages: [
      { role: 'system',  content: systemPrompt },
      ...messages.map((m) => ({ role: m.role, content: m.content })),
    ],
    temperature:    0.1,     // LOW temperature for factual RAG — we want precision not creativity
    max_tokens:     1024,
    presence_penalty: 0,
    frequency_penalty: 0,
  });

  const content = response.choices[0]?.message?.content ?? '';
  const usage   = response.usage!;

  // Estimate cost (gpt-4o-mini pricing as of 2024)
  const inputCostPer1k  = 0.000150;
  const outputCostPer1k = 0.000600;
  const estimatedCostUSD = (
    (usage.prompt_tokens     / 1000) * inputCostPer1k +
    (usage.completion_tokens / 1000) * outputCostPer1k
  );

  return {
    content,
    model,
    usage: {
      promptTokens:     usage.prompt_tokens,
      completionTokens: usage.completion_tokens,
      totalTokens:      usage.total_tokens,
      estimatedCostUSD,
    },
  };
}

// ─── Streaming generation ──────────────────────────────────────────────────────
export async function* generateAnswerStream(
  systemPrompt: string,
  messages:     ConversationTurn[],
  model:        string = 'gpt-4o-mini'
): AsyncGenerator<string> {
  const stream = await openai.chat.completions.create({
    model,
    stream: true,
    messages: [
      { role: 'system', content: systemPrompt },
      ...messages.map((m) => ({ role: m.role, content: m.content })),
    ],
    temperature: 0.1,
    max_tokens:  1024,
  });

  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content;
    if (delta) yield delta;  // yield each token as it arrives
  }
}
```

---

## 12. How Similarity Search Works Internally

This is what most tutorials skip. Understanding this helps you tune retrieval.

### Cosine Similarity vs Euclidean Distance

```
COSINE SIMILARITY: measures the ANGLE between two vectors (ignores magnitude)
  Score: -1 to 1 (we use 0 to 1 for normalized embeddings)
  Best for: text embeddings (word count doesn't matter, meaning does)

  cos(θ) = (A · B) / (|A| × |B|)

  [0.8, 0.6]  · [0.6, 0.8]    (similar directions)
  ──────────────────────────── = 0.96  (very similar)
  1           × 1

  [0.8, 0.6]  · [-0.6, -0.8]  (opposite directions)
  ──────────────────────────── = -0.96  (very different)
  1           × 1

EUCLIDEAN DISTANCE: measures the straight-line distance between vectors
  Best for: when magnitude matters (e.g. image pixel values)
  Less ideal for text embeddings (a longer document ≠ more important)
```

### Why 1536 Dimensions?

The number of dimensions is set by the embedding model. More dimensions = more capacity to encode meaning distinctions. But there are diminishing returns:

```
Model                    Dimensions   Quality    Cost
──────────────────────────────────────────────────
text-embedding-3-small   1536         Good       $0.02/1M tokens
text-embedding-3-large   3072         Better     $0.13/1M tokens
text-embedding-ada-002   1536         OK (older) $0.10/1M tokens
nomic-embed-text         768          Good       Free (local)
```

### The HNSW Index in Detail

```typescript
// Conceptual illustration of HNSW search (what pgvector does internally)

function hnsw_search_concept(queryVector: number[], topK: number) {
  // Level 2 — very few entry points (logarithmic fraction of all vectors)
  // Start at a random node, greedily move to closer neighbors
  let current = randomEntryPoint();
  current = greedySearch(queryVector, current, level=2);

  // Level 1 — more nodes, finer search
  current = greedySearch(queryVector, current, level=1);

  // Level 0 — all nodes, precise local search
  // ef_search: how many candidates to track simultaneously (higher = more accurate but slower)
  const candidates = beamSearch(queryVector, current, level=0, beamWidth=ef_search);

  return candidates.slice(0, topK);
}

// The ef_search parameter (set at query time) trades speed for accuracy:
// Low ef_search (10):   Fast, misses ~5% of true nearest neighbors
// High ef_search (100): Slower, misses ~0.1%
// Default in pgvector:  40
```

---

## 13. RAG Failure Modes & How to Fix Them

```
FAILURE: "The retrieved chunks are relevant but the answer is still wrong"
CAUSE:   The LLM ignores the context and uses its training memory (hallucination)
FIX:
  1. Lower temperature (0.0–0.1)
  2. Strengthen system prompt: "Answer ONLY from the provided context"
  3. Add a check: compare answer tokens to context tokens (low overlap = hallucination signal)

FAILURE: "The query is clearly in the docs but nothing is retrieved"
CAUSE A: Embedding model mismatch (different model at ingestion vs query)
FIX A:   Ensure EMBEDDING_MODEL constant is the same in both pipelines

CAUSE B: Chunk is too large — relevant sentence is buried, diluting its vector
FIX B:   Reduce chunk size (try 256 tokens), re-ingest

CAUSE C: Query uses different vocabulary than the document
FIX C:   Use hybrid retrieval (dense + sparse BM25)
         Or use HyDE (generate a hypothetical answer first, embed that)

FAILURE: "Retrieved chunks are off-topic / low quality"
CAUSE:   top-k is too high — fetching too many low-relevance chunks
FIX:
  1. Reduce topK (try 3 instead of 10)
  2. Add a score threshold: only return chunks with score > 0.75
  3. Add reranking

FAILURE: "The same document is retrieved multiple times"
CAUSE:   Overlapping chunks contain very similar content
FIX:     MMR (Maximal Marginal Relevance) — penalize retrieved chunks that are
         too similar to already-selected chunks (diversity-aware retrieval)

FAILURE: "Works great for short questions, terrible for complex multi-part questions"
CAUSE:   The query embedding averages too many concepts — poor retrieval
FIX:     Query decomposition — split complex query into sub-questions,
         retrieve for each, then merge context
```

---

## 14. Advanced RAG Patterns

### Query Decomposition

```typescript
// src/advanced/query-decomposition.ts
/**
 * Complex query: "Compare our refund policy for orders under and over $200,
 *                 and what's the timeline for each?"
 *
 * This has 3 sub-questions. A single embedding averages them → poor retrieval.
 * Decompose into sub-queries, retrieve independently, merge results.
 */
export async function decomposeAndRetrieve(
  query:   string,
  llmFn:   (prompt: string) => Promise<string>,
  retrieveFn: (q: string) => Promise<RetrievedChunk[]>
): Promise<{ subQueries: string[]; allChunks: RetrievedChunk[] }> {
  const decompositionPrompt = `Break this question into 2-4 simple sub-questions.
Return ONLY a JSON array of strings. No explanation.
Question: "${query}"`;

  const raw       = await llmFn(decompositionPrompt);
  const subQueries: string[] = JSON.parse(raw.match(/\[[\s\S]*?\]/)?.[0] ?? '[]');

  const chunkSets  = await Promise.all(subQueries.map(retrieveFn));
  const seen       = new Set<string>();
  const allChunks: RetrievedChunk[] = [];

  for (const chunks of chunkSets) {
    for (const chunk of chunks) {
      if (!seen.has(chunk.id)) {
        seen.add(chunk.id);
        allChunks.push(chunk);
      }
    }
  }

  return { subQueries, allChunks };
}
```

### Maximal Marginal Relevance (Diversity-Aware Retrieval)

```typescript
// src/advanced/mmr.ts
/**
 * MMR selects chunks that are:
 *   - Relevant to the query (high similarity to query vector)
 *   - Diverse from already-selected chunks (low similarity to each other)
 *
 * Prevents: 5 retrieved chunks that all say the same thing.
 * Ensures: coverage of different aspects of the topic.
 */
export function maximalMarginalRelevance(
  queryEmbedding: number[],
  candidates:     { chunk: RetrievedChunk; embedding: number[] }[],
  topK:           number = 5,
  lambda:         number = 0.5   // 0 = max diversity, 1 = max relevance
): RetrievedChunk[] {
  const selected: typeof candidates = [];
  const remaining = [...candidates];

  for (let i = 0; i < topK && remaining.length > 0; i++) {
    let bestScore = -Infinity;
    let bestIdx   = 0;

    for (let j = 0; j < remaining.length; j++) {
      const relevance = cosineSimilarity(queryEmbedding, remaining[j].embedding);

      // Penalty: how similar is this candidate to already-selected chunks?
      const maxSimilarityToSelected = selected.length === 0 ? 0
        : Math.max(...selected.map((s) => cosineSimilarity(remaining[j].embedding, s.embedding)));

      const mmrScore = lambda * relevance - (1 - lambda) * maxSimilarityToSelected;

      if (mmrScore > bestScore) {
        bestScore = mmrScore;
        bestIdx   = j;
      }
    }

    selected.push(remaining[bestIdx]);
    remaining.splice(bestIdx, 1);
  }

  return selected.map((s) => s.chunk);
}
```

---

## 15. Project Setup — Express + RAG

```
rag-api/
├── package.json
├── tsconfig.json
├── .env
├── docker-compose.yml         # Postgres + pgvector
└── src/
    ├── index.ts               # Express app bootstrap
    ├── embeddings/
    │   └── embedding-service.ts
    ├── ingestion/
    │   ├── loaders/
    │   │   ├── pdf-loader.ts
    │   │   ├── html-loader.ts
    │   │   └── csv-loader.ts
    │   ├── chunkers/
    │   │   └── chunker.ts
    │   └── ingestion-pipeline.ts
    ├── vectordb/
    │   └── pgvector-store.ts
    ├── retrieval/
    │   ├── retriever.ts
    │   └── reranker.ts
    ├── prompt/
    │   └── prompt-builder.ts
    ├── llm/
    │   └── llm-service.ts
    ├── rag/
    │   └── rag-pipeline.ts    # orchestrates everything
    └── routes/
        ├── query.routes.ts
        ├── ingest.routes.ts
        └── admin.routes.ts
```

```json
// package.json
{
  "name": "rag-api",
  "version": "1.0.0",
  "scripts": {
    "dev":   "ts-node-dev --respawn src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js"
  },
  "dependencies": {
    "express":           "^4.19.0",
    "openai":            "^4.52.0",
    "cohere-ai":         "^7.10.0",
    "pg":                "^8.12.0",
    "pdf-parse":         "^1.1.1",
    "jsdom":             "^24.1.0",
    "csv-parse":         "^5.5.6",
    "multer":            "^1.4.5-lts.1",
    "zod":               "^3.23.8",
    "morgan":            "^1.10.0",
    "cors":              "^2.8.5",
    "express-rate-limit":"^7.3.1",
    "uuid":              "^10.0.0"
  },
  "devDependencies": {
    "@types/express": "^4.17.21",
    "@types/pg":      "^8.11.6",
    "@types/multer":  "^1.4.11",
    "@types/jsdom":   "^21.1.7",
    "@types/morgan":  "^1.9.9",
    "@types/cors":    "^2.8.17",
    "@types/node":    "^20.0.0",
    "ts-node-dev":    "^2.0.0",
    "typescript":     "^5.4.0"
  }
}
```

```env
# .env
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
DATABASE_URL=postgresql://raguser:ragpass@localhost:5432/ragdb
PORT=3000
```

```yaml
# docker-compose.yml — Postgres with pgvector extension
version: "3.9"
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB:       ragdb
      POSTGRES_USER:     raguser
      POSTGRES_PASSWORD: ragpass
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

---

## 16. Ingestion Pipeline — Node.js

```typescript
// src/ingestion/ingestion-pipeline.ts
/**
 * The ingestion pipeline orchestrates:
 *   Load → Chunk → Embed → Store
 *
 * Designed to be idempotent: re-ingesting the same source
 * first deletes existing chunks, then re-creates them.
 * This makes it safe to re-run on document updates.
 */
import { loadPDF }                    from './loaders/pdf-loader';
import { loadHTML }                   from './loaders/html-loader';
import { recursiveChunker }           from './chunkers/chunker';
import { embedBatch }                 from '../embeddings/embedding-service';
import { upsertChunks, deleteBySource, setupVectorTable } from '../vectordb/pgvector-store';
import type { LoadedDocument }        from './loaders/types';

export interface IngestOptions {
  chunkSize?:    number;   // characters per chunk (default: 1000)
  chunkOverlap?: number;   // character overlap between chunks (default: 200)
}

export interface IngestResult {
  source:      string;
  chunksCount: number;
  durationMs:  number;
}

// ─── Ingest a single PDF file ─────────────────────────────────────────────────
export async function ingestPDF(
  filePath: string,
  opts:     IngestOptions = {}
): Promise<IngestResult> {
  const start = Date.now();
  console.log(`[Ingest] Starting PDF: ${filePath}`);

  // Step 1: Load
  const docs = await loadPDF(filePath);
  console.log(`[Ingest] Loaded ${docs.length} pages`);

  // Step 2: Chunk all pages
  const allChunks = docs.flatMap((doc) =>
    recursiveChunker(doc.content, doc.metadata, opts.chunkSize, opts.chunkOverlap)
  ).filter((c) => c.content.length > 50);  // skip tiny fragments

  console.log(`[Ingest] Created ${allChunks.length} chunks`);

  // Step 3: Delete old chunks for this source (idempotency)
  await deleteBySource(filePath);

  // Step 4: Embed all chunks in batches
  const embeddings = await embedBatch(allChunks.map((c) => c.content));
  console.log(`[Ingest] Generated ${embeddings.length} embeddings`);

  // Step 5: Store in vector DB
  await upsertChunks(allChunks, embeddings);

  const durationMs = Date.now() - start;
  console.log(`[Ingest] ✅ Done in ${durationMs}ms`);

  return { source: filePath, chunksCount: allChunks.length, durationMs };
}

// ─── Ingest a URL ─────────────────────────────────────────────────────────────
export async function ingestURL(
  url:  string,
  opts: IngestOptions = {}
): Promise<IngestResult> {
  const start = Date.now();
  console.log(`[Ingest] Starting URL: ${url}`);

  const doc    = await loadHTML(url);
  const chunks = recursiveChunker(doc.content, doc.metadata, opts.chunkSize, opts.chunkOverlap)
                   .filter((c) => c.content.length > 50);

  await deleteBySource(url);
  const embeddings = await embedBatch(chunks.map((c) => c.content));
  await upsertChunks(chunks, embeddings);

  return { source: url, chunksCount: chunks.length, durationMs: Date.now() - start };
}

// ─── Ingest raw text (e.g. from a database row, webhook payload) ──────────────
export async function ingestText(
  content:  string,
  source:   string,
  metadata: Record<string, unknown> = {},
  opts:     IngestOptions = {}
): Promise<IngestResult> {
  const start  = Date.now();
  const chunks = recursiveChunker(content, { source, type: 'txt', ...metadata }, opts.chunkSize, opts.chunkOverlap)
                   .filter((c) => c.content.length > 50);

  await deleteBySource(source);
  const embeddings = await embedBatch(chunks.map((c) => c.content));
  await upsertChunks(chunks, embeddings);

  return { source, chunksCount: chunks.length, durationMs: Date.now() - start };
}
```

---

## 17. Query Pipeline — Express API

```typescript
// src/rag/rag-pipeline.ts
/**
 * The RAG pipeline orchestrates the query flow:
 *   Embed query → Retrieve → (Rerank) → Build prompt → Generate → Return
 *
 * This is the hot path — called on every user question.
 * Every step is timed so you can identify bottlenecks.
 */
import { embedText }            from '../embeddings/embedding-service';
import { hybridRetrieval }      from '../retrieval/retriever';
import { rerankChunks }         from '../retrieval/reranker';
import { buildRAGPrompt }        from '../prompt/prompt-builder';
import { generateAnswer, generateAnswerStream } from '../llm/llm-service';
import type { ConversationTurn } from '../prompt/prompt-builder';
import type { RetrievedChunk }   from '../retrieval/retriever';

export interface RAGRequest {
  query:          string;
  history?:       ConversationTurn[];
  filter?:        { source?: string; tags?: string[] };
  topK?:          number;
  useReranker?:   boolean;
  stream?:        boolean;
}

export interface RAGResponse {
  answer:    string;
  sources:   SourceCitation[];
  timing:    TimingBreakdown;
  usage:     { totalTokens: number; estimatedCostUSD: number };
}

export interface SourceCitation {
  source:     string;
  title?:     string;
  page?:      number;
  score:      number;
  excerpt:    string;    // first 200 chars of the chunk
}

export interface TimingBreakdown {
  embedQueryMs:    number;
  retrievalMs:     number;
  rerankerMs:      number;
  llmMs:           number;
  totalMs:         number;
}

export async function runRAGPipeline(req: RAGRequest): Promise<RAGResponse> {
  const {
    query,
    history       = [],
    filter,
    topK          = 5,
    useReranker   = true,
  } = req;

  const timing: TimingBreakdown = { embedQueryMs: 0, retrievalMs: 0, rerankerMs: 0, llmMs: 0, totalMs: 0 };
  const pipelineStart = Date.now();

  // ── Step 1: Retrieve relevant chunks ────────────────────────────────────────
  const retrievalStart = Date.now();
  // Hybrid retrieval: semantic + keyword for best coverage
  let chunks = await hybridRetrieval(query, topK * 2, filter);   // fetch 2× topK for reranker
  timing.retrievalMs = Date.now() - retrievalStart;

  if (chunks.length === 0) {
    // No relevant chunks found — answer with no context
    return {
      answer:  "I don't have information on that topic in my knowledge base.",
      sources: [],
      timing:  { ...timing, totalMs: Date.now() - pipelineStart },
      usage:   { totalTokens: 0, estimatedCostUSD: 0 },
    };
  }

  // ── Step 2: Rerank (optional, improves precision significantly) ─────────────
  const rerankerStart = Date.now();
  if (useReranker && chunks.length > topK) {
    try {
      chunks = await rerankChunks(query, chunks, topK);
    } catch {
      // Reranker is optional — fall back to vector similarity ranking if it fails
      chunks = chunks.slice(0, topK);
    }
  } else {
    chunks = chunks.slice(0, topK);
  }
  timing.rerankerMs = Date.now() - rerankerStart;

  // ── Step 3: Build prompt ────────────────────────────────────────────────────
  const { systemPrompt, messages } = buildRAGPrompt({ query, chunks, history });

  // ── Step 4: Generate answer ─────────────────────────────────────────────────
  const llmStart = Date.now();
  const llmResponse = await generateAnswer(systemPrompt, messages);
  timing.llmMs = Date.now() - llmStart;

  timing.totalMs = Date.now() - pipelineStart;

  // ── Step 5: Format source citations ─────────────────────────────────────────
  const sources: SourceCitation[] = chunks
    .filter((c) => c.score > 0.5)   // only cite chunks with decent relevance score
    .map((c) => ({
      source:  c.source,
      title:   c.metadata?.title as string | undefined,
      page:    c.metadata?.pageNumber as number | undefined,
      score:   Math.round(c.score * 100) / 100,
      excerpt: c.content.slice(0, 200).trim() + (c.content.length > 200 ? '…' : ''),
    }));

  return {
    answer: llmResponse.content,
    sources,
    timing,
    usage: {
      totalTokens:      llmResponse.usage.totalTokens,
      estimatedCostUSD: llmResponse.usage.estimatedCostUSD,
    },
  };
}

// ─── Streaming version ────────────────────────────────────────────────────────
export async function* runRAGPipelineStream(
  req: RAGRequest
): AsyncGenerator<{ type: 'token' | 'sources' | 'done'; data: unknown }> {
  const { query, history = [], filter, topK = 5, useReranker = true } = req;

  // Retrieve + rerank (non-streaming)
  let chunks = await hybridRetrieval(query, topK * 2, filter);

  if (useReranker && chunks.length > topK) {
    try { chunks = await rerankChunks(query, chunks, topK); }
    catch { chunks = chunks.slice(0, topK); }
  } else {
    chunks = chunks.slice(0, topK);
  }

  // Yield sources FIRST so the client can show them while the answer streams
  const sources: SourceCitation[] = chunks.map((c) => ({
    source:  c.source,
    title:   c.metadata?.title as string | undefined,
    page:    c.metadata?.pageNumber as number | undefined,
    score:   Math.round(c.score * 100) / 100,
    excerpt: c.content.slice(0, 200).trim(),
  }));
  yield { type: 'sources', data: sources };

  // Stream the LLM answer token by token
  const { systemPrompt, messages } = buildRAGPrompt({ query, chunks, history });

  for await (const token of generateAnswerStream(systemPrompt, messages)) {
    yield { type: 'token', data: token };
  }

  yield { type: 'done', data: null };
}
```

---

## 18. Full REST API Implementation

```typescript
// src/index.ts
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import { setupVectorTable } from './vectordb/pgvector-store';
import queryRoutes  from './routes/query.routes';
import ingestRoutes from './routes/ingest.routes';
import adminRoutes  from './routes/admin.routes';

const app  = express();
const PORT = process.env.PORT ?? 3000;

app.use(cors());
app.use(morgan('dev'));
app.use(express.json({ limit: '10mb' }));

app.use('/api/query',  queryRoutes);
app.use('/api/ingest', ingestRoutes);
app.use('/api/admin',  adminRoutes);

app.get('/health', (req, res) => res.json({ status: 'ok' }));

async function start() {
  await setupVectorTable();
  app.listen(PORT, () => console.log(`[Server] RAG API running on http://localhost:${PORT}`));
}

start().catch(console.error);
```

```typescript
// src/routes/query.routes.ts
/**
 * Query routes — the user-facing API.
 *
 * POST /api/query          — single question, full JSON response
 * POST /api/query/stream   — single question, SSE streaming response
 */
import { Router, Request, Response } from 'express';
import { z }                          from 'zod';
import rateLimit                      from 'express-rate-limit';
import { runRAGPipeline, runRAGPipelineStream } from '../rag/rag-pipeline';

const router = Router();

// ─── Rate limiting ─────────────────────────────────────────────────────────────
// LLM calls are expensive — rate limit by IP to control costs
const queryLimiter = rateLimit({
  windowMs: 60 * 1000,    // 1 minute
  max:      20,           // 20 queries per minute per IP
  message:  { error: 'Too many requests. Please wait a moment.' },
});

// ─── Input validation schema ──────────────────────────────────────────────────
const QuerySchema = z.object({
  query:    z.string().min(3).max(2000),
  history:  z.array(z.object({
    role:    z.enum(['user', 'assistant']),
    content: z.string(),
  })).max(10).optional(),
  filter: z.object({
    source: z.string().optional(),
    tags:   z.array(z.string()).optional(),
  }).optional(),
  topK:         z.number().int().min(1).max(20).default(5),
  useReranker:  z.boolean().default(true),
});

// ─── POST /api/query — standard JSON response ──────────────────────────────────
router.post('/', queryLimiter, async (req: Request, res: Response) => {
  const parsed = QuerySchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({
      error:   'Invalid request',
      details: parsed.error.flatten(),
    });
  }

  const startTime = Date.now();

  try {
    const result = await runRAGPipeline({
      ...parsed.data,
      stream: false,
    });

    return res.json({
      success: true,
      data:    result,
      meta: {
        requestId: generateRequestId(),
        durationMs: Date.now() - startTime,
      },
    });
  } catch (err) {
    console.error('[Query] Pipeline error:', err);
    return res.status(500).json({
      error:   'Failed to process query',
      message: err instanceof Error ? err.message : 'Unknown error',
    });
  }
});

// ─── POST /api/query/stream — Server-Sent Events streaming ────────────────────
/**
 * SSE streams the LLM response token-by-token so the UI can
 * start displaying text immediately rather than waiting for the full answer.
 *
 * Client-side usage:
 *   const es = new EventSource('/api/query/stream?query=...');
 *   es.onmessage = (e) => { const data = JSON.parse(e.data); ... }
 *
 * Or fetch-based streaming:
 *   const res = await fetch('/api/query/stream', { method: 'POST', body: ... });
 *   const reader = res.body.getReader();
 */
router.post('/stream', queryLimiter, async (req: Request, res: Response) => {
  const parsed = QuerySchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: 'Invalid request', details: parsed.error.flatten() });
  }

  // Set SSE headers
  res.setHeader('Content-Type',  'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection',    'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');  // disable nginx buffering
  res.flushHeaders();

  const send = (type: string, data: unknown) => {
    res.write(`data: ${JSON.stringify({ type, data })}\n\n`);
  };

  try {
    const generator = runRAGPipelineStream({ ...parsed.data, stream: true });

    for await (const event of generator) {
      send(event.type as string, event.data);

      if (event.type === 'done') break;
    }
  } catch (err) {
    send('error', { message: err instanceof Error ? err.message : 'Unknown error' });
  } finally {
    res.end();
  }
});

function generateRequestId(): string {
  return Math.random().toString(36).slice(2, 10).toUpperCase();
}

export default router;
```

```typescript
// src/routes/ingest.routes.ts
/**
 * Ingestion routes — document management API.
 *
 * POST /api/ingest/file    — upload a PDF or text file
 * POST /api/ingest/url     — ingest a web page by URL
 * POST /api/ingest/text    — ingest raw text with metadata
 * DELETE /api/ingest/:source — remove a document from the knowledge base
 */
import { Router, Request, Response } from 'express';
import multer from 'multer';
import path   from 'path';
import { z }  from 'zod';
import {
  ingestPDF,
  ingestURL,
  ingestText,
} from '../ingestion/ingestion-pipeline';
import { deleteBySource } from '../vectordb/pgvector-store';

const router  = Router();
const upload  = multer({
  dest:   '/tmp/rag-uploads/',
  limits: { fileSize: 50 * 1024 * 1024 }, // 50 MB max
  fileFilter: (_req, file, cb) => {
    const allowed = ['.pdf', '.txt', '.md'];
    const ext     = path.extname(file.originalname).toLowerCase();
    cb(null, allowed.includes(ext));
  },
});

// ─── POST /api/ingest/file ─────────────────────────────────────────────────────
router.post('/file', upload.single('file'), async (req: Request, res: Response) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded or unsupported format (PDF, TXT, MD only)' });
  }

  const { path: filePath, originalname } = req.file;
  const ext = path.extname(originalname).toLowerCase();

  try {
    let result;

    if (ext === '.pdf') {
      result = await ingestPDF(filePath, {
        chunkSize:    parseInt(req.body.chunkSize ?? '1000'),
        chunkOverlap: parseInt(req.body.chunkOverlap ?? '200'),
      });
    } else {
      // Plain text / markdown
      const fs      = await import('fs/promises');
      const content = await fs.readFile(filePath, 'utf-8');
      result = await ingestText(content, originalname, { title: originalname });
    }

    return res.status(201).json({
      success: true,
      message: `Ingested ${result.chunksCount} chunks from "${originalname}"`,
      data:    result,
    });
  } catch (err) {
    console.error('[Ingest] File error:', err);
    return res.status(500).json({
      error:   'Ingestion failed',
      message: err instanceof Error ? err.message : 'Unknown error',
    });
  }
});

// ─── POST /api/ingest/url ─────────────────────────────────────────────────────
const URLSchema = z.object({
  url:          z.string().url(),
  chunkSize:    z.number().optional(),
  chunkOverlap: z.number().optional(),
});

router.post('/url', async (req: Request, res: Response) => {
  const parsed = URLSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: 'Invalid URL', details: parsed.error.flatten() });
  }

  try {
    const result = await ingestURL(parsed.data.url, {
      chunkSize:    parsed.data.chunkSize,
      chunkOverlap: parsed.data.chunkOverlap,
    });

    return res.status(201).json({
      success: true,
      message: `Ingested ${result.chunksCount} chunks from ${parsed.data.url}`,
      data:    result,
    });
  } catch (err) {
    return res.status(500).json({ error: 'Ingestion failed', message: (err as Error).message });
  }
});

// ─── POST /api/ingest/text ────────────────────────────────────────────────────
const TextSchema = z.object({
  content:  z.string().min(10),
  source:   z.string().min(1),
  metadata: z.record(z.unknown()).optional(),
});

router.post('/text', async (req: Request, res: Response) => {
  const parsed = TextSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: 'Invalid request', details: parsed.error.flatten() });
  }

  try {
    const result = await ingestText(parsed.data.content, parsed.data.source, parsed.data.metadata);
    return res.status(201).json({
      success: true,
      message: `Ingested ${result.chunksCount} chunks`,
      data:    result,
    });
  } catch (err) {
    return res.status(500).json({ error: 'Ingestion failed', message: (err as Error).message });
  }
});

// ─── DELETE /api/ingest/:source ───────────────────────────────────────────────
router.delete('/', async (req: Request, res: Response) => {
  const source = req.query.source as string;
  if (!source) return res.status(400).json({ error: 'source query param required' });

  try {
    await deleteBySource(source);
    return res.json({ success: true, message: `Deleted all chunks for source: ${source}` });
  } catch (err) {
    return res.status(500).json({ error: 'Delete failed', message: (err as Error).message });
  }
});

export default router;
```

```typescript
// src/routes/admin.routes.ts
/**
 * Admin routes — introspection and debugging.
 *
 * GET /api/admin/stats   — chunk counts, storage, top sources
 * GET /api/admin/search  — raw vector search (debugging retrieval)
 */
import { Router, Request, Response } from 'express';
import { pool }          from '../vectordb/pgvector-store';
import { denseRetrieval } from '../retrieval/retriever';

const router = Router();

router.get('/stats', async (_req: Request, res: Response) => {
  const [countResult, sourcesResult] = await Promise.all([
    pool.query('SELECT COUNT(*) as total FROM document_chunks'),
    pool.query(`
      SELECT source, COUNT(*) as chunks
      FROM document_chunks
      GROUP BY source
      ORDER BY chunks DESC
      LIMIT 20
    `),
  ]);

  res.json({
    totalChunks: parseInt(countResult.rows[0].total),
    sources:     sourcesResult.rows,
  });
});

router.get('/search', async (req: Request, res: Response) => {
  const query = req.query.q as string;
  const topK  = parseInt(req.query.k as string ?? '5');

  if (!query) return res.status(400).json({ error: 'q param required' });

  const chunks = await denseRetrieval(query, topK);
  res.json({ query, results: chunks });
});

export default router;
```

---

## 19. Streaming Responses

```typescript
// React client — consuming the SSE stream
/**
 * Shows how the frontend can display tokens as they stream in,
 * and show source citations immediately (they arrive before the answer).
 */
async function askRAG(question: string) {
  const response = await fetch('/api/query/stream', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ query: question, topK: 5, useReranker: true }),
  });

  const reader  = response.body!.getReader();
  const decoder = new TextDecoder();
  let answerText = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // SSE data comes as "data: {...}\n\n" — split and parse each line
    const lines = decoder.decode(value).split('\n');

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;

      const event = JSON.parse(line.slice(6));

      if (event.type === 'sources') {
        // Show sources immediately — user sees WHERE the answer comes from
        // before they see the answer itself
        renderSources(event.data);
      }

      if (event.type === 'token') {
        // Append each token to the displayed text as it arrives
        answerText += event.data;
        renderAnswer(answerText);
      }

      if (event.type === 'done') {
        finalize();
        break;
      }
    }
  }
}
```

---

## 20. Evaluation — How to Know if Your RAG is Good

### The Three Metrics That Matter

```typescript
// src/evaluation/evaluator.ts
/**
 * RAG evaluation uses three core metrics:
 *
 * 1. CONTEXT RECALL     — "Did retrieval find ALL the relevant info?"
 *    Measures: what % of the correct answer is covered by retrieved chunks
 *    Low recall = relevant chunks exist but weren't retrieved (chunking/retrieval problem)
 *
 * 2. CONTEXT PRECISION  — "Are the retrieved chunks actually relevant?"
 *    Measures: what % of retrieved chunks are actually relevant to the question
 *    Low precision = too much noise retrieved (topK too high, bad chunking)
 *
 * 3. ANSWER FAITHFULNESS — "Does the answer stick to the retrieved context?"
 *    Measures: what % of answer claims are grounded in the context
 *    Low faithfulness = hallucination (LLM ignoring context, using training memory)
 *
 * Tools: Ragas (Python) is the gold standard for automated RAG evaluation.
 * For Node.js: you can call an LLM-as-judge to score each metric.
 */

export async function evaluateRAGResponse(
  question:        string,
  retrievedChunks: RetrievedChunk[],
  answer:          string,
  groundTruth:     string,        // the correct answer (for evaluation datasets)
  judgeLLM:        (prompt: string) => Promise<string>
): Promise<{ contextPrecision: number; faithfulness: number }> {

  // ── Faithfulness — does the answer hallucinate? ───────────────────────────
  const faithfulnessPrompt = `
Given this context and answer, rate how faithful the answer is to the context on a 0-10 scale.
A faithful answer ONLY makes claims supported by the context.
Return ONLY a JSON object: {"score": <number>, "reason": "<brief reason>"}

Context: ${retrievedChunks.map((c) => c.content).join('\n\n')}
Answer: ${answer}`;

  const faithfulnessRaw = await judgeLLM(faithfulnessPrompt);
  const faithfulness    = JSON.parse(faithfulnessRaw).score / 10;

  // ── Context Precision — are retrieved chunks relevant? ────────────────────
  const precisionScores = await Promise.all(
    retrievedChunks.map(async (chunk) => {
      const prompt = `Is this context chunk relevant to answering the question? Return ONLY {"relevant": true/false}
Question: ${question}
Chunk: ${chunk.content}`;
      const raw = await judgeLLM(prompt);
      return JSON.parse(raw).relevant ? 1 : 0;
    })
  );
  const contextPrecision = precisionScores.reduce((a, b) => a + b, 0) / precisionScores.length;

  return { contextPrecision, faithfulness };
}
```

### Golden Dataset Test

```typescript
// src/evaluation/golden-dataset.ts
/**
 * Build a golden dataset of question → expected answer pairs.
 * Run your RAG pipeline against it and track metric trends over time.
 * This is how you know whether a change (chunk size, topK, model) improved things.
 */
export const goldenDataset = [
  {
    question:    "What is the refund policy for orders over $200?",
    groundTruth: "Orders over $200 require manager approval for refunds",
    sources:     ["refund-policy.pdf"],
  },
  {
    question:    "How long does shipping take?",
    groundTruth: "Standard shipping takes 5-7 business days",
    sources:     ["shipping-policy.pdf"],
  },
  // Add 20-50 of these for a meaningful evaluation suite
];

export async function runEvalSuite() {
  let totalFaithfulness    = 0;
  let totalContextPrecision = 0;

  for (const testCase of goldenDataset) {
    const result = await runRAGPipeline({ query: testCase.question });
    const metrics = await evaluateRAGResponse(
      testCase.question,
      result.sources as any,
      result.answer,
      testCase.groundTruth,
      (p) => generateAnswer('', [{ role: 'user', content: p }]).then((r) => r.content)
    );

    console.log(`Q: ${testCase.question}`);
    console.log(`  Faithfulness:     ${(metrics.faithfulness * 100).toFixed(0)}%`);
    console.log(`  Context Precision: ${(metrics.contextPrecision * 100).toFixed(0)}%`);

    totalFaithfulness     += metrics.faithfulness;
    totalContextPrecision += metrics.contextPrecision;
  }

  console.log('\n=== EVAL RESULTS ===');
  console.log(`Avg Faithfulness:      ${(totalFaithfulness     / goldenDataset.length * 100).toFixed(1)}%`);
  console.log(`Avg Context Precision: ${(totalContextPrecision / goldenDataset.length * 100).toFixed(1)}%`);
}
```

---

## 21. Production Checklist

| # | Category | Check | Reason |
|---|---|---|---|
| 1 | **Chunking** | Tune `chunkSize` for your content type | PDFs: 1000 chars; Code: 500; Legal: 1500 |
| 2 | **Chunking** | Always set `chunkOverlap` > 0 | Prevents losing context at boundaries |
| 3 | **Embedding** | Pin the embedding model version | OpenAI versioned models (v1, v2, v3) are incompatible |
| 4 | **Embedding** | Cache embeddings in Redis | Avoid re-embedding the same query twice |
| 5 | **Retrieval** | Use hybrid (dense + sparse) | BM25 catches what semantic search misses |
| 6 | **Retrieval** | Add score threshold filtering | Don't send low-quality chunks to the LLM |
| 7 | **Reranking** | Enable for production | 20-40% answer quality improvement |
| 8 | **Prompt** | Low temperature (0.0-0.1) | RAG needs precision, not creativity |
| 9 | **Prompt** | Include "I don't know" instruction | Prevents hallucination on gaps |
| 10 | **LLM** | Track token usage per request | LLM cost grows with context size |
| 11 | **API** | Rate limit by IP | Each query costs money |
| 12 | **API** | Validate and cap `topK` | Large topK = large LLM context = expensive |
| 13 | **Storage** | Index the embedding column (HNSW) | Without index: full table scan on every query |
| 14 | **Storage** | Index `source` and `metadata` columns | Enables fast filtering without vector scan |
| 15 | **Observability** | Log retrieval scores | Dropping scores = doc update needed |
| 16 | **Evaluation** | Build a golden dataset | Only way to know if changes help or hurt |
| 17 | **Security** | Sanitize metadata before storage | Prevent injection via document metadata |
| 18 | **Freshness** | Re-ingest on document updates | Stale chunks = wrong answers confidently delivered |

---

## Mental Model — One Paragraph Summary

> RAG is a two-phase system. In the **ingestion phase**, you parse your documents, split them into semantically coherent chunks with overlap, convert each chunk to a vector using an embedding model, and store the vectors + text in a vector database. In the **query phase**, you embed the user's question with the *same* embedding model, find the most similar chunk vectors using approximate nearest-neighbor search (HNSW), optionally rerank the results with a cross-encoder for precision, inject the top chunks into the LLM's prompt as context, and generate an answer grounded in your documents. The entire system's quality depends on three things: chunking strategy (how you split documents), retrieval quality (how well you find the right chunks), and prompt design (whether the LLM stays grounded in the context). Everything else is optimization.

---

*Node.js 20 · Express 4 · OpenAI API · pgvector (Postgres) · Cohere Rerank · TypeScript 5*
