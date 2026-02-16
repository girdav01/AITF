# RAG Span Conventions (RAG_CONTEXT)

Status: **Normative** | CoSAI WS2 Alignment: **RAG_CONTEXT** | OCSF Class: **7004 Data Retrieval**

AITF defines semantic conventions for Retrieval-Augmented Generation (RAG) pipelines, covering query processing, vector retrieval, document scoring, reranking, and context quality evaluation. This specification defines the normative field requirements aligned with CoSAI Working Stream 2 (Telemetry for AI) and mapped to applicable compliance and threat frameworks.

Key words "MUST", "SHOULD", "MAY" follow [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

---

## Overview

RAG pipelines combine retrieval from external knowledge bases with generative AI to produce grounded, context-aware responses. AITF provides telemetry across the full RAG lifecycle:

```
RAG Pipeline:
  query -> embed -> retrieve -> [rerank] -> augment -> generate -> [evaluate]

Spans:
  aitf.rag.pipeline      (root)
    aitf.rag.query       (query embedding)
    aitf.rag.retrieve    (vector search)
    aitf.rag.rerank      (optional reranking)
    gen_ai.inference      (generation with context)
    aitf.rag.evaluate    (optional quality evaluation)
```

---

## Span: `aitf.rag.pipeline`

Represents a complete RAG pipeline execution.

### Span Name

Format: `rag.pipeline {aitf.rag.pipeline.name}`

### Span Kind

`INTERNAL`

### Normative Field Table

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.rag.pipeline.name` | string | **Required** | Pipeline name/identifier | NIST AI RMF MAP-1.1 |
| `aitf.rag.pipeline.stage` | string | **Required** | Current stage: `"retrieve"`, `"rerank"`, `"generate"`, `"evaluate"` | NIST AI RMF MEASURE-2.5 |
| `aitf.rag.query` | string | **Required** | User query text | OWASP LLM01 (Prompt Injection), MITRE ATLAS [AML.T0051](https://atlas.mitre.org/techniques/AML.T0051) |

---

## Span: `aitf.rag.query`

Represents query processing and embedding generation for retrieval.

### Span Name

Format: `rag.query {aitf.rag.pipeline.name}`

### Span Kind

`INTERNAL`

### Normative Field Table

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.rag.query` | string | **Required** | User query text | OWASP LLM01, MITRE ATLAS AML.T0051 |
| `aitf.rag.query.embedding_model` | string | **Recommended** | Embedding model used | MITRE ATLAS [AML.T0044](https://atlas.mitre.org/techniques/AML.T0044), EU AI Act Art.13 |
| `aitf.rag.query.embedding_dimensions` | int | **Optional** | Embedding vector dimensions | — |

---

## Span: `aitf.rag.retrieve`

Represents the retrieval/vector search phase of a RAG pipeline.

### Span Name

Format: `rag.retrieve {aitf.rag.retrieve.database}`

### Span Kind

`CLIENT`

### Normative Field Table

#### Database & Query (CoSAI WS2)

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.rag.retrieve.database` | string | **Required** | Vector database name (e.g. `"pinecone"`, `"chromadb"`, `"weaviate"`, `"pgvector"`) | OWASP LLM08 (Vector/Embedding Weaknesses), MITRE ATLAS [AML.T0043](https://atlas.mitre.org/techniques/AML.T0043) |
| `aitf.rag.query` | string | **Required** | Query text used for retrieval | OWASP LLM01, MITRE ATLAS AML.T0051 |
| `aitf.rag.retrieve.index` | string | **Recommended** | Index/collection name | NIST AI RMF MAP-1.5 |
| `aitf.rag.retrieve.top_k` | int | **Recommended** | Number of results requested | NIST AI RMF MEASURE-2.5 |
| `aitf.rag.retrieve.filter` | string | **Optional** | Metadata filter (JSON) | — |

#### Retrieval Results (CoSAI WS2)

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.rag.retrieve.results_count` | int | **Required** | Actual number of documents returned | NIST AI RMF MEASURE-2.5, OWASP LLM08 |
| `aitf.rag.retrieval.docs` | string | **Recommended** | JSON array of retrieved document summaries (id, score, snippet) | OWASP LLM08, MITRE ATLAS [AML.T0043](https://atlas.mitre.org/techniques/AML.T0043) |
| `aitf.rag.retrieve.min_score` | double | **Recommended** | Minimum similarity score among results | OWASP LLM08, NIST AI RMF MEASURE-2.5 |
| `aitf.rag.retrieve.max_score` | double | **Recommended** | Maximum similarity score among results | OWASP LLM08, NIST AI RMF MEASURE-2.5 |

#### Per-Document Fields (CoSAI WS2)

These fields are emitted as span events (`rag.doc.retrieved`) for each retrieved document, or as JSON within `aitf.rag.retrieval.docs`:

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.rag.doc.id` | string | **Recommended** | Document/chunk identifier | NIST AI RMF MAP-1.5, EU AI Act Art.12 (Record-Keeping) |
| `aitf.rag.doc.score` | double | **Recommended** | Similarity/relevance score for this document (0.0–1.0) | OWASP LLM08, NIST AI RMF MEASURE-2.5 |
| `aitf.rag.doc.provenance` | string | **Recommended** | Document source/origin (e.g. URL, collection, upload ID) | OWASP LLM09 (Misinformation), EU AI Act Art.13 (Transparency) |

---

## Span: `aitf.rag.rerank`

Represents the optional reranking phase of a RAG pipeline.

### Span Name

Format: `rag.rerank {aitf.rag.rerank.model}`

### Span Kind

`CLIENT`

### Normative Field Table

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.rag.rerank.model` | string | **Required** | Reranking model name (e.g. `"cross-encoder/ms-marco"`) | MITRE ATLAS AML.T0044, EU AI Act Art.13 |
| `aitf.rag.rerank.input_count` | int | **Required** | Number of documents before reranking | NIST AI RMF MEASURE-2.5 |
| `aitf.rag.rerank.output_count` | int | **Required** | Number of documents after reranking | NIST AI RMF MEASURE-2.5 |

---

## Span: `aitf.rag.evaluate`

Represents quality evaluation of the RAG pipeline output.

### Span Name

Format: `rag.evaluate {aitf.rag.pipeline.name}`

### Span Kind

`INTERNAL`

### Normative Field Table

| Field Name | Type | Requirement | Description | Compliance |
|---|---|---|---|---|
| `aitf.rag.quality.context_relevance` | double | **Recommended** | Context relevance score (0.0–1.0) | NIST AI RMF MEASURE-2.5 |
| `aitf.rag.quality.answer_relevance` | double | **Recommended** | Answer relevance score (0.0–1.0) | NIST AI RMF MEASURE-2.5 |
| `aitf.rag.quality.faithfulness` | double | **Recommended** | Answer faithfulness to retrieved context (0.0–1.0) | OWASP LLM09 (Misinformation), NIST AI RMF MEASURE-2.5 |
| `aitf.rag.quality.groundedness` | double | **Recommended** | How grounded the answer is in source material (0.0–1.0) | OWASP LLM09, EU AI Act Art.13 |

---

## CoSAI WS2 Field Mapping

Cross-reference between CoSAI WS2 `RAG_CONTEXT` field names and AITF attribute keys:

| CoSAI WS2 Field | AITF Attribute | Notes |
|---|---|---|
| `rag.database.name` | `aitf.rag.retrieve.database` | Vector DB identifier |
| `rag.query.text` | `aitf.rag.query` | User query text |
| `rag.retrieval.docs` | `aitf.rag.retrieval.docs` | New in CoSAI WS2 alignment; JSON array |
| `rag.doc.id` | `aitf.rag.doc.id` | New in CoSAI WS2 alignment |
| `rag.doc.score` | `aitf.rag.doc.score` | New in CoSAI WS2 alignment |
| `rag.doc.provenance` | `aitf.rag.doc.provenance` | New in CoSAI WS2 alignment |

---

## Example: RAG Pipeline with Reranking

```
Span: rag.pipeline knowledge-base
  aitf.rag.pipeline.name: "knowledge-base"
  aitf.rag.pipeline.stage: "generate"
  aitf.rag.query: "What are the OWASP LLM Top 10 risks?"
  |
  +- Span: rag.query knowledge-base
  |    aitf.rag.query: "What are the OWASP LLM Top 10 risks?"
  |    aitf.rag.query.embedding_model: "text-embedding-3-small"
  |    aitf.rag.query.embedding_dimensions: 1536
  |
  +- Span: rag.retrieve pinecone
  |    aitf.rag.retrieve.database: "pinecone"
  |    aitf.rag.retrieve.index: "security-docs"
  |    aitf.rag.retrieve.top_k: 10
  |    aitf.rag.retrieve.results_count: 10
  |    aitf.rag.retrieve.min_score: 0.72
  |    aitf.rag.retrieve.max_score: 0.96
  |    aitf.rag.retrieval.docs: "[{\"id\":\"doc-001\",\"score\":0.96,\"provenance\":\"owasp.org/llm-top-10\"}]"
  |    Events:
  |      rag.doc.retrieved: {aitf.rag.doc.id: "doc-001", aitf.rag.doc.score: 0.96, aitf.rag.doc.provenance: "owasp.org/llm-top-10"}
  |      rag.doc.retrieved: {aitf.rag.doc.id: "doc-002", aitf.rag.doc.score: 0.91, aitf.rag.doc.provenance: "atlas.mitre.org"}
  |
  +- Span: rag.rerank cross-encoder/ms-marco
  |    aitf.rag.rerank.model: "cross-encoder/ms-marco"
  |    aitf.rag.rerank.input_count: 10
  |    aitf.rag.rerank.output_count: 5
  |
  +- Span: chat gpt-4o
  |    gen_ai.system: "openai"
  |    gen_ai.usage.input_tokens: 2500
  |    gen_ai.usage.output_tokens: 800
  |
  +- Span: rag.evaluate knowledge-base
       aitf.rag.quality.context_relevance: 0.92
       aitf.rag.quality.answer_relevance: 0.88
       aitf.rag.quality.faithfulness: 0.95
       aitf.rag.quality.groundedness: 0.93
```
