# Milvus Vector Database — Dockerized Semantic Search

A containerized Milvus deployment demonstrating end-to-end vector search: schema design, embedding generation, indexing, and similarity search over natural language text.

---

## What this is

A working semantic search pipeline built on **Milvus**, a purpose-built vector database. Text goes in, gets converted to a dense embedding, gets indexed for fast similarity search, and a natural-language query can retrieve the closest match — the same core mechanism behind RAG systems, recommendation engines, and semantic document search.

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    etcd     │     │    MinIO    │     │   Milvus     │
│ (metadata   │◄────│ (object     │◄────│  standalone  │
│  store)     │     │  storage)   │     │   server     │
└─────────────┘     └─────────────┘     └─────────────┘
                                                │
                                         PyMilvus client
                                                │
                                    sentence-transformers
                                     (all-MiniLM-L6-v2)
```

Milvus standalone doesn't run alone — it depends on **etcd** for metadata coordination and **MinIO** for object storage, both orchestrated together via Docker Compose so the whole stack comes up with one command.

---

## Stack

| Component | Role |
|---|---|
| Milvus v2.4.0 | Vector database engine |
| etcd | Metadata/coordination store |
| MinIO | Object storage backend |
| PyMilvus | Python client SDK |
| sentence-transformers (`all-MiniLM-L6-v2`) | Text → 384-dim embedding |
| Docker Compose | Multi-container orchestration |

---

## Schema design

Collection `documents`:

| Field | Type | Notes |
|---|---|---|
| `id` | INT64 | Primary key |
| `text` | VARCHAR(500) | Original source text |
| `embedding` | FLOAT_VECTOR(384) | Dense embedding from MiniLM |

**Index:** `IVF_FLAT` with `L2` distance metric, `nlist=128`. IVF_FLAT partitions the vector space into clusters (controlled by `nlist`) and searches only the most relevant clusters at query time — a solid balance of speed and exact-enough results for moderate-scale collections, without the memory overhead of a full graph-based index.

---

## Pipeline

1. **`docker-compose.yml`** — spins up etcd, MinIO, and Milvus standalone together
2. **`docker_test.py`** — verifies the client can connect to the running Milvus instance
3. **`milvus_collection.py`** — defines the schema, creates the collection, builds the IVF_FLAT index
4. **`milvus_insert_docker.py`** — encodes text with `all-MiniLM-L6-v2` and inserts the resulting vector alongside its source text
5. **`milvus_search_docker.py`** — encodes a natural-language query the same way, then searches the collection for the nearest vector by L2 distance

---

## Run it

```bash
# 1. Start the stack
docker-compose up -d

# 2. Verify connection
python docker_test.py

# 3. Create the collection + index
python milvus_collection.py

# 4. Insert a document
python milvus_insert_docker.py

# 5. Run a semantic search
python milvus_search_docker.py
```

---

## Why this matters

This is the foundational pattern behind retrieval-augmented generation (RAG): instead of keyword matching, text is compared by *meaning* — `all-MiniLM-L6-v2` maps semantically similar sentences to nearby points in 384-dimensional space, so a query like "What is a vector database?" can retrieve a document that never uses those exact words, as long as the meaning is close.

## Notes

- Default MinIO credentials (`minioadmin`/`minioadmin`) are Docker Compose demo defaults — swap these for real secrets before using this outside a local sandbox.
- `nlist=128` is tuned for small-to-moderate collections; for larger datasets this would need retuning (or a switch to a graph-based index like HNSW) to keep recall high without a linear scan.
