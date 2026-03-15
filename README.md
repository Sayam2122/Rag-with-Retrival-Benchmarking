# RAG with Retrieval Benchmarking

An assignment-ready Retrieval-Augmented Generation (RAG) pipeline built on arXiv paper data.

## What This Project Does

- Loads and prepares paper documents from a CSV dataset.
- Splits documents into overlapping chunks for better retrieval.
- Builds normalized embeddings with Sentence Transformers.
- Indexes chunks in FAISS for vector search.
- Adds BM25 keyword retrieval and hybrid reranking.
- Generates answers from retrieved context with a Hugging Face model.
- Evaluates retrieval using Precision@k, Recall@k, and MRR.

## Project Files

- `rag_assignment_pipeline.py`: Main end-to-end pipeline.
- `arxiv_600_papers.csv`: Input dataset (expected by default).
- `faiss_index.bin`: Saved FAISS index (created after build).
- `faiss_metadata.json`: Chunk metadata (created after build).

## Install

```bash
pip install pandas numpy tqdm sentence-transformers faiss-cpu rank-bm25 transformers tf-keras
```

## Usage

### 1) Build chunked index

```bash
python rag_assignment_pipeline.py --build --chunk-size 300 --chunk-overlap 50
```

### 2) Retrieve only

```bash
python rag_assignment_pipeline.py --query "diffusion models for image generation" --top-k 5 --retrieval-mode hybrid
```

### 3) Retrieve + generate answer

```bash
python rag_assignment_pipeline.py --query "What are diffusion models for image generation?" --top-k 5 --retrieval-mode hybrid --generate
```

### 4) Run benchmarking

```bash
python rag_assignment_pipeline.py --benchmark --top-k 5
```

The script reports:

- `precision@5`
- `recall@5`
- `mrr`

for both vector and hybrid retrieval modes.

## Notes

- The script prints full context and prompt sent to the generation model for debugging.
- Generation currently uses `google/flan-t5-small` by default.
- For small models, top-3 context passages often improve answer quality over longer context.
