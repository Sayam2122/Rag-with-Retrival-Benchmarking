import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Reduce noisy backend logs and warnings during runs.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import pipeline


DATA_PATH = Path("arxiv_600_papers.csv")
INDEX_PATH = Path("faiss_index.bin")
META_PATH = Path("faiss_metadata.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
GENERATOR = None
GENERATOR_MODEL_NAME = None


def load_dataset(csv_path: Path, max_docs: int = 600) -> pd.DataFrame:
	if not csv_path.exists():
		raise FileNotFoundError(f"Dataset not found: {csv_path}")

	df = pd.read_csv(csv_path)
	df = df.head(max_docs).copy()

	if "document" not in df.columns:
		missing = [c for c in ["title", "abstract"] if c not in df.columns]
		if missing:
			raise ValueError(f"Missing required columns for document creation: {missing}")
		df["document"] = df["title"].fillna("") + ". " + df["abstract"].fillna("")

	df["document"] = df["document"].fillna("")
	return df


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
	words = str(text).split()
	if not words:
		return []

	chunks = []
	start = 0
	step = max(1, chunk_size - overlap)
	while start < len(words):
		end = start + chunk_size
		chunk = " ".join(words[start:end]).strip()
		if chunk:
			chunks.append(chunk)
		start += step

	return chunks


def build_chunked_dataframe(df: pd.DataFrame, chunk_size: int = 300, overlap: int = 50) -> pd.DataFrame:
	rows = []
	for idx, row in df.iterrows():
		chunks = chunk_text(row.get("document", ""), chunk_size=chunk_size, overlap=overlap)
		for chunk_id, chunk in enumerate(chunks):
			rows.append(
				{
					"doc_id": int(idx),
					"chunk_id": int(chunk_id),
					"title": str(row.get("title", "")),
					"arxiv_id": str(row.get("arxiv_id", "")),
					"document": chunk,
				}
			)

	if not rows:
		raise ValueError("No chunks were created from input documents.")

	return pd.DataFrame(rows)


def encode_documents(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
	embeddings = model.encode(
		texts,
		batch_size=batch_size,
		show_progress_bar=True,
		convert_to_numpy=True,
		normalize_embeddings=True,
	)
	return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
	dim = embeddings.shape[1]
	index = faiss.IndexFlatIP(dim)
	index.add(embeddings)
	return index


def save_artifacts(index: faiss.Index, df: pd.DataFrame, index_path: Path, meta_path: Path) -> None:
	faiss.write_index(index, str(index_path))
	metadata = {
		"columns": df.columns.tolist(),
		"records": df.to_dict(orient="records"),
	}
	meta_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")


def load_artifacts(index_path: Path, meta_path: Path) -> Tuple[faiss.Index, pd.DataFrame]:
	if not index_path.exists() or not meta_path.exists():
		raise FileNotFoundError("Index or metadata not found. Run with --build first.")

	index = faiss.read_index(str(index_path))
	metadata = json.loads(meta_path.read_text(encoding="utf-8"))
	df = pd.DataFrame(metadata["records"])
	return index, df


def retrieve(
	query: str,
	model: SentenceTransformer,
	index: faiss.Index,
	df: pd.DataFrame,
	top_k: int = 5,
) -> List[Dict]:
	query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
	scores, indices = index.search(query_vec, top_k)

	results = []
	for score, idx in zip(scores[0], indices[0]):
		if idx < 0 or idx >= len(df):
			continue
		row = df.iloc[idx]
		paper_doc_id = int(row.get("doc_id", idx))
		results.append(
			{
				"rank": len(results) + 1,
				"score": float(score),
				"doc_id": paper_doc_id,
				"chunk_row_id": int(idx),
				"chunk_id": int(row.get("chunk_id", -1)),
				"title": str(row.get("title", "")),
				"arxiv_id": str(row.get("arxiv_id", "")),
				"document": str(row.get("document", "")),
			}
		)
	return results


def build_bm25_index(df: pd.DataFrame) -> BM25Okapi:
	tokenized_docs = [str(doc).split() for doc in df["document"].tolist()]
	return BM25Okapi(tokenized_docs)


def bm25_search(query: str, bm25: BM25Okapi, df: pd.DataFrame, top_k: int = 5) -> List[Dict]:
	tokens = query.split()
	scores = bm25.get_scores(tokens)
	top_indices = np.argsort(scores)[::-1][:top_k]

	results = []
	for rank, idx in enumerate(top_indices, start=1):
		row = df.iloc[idx]
		paper_doc_id = int(row.get("doc_id", idx))
		results.append(
			{
				"rank": rank,
				"score": float(scores[idx]),
				"doc_id": paper_doc_id,
				"chunk_row_id": int(idx),
				"chunk_id": int(row.get("chunk_id", -1)),
				"title": str(row.get("title", "")),
				"arxiv_id": str(row.get("arxiv_id", "")),
				"document": str(row.get("document", "")),
			}
		)
	return results


def _normalize_scores(results: List[Dict]) -> List[Dict]:
	if not results:
		return []
	scores = np.array([r["score"] for r in results], dtype="float32")
	min_s = float(np.min(scores))
	max_s = float(np.max(scores))
	if abs(max_s - min_s) < 1e-12:
		for r in results:
			r["norm_score"] = 1.0
		return results

	for r in results:
		r["norm_score"] = (r["score"] - min_s) / (max_s - min_s)
	return results


def hybrid_retrieve(
	query: str,
	model: SentenceTransformer,
	index: faiss.Index,
	bm25: BM25Okapi,
	df: pd.DataFrame,
	top_k: int = 5,
) -> List[Dict]:
	vector_results = retrieve(query, model, index, df, top_k=top_k)
	keyword_results = bm25_search(query, bm25, df, top_k=top_k)

	vector_results = _normalize_scores(vector_results)
	keyword_results = _normalize_scores(keyword_results)

	combined: Dict[int, Dict] = {}
	for item in vector_results:
		doc_key = item["doc_id"]
		combined[doc_key] = {
			**item,
			"vector_score": item.get("norm_score", 0.0),
			"keyword_score": 0.0,
			"score": item.get("norm_score", 0.0),
		}

	for item in keyword_results:
		doc_key = item["doc_id"]
		if doc_key not in combined:
			combined[doc_key] = {
				**item,
				"vector_score": 0.0,
				"keyword_score": item.get("norm_score", 0.0),
				"score": item.get("norm_score", 0.0),
			}
		else:
			combined[doc_key]["keyword_score"] = item.get("norm_score", 0.0)
			combined[doc_key]["score"] = combined[doc_key]["vector_score"] + combined[doc_key]["keyword_score"]

	results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]
	for rank, item in enumerate(results, start=1):
		item["rank"] = rank
	return results


def build_generation_prompt(query: str, retrieved_docs: List[Dict], max_docs: int = 3) -> str:
	if not retrieved_docs:
		return f"Question: {query}\n\nContext: No relevant documents were retrieved."

	context = build_llm_context(retrieved_docs, max_docs=max_docs)
	prompt = f"""
You are a research assistant.

Answer the question using the research paper excerpts below.

Instructions:
- Use only relevant papers.
- Ignore papers unrelated to the question.
- Write a clear explanation in 4-5 sentences.
- Do not copy text directly; summarize the idea.

Question:
{query}

Research Papers:
{context}

Answer:
"""
	return prompt.strip()


def build_llm_context(retrieved_docs: List[Dict], max_docs: int = 3) -> str:
	context_snippets = []
	for i, item in enumerate(retrieved_docs[:max_docs]):
		snippet = item["document"][:500].replace("\n", " ")
		context_snippets.append(f"Paper {i+1}: {snippet}")
	return "\n\n".join(context_snippets)


def generate_answer_with_llm(
	query: str,
	retrieved_docs: List[Dict],
	model_name: str = GEN_MODEL_NAME,
	max_new_tokens: int = 200,
) -> str:
	global GENERATOR, GENERATOR_MODEL_NAME
	do_sample = False
	temperature = None
	context = build_llm_context(retrieved_docs, max_docs=3)
	print("\n===== CONTEXT SENT TO LLM =====\n")
	print(context)
	print("\n===============================\n")
	prompt = build_generation_prompt(query, retrieved_docs, max_docs=3)
	print("\n===== PROMPT SENT TO MODEL =====\n")
	print(prompt)
	print("\n================================\n")
	print("Generation settings:")
	print(f"max_new_tokens={max_new_tokens}")
	print(f"do_sample={do_sample}")
	print(f"temperature={temperature}")

	try:
		if GENERATOR is None or GENERATOR_MODEL_NAME != model_name:
			GENERATOR = pipeline("text2text-generation", model=model_name)
			GENERATOR_MODEL_NAME = model_name
		outputs = GENERATOR(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, truncation=True)
		if outputs and isinstance(outputs, list):
			text = outputs[0].get("generated_text", "").strip()
			if text:
				return text
		return "LLM generation returned an empty response."
	except Exception as exc:
		return f"LLM generation failed: {exc}"


def build_eval_queries(df: pd.DataFrame, n_queries: int = 50) -> List[Tuple[str, int]]:
	if "doc_id" in df.columns:
		paper_df = df.sort_values(by=["doc_id", "chunk_id"]).drop_duplicates(subset=["doc_id"], keep="first")
	else:
		paper_df = df.copy()

	n = min(n_queries, len(paper_df))
	eval_pairs = []
	for i in range(n):
		row = paper_df.iloc[i]
		title = str(row.get("title", ""))
		relevant_doc_id = int(row.get("doc_id", i))
		query = f"Find the paper: {title}"
		eval_pairs.append((query, relevant_doc_id))
	return eval_pairs


def compute_metrics_at_k(
	index: faiss.Index,
	model: SentenceTransformer,
	df: pd.DataFrame,
	bm25: Optional[BM25Okapi] = None,
	retrieval_mode: str = "vector",
	k: int = 5,
) -> Dict[str, float]:
	eval_pairs = build_eval_queries(df)
	precisions: List[float] = []
	recalls: List[float] = []
	reciprocal_ranks: List[float] = []

	for query, relevant_doc_id in tqdm(eval_pairs, desc="Benchmarking"):
		if retrieval_mode == "hybrid":
			if bm25 is None:
				raise ValueError("BM25 index is required for hybrid benchmarking.")
			results = hybrid_retrieve(query, model, index, bm25, df, top_k=k)
		else:
			results = retrieve(query, model, index, df, top_k=k)
		retrieved_ids = [r["doc_id"] for r in results]

		hit = 1 if relevant_doc_id in retrieved_ids else 0
		precisions.append(hit / float(k))
		recalls.append(float(hit))

		rr = 0.0
		for rank, doc_id in enumerate(retrieved_ids, start=1):
			if doc_id == relevant_doc_id:
				rr = 1.0 / rank
				break
		reciprocal_ranks.append(rr)

	return {
		f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
		f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
		"mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
		"num_eval_queries": len(eval_pairs),
		"retrieval_mode": retrieval_mode,
	}


def print_retrieval_results(results: List[Dict]) -> None:
	if not results:
		print("No results found.")
		return

	print("\nTop retrieved documents:")
	for item in results:
		chunk_id = item.get("chunk_id", -1)
		print(
			f"{item['rank']}. score={item['score']:.4f} | doc_id={item['doc_id']} | "
			f"chunk_id={chunk_id} | title={item['title']}"
		)


def main() -> None:
	parser = argparse.ArgumentParser(description="RAG assignment pipeline with embeddings, FAISS retrieval, and benchmarking.")
	parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to CSV dataset")
	parser.add_argument("--build", action="store_true", help="Build embeddings and FAISS index")
	parser.add_argument("--query", type=str, default=None, help="User query for retrieval")
	parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
	parser.add_argument("--benchmark", action="store_true", help="Run retrieval benchmark metrics")
	parser.add_argument("--generate", action="store_true", help="Generate final answer with an LLM")
	parser.add_argument("--gen-model", type=str, default=GEN_MODEL_NAME, help="Hugging Face model for generation")
	parser.add_argument("--retrieval-mode", type=str, default="hybrid", choices=["vector", "hybrid"], help="Retrieval mode")
	parser.add_argument("--chunk-size", type=int, default=300, help="Chunk size in words")
	parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap in words")
	args = parser.parse_args()

	data_path = Path(args.data)
	model = SentenceTransformer(MODEL_NAME)

	if args.build:
		raw_df = load_dataset(data_path, max_docs=600)
		df = build_chunked_dataframe(raw_df, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
		documents = df["document"].tolist()
		print(f"Loaded {len(documents)} documents")

		embeddings = encode_documents(model, documents)
		print(f"Embeddings shape: {embeddings.shape}")

		index = build_faiss_index(embeddings)
		save_artifacts(index, df, INDEX_PATH, META_PATH)
		print(f"Saved index to {INDEX_PATH} and metadata to {META_PATH}")

	if args.query or args.benchmark:
		index, df = load_artifacts(INDEX_PATH, META_PATH)
		bm25 = build_bm25_index(df)

		if args.query:
			if args.retrieval_mode == "hybrid":
				results = hybrid_retrieve(args.query, model, index, bm25, df, top_k=args.top_k)
			else:
				results = retrieve(args.query, model, index, df, top_k=args.top_k)
			print_retrieval_results(results)

			if args.generate:
				print(f"\nUsing {min(3, len(results))} retrieved documents for generation")
				final_answer = generate_answer_with_llm(args.query, results, model_name=args.gen_model)
				print("\nFinal Answer:\n" + final_answer)
			else:
				print("\nGenerated prompt (context passed to LLM):\n" + build_generation_prompt(args.query, results))

		if args.benchmark:
			vector_metrics = compute_metrics_at_k(index, model, df, retrieval_mode="vector", k=args.top_k)
			hybrid_metrics = compute_metrics_at_k(index, model, df, bm25=bm25, retrieval_mode="hybrid", k=args.top_k)

			print("\nBenchmark metrics (Vector):")
			for k, v in vector_metrics.items():
				print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

			print("\nBenchmark metrics (Hybrid):")
			for k, v in hybrid_metrics.items():
				print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
	main()
