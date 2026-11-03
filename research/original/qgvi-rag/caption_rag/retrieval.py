"""A-RAG sentence index with view-level hybrid retrieval.

The index is built by the user's local A-RAG ``scripts/build_index.py``.
This adapter pools *all* sentence scores per view before truncating results,
adds token-based BM25, and fuses both rankings with RRF (constant 60). Scores
are retrieval ranks/similarities, not probabilities of visual correctness.

Only load trusted local index directories produced by ``build_index``. A-RAG
uses pickle; the adjacent hashes detect corruption, not malicious artifacts.
Do not download or accept external pickle files as retrieval inputs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import re
from typing import Any


_ADAPTER_VERSION = 1
# Bound the temporary query-by-sentence score matrix independently of the
# caller's embedding batch size. A single query still needs one score per
# indexed sentence, so one row is the lower bound for this allocation.
_MAX_SCORE_BYTES = 64 * 1024 * 1024
_STOP_WORDS = frozenset(
    "a an the and or of to in on at for from by with is are was were be been "
    "being it its this that these those there here as i me my we our you your "
    "he she they them their can could would should will do does did please "
    "show shows showing find get give want need image images photo photos "
    "picture pictures view views scene scenes relevant retrieve looking see".split()
)
_CJK_STOP = frozenset("的了和是在有与及把被这那我你它们请找看")
_TOKEN_RE = re.compile(r"[\u3400-\u9fff]+|[^\W_]+", flags=re.UNICODE)


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _tokens(text: str) -> list[str]:
    """Words plus CJK unigrams/bigrams; no external tokenizer is required."""
    result = []
    for token in _TOKEN_RE.findall(text.casefold()):
        if "\u3400" <= token[0] <= "\u9fff":
            result.extend(char for char in token if char not in _CJK_STOP)
            result.extend(
                token[i : i + 2]
                for i in range(len(token) - 1)
                if not any(char in _CJK_STOP for char in token[i : i + 2])
            )
        elif token not in _STOP_WORDS:
            result.append(token)
    return result


def _load_model(model_name: str, device: str):
    # Keep manifest validation, help, and caption generation independent of torch.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def _validate_chunks(chunks: Any) -> dict[str, dict]:
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("A-RAG needs a non-empty list of caption chunks")
    lookup = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Each caption chunk must be an object")
        view_id, text = chunk.get("id"), chunk.get("text")
        if not isinstance(view_id, str) or not view_id.strip():
            raise ValueError("Every caption chunk needs a non-empty string id")
        if view_id in lookup:
            raise ValueError(f"Duplicate caption view id: {view_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Caption text is empty for view {view_id}")
        lookup[view_id] = chunk
    return lookup


def _validate_index(data: Any, chunks: list[dict], expected_model: str):
    import numpy as np

    lookup = _validate_chunks(chunks)
    if not isinstance(data, dict) or data.get("model_name") != expected_model:
        raise ValueError("A-RAG index embedding model does not match its metadata")
    if data.get("chunks") != lookup:
        raise ValueError("A-RAG index chunk provenance differs from chunks.json")
    sentences = data.get("sentences")
    membership = data.get("sentence_to_chunk")
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("A-RAG index has no sentences")
    if not isinstance(membership, list) or len(membership) != len(sentences):
        raise ValueError("A-RAG sentence membership length is invalid")
    for sentence, view_id in zip(sentences, membership):
        if not isinstance(view_id, str) or view_id not in lookup:
            raise ValueError("A-RAG sentence refers to an unknown view id")
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError("A-RAG index contains an empty sentence")
        if sentence not in lookup[view_id]["text"]:
            raise ValueError(f"A-RAG sentence does not belong to view {view_id}")
    if set(membership) != set(lookup):
        raise ValueError("Some caption views have no indexed sentences")
    try:
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("A-RAG index embeddings must be a numeric matrix") from exc
    if embeddings.ndim != 2 or embeddings.shape[0] != len(sentences) or embeddings.shape[1] < 1:
        raise ValueError("A-RAG embedding dimensions do not match its sentences")
    if not np.isfinite(embeddings).all():
        raise ValueError("A-RAG index contains non-finite embeddings")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("A-RAG index contains invalid or zero-length embeddings")
    return lookup, sentences, membership, embeddings / norms


def build_index(
    chunks: list[dict],
    output_dir: Path,
    arag_root: Path,
    model_name: str,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict:
    """Invoke the actual local A-RAG builder, preserving chunk/view metadata.

    The caller should pass a fresh staging directory and install it atomically
    after this function returns. Returned metadata contains no staging paths.
    """
    _validate_chunks(chunks)
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("An embedding model name is required")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Embedding batch size must be a positive integer")
    script = Path(arag_root).expanduser().resolve() / "scripts" / "build_index.py"
    if not script.is_file():
        raise FileNotFoundError(f"Local A-RAG builder not found: {script}")
    script_hash = _sha256(script)
    spec = importlib.util.spec_from_file_location(f"_caption_arag_builder_{script_hash[:12]}", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the local A-RAG builder: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for chunk in chunks:
        if not module.split_sentences(chunk["text"]):
            raise ValueError(
                f"A-RAG cannot index view {chunk['id']}: its sentence splitter "
                "requires a sentence longer than 10 characters"
            )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "chunks.json"
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    module.build_index(
        chunks_file=str(chunks_path),
        output_dir=str(output_dir),
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    index_path = output_dir / "sentence_index.pkl"
    # This pickle was just generated by the explicitly selected local builder.
    with index_path.open("rb") as stream:
        data = pickle.load(stream)
    lookup, sentences, _, embeddings = _validate_index(data, chunks, model_name)
    metadata = {
        "adapter_version": _ADAPTER_VERSION,
        "builder": "A-RAG/scripts/build_index.py",
        "builder_sha256": script_hash,
        "model_name": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "chunk_count": len(lookup),
        "sentence_count": len(sentences),
        "chunks_sha256": _sha256(chunks_path),
        "index_sha256": _sha256(index_path),
        "retrieval": "max_sentence_cosine+bm25+rrf60",
    }
    (output_dir / "adapter.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metadata


class CaptionRetriever:
    """Retrieve views from a trusted local A-RAG index without a 20-view cap."""

    def __init__(self, index_dir: Path, device: str = "cpu"):
        import numpy as np

        self.index_dir = Path(index_dir)
        self.metadata = json.loads((self.index_dir / "adapter.json").read_text(encoding="utf-8"))
        if self.metadata.get("adapter_version") != _ADAPTER_VERSION:
            raise ValueError("Unsupported A-RAG caption adapter version; rebuild the index")
        for name, key in (("chunks.json", "chunks_sha256"), ("sentence_index.pkl", "index_sha256")):
            if _sha256(self.index_dir / name) != self.metadata.get(key):
                raise ValueError(f"A-RAG artifact integrity check failed for {name}; rebuild the index")
        self.model_name = self.metadata.get("model_name")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("A-RAG metadata has no embedding model")
        chunks = json.loads((self.index_dir / "chunks.json").read_text(encoding="utf-8"))
        with (self.index_dir / "sentence_index.pkl").open("rb") as stream:
            data = pickle.load(stream)
        self.chunks, self.sentences, self.sentence_to_chunk, self.embeddings = _validate_index(
            data, chunks, self.model_name
        )
        for key, actual in (
            ("chunk_count", len(self.chunks)),
            ("sentence_count", len(self.sentences)),
            ("embedding_dimension", self.embeddings.shape[1]),
        ):
            if self.metadata.get(key) != actual:
                raise ValueError(f"A-RAG metadata {key} does not match its index")
        self.model = _load_model(self.model_name, device)
        if self.model.get_sentence_embedding_dimension() != self.embeddings.shape[1]:
            raise ValueError("Query model dimension does not match the A-RAG index; rebuild the index")
        self.view_ids = sorted(self.chunks)
        view_positions = {view_id: position for position, view_id in enumerate(self.view_ids)}
        self._sentence_views = np.asarray([view_positions[v] for v in self.sentence_to_chunk], dtype=np.intp)
        self._view_sentence_indices = {view_id: [] for view_id in self.view_ids}
        for position, view_id in enumerate(self.sentence_to_chunk):
            self._view_sentence_indices[view_id].append(position)
        self._term_counts = {view_id: Counter(_tokens(self.chunks[view_id]["text"])) for view_id in self.view_ids}
        self._lengths = {view_id: sum(counts.values()) for view_id, counts in self._term_counts.items()}
        self._average_length = sum(self._lengths.values()) / len(self.view_ids) or 1.0
        self._document_frequency = Counter(term for counts in self._term_counts.values() for term in counts)

    def _lexical_scores(self, query: str) -> dict[str, float]:
        # BM25 is an adapter enhancement over A-RAG's literal substring counts.
        terms = set(_tokens(query))
        total = len(self.view_ids)
        result = {}
        for view_id in self.view_ids:
            counts = self._term_counts[view_id]
            norm = 1.2 * (0.25 + 0.75 * self._lengths[view_id] / self._average_length)
            score = 0.0
            for term in sorted(terms):
                count = counts.get(term, 0)
                if count:
                    frequency = self._document_frequency[term]
                    idf = math.log1p((total - frequency + 0.5) / (frequency + 0.5))
                    score += idf * count * 2.2 / (count + norm)
            result[view_id] = score
        return result

    def search(self, query: str, top_k: int = 100) -> list[dict]:
        """Return the same ranking as a one-element ``search_many`` call."""
        return self.search_many([query], top_k=top_k, batch_size=1)[0]

    def search_many(
        self, queries: Sequence[str], top_k: int = 100, batch_size: int = 128
    ) -> list[list[dict]]:
        """Retrieve multiple queries in input order, encoding each exactly once.

        Embeddings are produced in batches; score matrices use smaller batches
        when needed to stay within 64 MiB (or one full sentence-score row).
        Pooling, BM25, RRF, tie-breaking, and matched sentences are shared with
        ``search``. Results have the same per-query schema. Empty input returns
        an empty list without invoking the model. Duplicate queries are kept.
        """
        import numpy as np

        if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
            raise ValueError("queries must be a sequence of non-empty strings")
        if any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("Search query cannot be empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        score_batch_size = max(1, _MAX_SCORE_BYTES // (len(self.sentences) * np.dtype(np.float32).itemsize))
        results = []
        for start in range(0, len(queries), batch_size):
            batch = list(queries[start : start + batch_size])
            encoded = np.asarray(
                self.model.encode(batch, batch_size=batch_size, normalize_embeddings=True),
                dtype=np.float32,
            )
            if encoded.shape != (len(batch), self.embeddings.shape[1]) or not np.isfinite(encoded).all():
                raise ValueError("Query embedding dimension or values do not match the A-RAG index")
            norms = np.linalg.norm(encoded, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError("Query embedding is zero or invalid")
            encoded = encoded / norms
            for offset in range(0, len(batch), score_batch_size):
                vectors = encoded[offset : offset + score_batch_size]
                # Use the same reduction order for one and many queries. This
                # avoids BLAS GEMV/GEMM differences changing near-tie ranks.
                scores = np.einsum("sd,qd->qs", self.embeddings, vectors, optimize=False)
                for row, sentence_scores in enumerate(scores):
                    results.append(self._rank_sentences(batch[offset + row], sentence_scores, top_k))
                del scores
        return results

    def _rank_sentences(self, query: str, sentence_scores, top_k: int) -> list[dict]:
        """Pool all sentences, then apply the common per-view hybrid ranking."""
        import numpy as np

        pooled = np.full(len(self.view_ids), -np.inf, dtype=np.float32)
        np.maximum.at(pooled, self._sentence_views, sentence_scores)
        semantic = dict(zip(self.view_ids, map(float, pooled)))
        lexical = self._lexical_scores(query)
        semantic_order = sorted(self.view_ids, key=lambda view_id: (-semantic[view_id], view_id))
        lexical_order = sorted(
            (view_id for view_id in self.view_ids if lexical[view_id] > 0),
            key=lambda view_id: (-lexical[view_id], view_id),
        )
        fused = {view_id: 1.0 / (60 + rank) for rank, view_id in enumerate(semantic_order, 1)}
        for rank, view_id in enumerate(lexical_order, 1):
            fused[view_id] += 1.0 / (60 + rank)
        ranked = sorted(self.view_ids, key=lambda view_id: (-fused[view_id], view_id))[:top_k]
        results = []
        for view_id in ranked:
            indices = sorted(self._view_sentence_indices[view_id], key=lambda i: (-float(sentence_scores[i]), i))
            matched = list(dict.fromkeys(self.sentences[i] for i in indices))[:5]
            results.append({
                "view_id": view_id,
                "score": fused[view_id],
                "semantic_score": semantic[view_id],
                "lexical_score": lexical[view_id],
                "matched_sentences": matched,
            })
        return results
