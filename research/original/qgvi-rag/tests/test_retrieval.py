"""Offline A-RAG adapter tests: synthetic embeddings, no model/API downloads."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
import types

import numpy as np
import pytest

from caption_rag import retrieval


class FakeModel:
    def __init__(self, model_name="test/embedding", device="cpu", dimension=2):
        self.model_name = model_name
        self.device = device
        self.dimension = dimension

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        return np.tile([1.0, 0.0], (len(texts), 1)).astype(np.float32)


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(retrieval, "_load_model", FakeModel)


def make_index(tmp_path, rows, embeddings=None, mutate=None):
    """Rows are (view ID, list of sentences); artifacts stay local to this test."""
    chunks = [
        {"id": view_id, "text": ". ".join(sentences), "view": {"image_path": f"{view_id}.png"}}
        for view_id, sentences in rows
    ]
    sentences = [sentence for _, values in rows for sentence in values]
    membership = [view_id for view_id, values in rows for _ in values]
    data = {
        "sentences": sentences,
        "sentence_to_chunk": membership,
        "chunks": {chunk["id"]: json.loads(json.dumps(chunk)) for chunk in chunks},
        "model_name": "test/embedding",
        "embeddings": np.asarray(embeddings if embeddings is not None else [[1.0, 0.0]] * len(sentences)),
    }
    if mutate:
        mutate(data)
    chunks_path = tmp_path / "chunks.json"
    index_path = tmp_path / "sentence_index.pkl"
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")
    with index_path.open("wb") as stream:
        pickle.dump(data, stream)
    metadata = {
        "adapter_version": 1,
        "model_name": "test/embedding",
        "embedding_dimension": 2,
        "chunk_count": len(chunks),
        "sentence_count": len(sentences),
        "chunks_sha256": retrieval._sha256(chunks_path),
        "index_sha256": retrieval._sha256(index_path),
    }
    (tmp_path / "adapter.json").write_text(json.dumps(metadata), encoding="utf-8")
    return tmp_path


def test_more_than_twenty_views_with_many_sentences_per_view(tmp_path):
    # An upstream top_k*3 sentence cutoff would contain only the first view.
    crowded = [f"The dominant view contains descriptive sentence {i}" for i in range(150)]
    rows = [("v000", crowded)] + [(f"v{i:03}", [f"Distinct reference building number {i}"]) for i in range(1, 41)]
    embeddings = [[1, 0]] * 150 + [[0.8, 0.6]] * 40
    retriever = retrieval.CaptionRetriever(make_index(tmp_path, rows, embeddings))
    results = retriever.search("unmentioned", top_k=35)
    assert len(results) == 35
    assert len({result["view_id"] for result in results}) == 35
    assert results[0]["view_id"] == "v000"
    assert len(results[0]["matched_sentences"]) == 5
    assert retriever.model_name == "test/embedding"
    assert retriever.chunks["v001"]["view"]["image_path"] == "v001.png"


def test_sentence_scores_pool_max_before_ranking(tmp_path):
    rows = [("a", ["An ordinary first sentence", "A highly relevant second sentence"]), ("b", ["The other reference sentence"])]
    retriever = retrieval.CaptionRetriever(make_index(tmp_path, rows, [[0, 1], [1, 0], [0.8, 0.6]]))
    results = retriever.search("unmentioned", top_k=2)
    assert [result["view_id"] for result in results] == ["a", "b"]
    assert results[0]["semantic_score"] == pytest.approx(1.0)
    assert results[0]["matched_sentences"][0] == "A highly relevant second sentence"
    assert results[0]["score"] == pytest.approx(1 / 61)


def test_fusion_is_deterministic_and_combines_lexical_with_semantic(tmp_path):
    rows = [("z", ["A red brick warehouse beside a tree"]), ("a", ["A gray concrete building beside a tree"])]
    retriever = retrieval.CaptionRetriever(make_index(tmp_path, rows))
    # Semantic ties are sorted by ID, while the query keyword ranks z first.
    results = retriever.search("please find the red view", top_k=2)
    assert [result["view_id"] for result in results] == ["z", "a"]
    assert results[0]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert results[0]["lexical_score"] > 0
    assert results[1]["lexical_score"] == 0
    assert results == retriever.search("please find the red view", top_k=2)
    assert [result["view_id"] for result in retriever.search("unknownterm")] == ["a", "z"]


def test_stop_words_do_not_dominate_and_cjk_terms_match(tmp_path):
    rows = [("a", ["The view is showing the scene with a building"]), ("z", ["红色建筑旁边有一棵树和一个蓝色路牌"])]
    retriever = retrieval.CaptionRetriever(make_index(tmp_path, rows))
    assert all(result["lexical_score"] == 0 for result in retriever.search("please show the view"))
    results = retriever.search("红色建筑")
    assert results[0]["view_id"] == "z"
    assert results[0]["lexical_score"] > 0


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda d: d["sentence_to_chunk"].__setitem__(0, "unknown"), "unknown view"),
        (lambda d: d["sentences"].__setitem__(0, "An invented sentence"), "does not belong"),
        (lambda d: d["chunks"]["a"].__setitem__("view", {}), "provenance"),
        (lambda d: d.__setitem__("model_name", "other-model"), "model"),
        (lambda d: d.__setitem__("embeddings", np.array([[float("nan"), 0]])), "non-finite"),
        (lambda d: d.__setitem__("embeddings", np.array([[0, 0]])), "zero-length"),
        (lambda d: d.__setitem__("embeddings", np.array([1, 0])), "dimensions"),
        (lambda d: d.__setitem__("sentence_to_chunk", []), "membership length"),
    ],
)
def test_rejects_invalid_index_membership_and_embeddings(tmp_path, mutation, message):
    index = make_index(tmp_path, [("a", ["A red brick building beside a tree"])], mutate=mutation)
    with pytest.raises(ValueError, match=message):
        retrieval.CaptionRetriever(index)


def test_rejects_corrupt_pickle_before_loading(tmp_path, monkeypatch):
    index = make_index(tmp_path, [("a", ["A red brick building beside a tree"])])
    (index / "sentence_index.pkl").write_bytes(b"not a pickle")
    monkeypatch.setattr(pickle, "load", lambda _: pytest.fail("Unpickled a corrupt artifact"))
    with pytest.raises(ValueError, match="integrity"):
        retrieval.CaptionRetriever(index)


def test_rejects_query_model_and_query_dimension_mismatch(tmp_path, monkeypatch):
    index = make_index(tmp_path, [("a", ["A red brick building beside a tree"])])
    monkeypatch.setattr(retrieval, "_load_model", lambda *args: FakeModel(dimension=3))
    with pytest.raises(ValueError, match="Query model dimension"):
        retrieval.CaptionRetriever(index)
    monkeypatch.setattr(retrieval, "_load_model", FakeModel)
    retriever = retrieval.CaptionRetriever(index)
    monkeypatch.setattr(retriever.model, "encode", lambda *args, **kwargs: [[1, 0, 0]])
    with pytest.raises(ValueError, match="Query embedding dimension"):
        retriever.search("building")


@pytest.mark.parametrize("embedding", [[[float("inf"), 0]], [[0, 0]]])
def test_rejects_invalid_query_embeddings(tmp_path, monkeypatch, embedding):
    index = make_index(tmp_path, [("a", ["A red brick building beside a tree"])])
    retriever = retrieval.CaptionRetriever(index)
    monkeypatch.setattr(retriever.model, "encode", lambda *args, **kwargs: embedding)
    with pytest.raises(ValueError, match="Query embedding"):
        retriever.search("building")


def test_rejects_views_missing_from_sentence_membership(tmp_path):
    rows = [("a", ["The first view shows a red house"]), ("b", ["The second view shows a blue house"])]

    def remove_last_sentence(data):
        data["sentences"].pop()
        data["sentence_to_chunk"].pop()
        data["embeddings"] = data["embeddings"][:1]

    index = make_index(tmp_path, rows, mutate=remove_last_sentence)
    with pytest.raises(ValueError, match="no indexed sentences"):
        retrieval.CaptionRetriever(index)


@pytest.mark.parametrize("query,top_k", [("", 5), ("building", 0), ("building", -1), ("building", True), ("building", 1.2)])
def test_rejects_invalid_search_arguments(tmp_path, query, top_k):
    index = make_index(tmp_path, [("a", ["A red brick building beside a tree"])])
    with pytest.raises(ValueError):
        retrieval.CaptionRetriever(index).search(query, top_k=top_k)


def test_actual_local_arag_builder_preserves_metadata_without_model_download(tmp_path, monkeypatch):
    arag_root = Path.home() / "others" / "arag-upstream"
    if not (arag_root / "scripts" / "build_index.py").is_file():
        pytest.skip("The optional local A-RAG checkout is not installed")
    fake_package = types.ModuleType("sentence_transformers")
    fake_package.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_package)
    chunks = [{"id": "v1", "text": "A red building stands beside a large tree.", "view": {"c2w": [[1, 0], [0, 1]]}}]
    metadata = retrieval.build_index(chunks, tmp_path / "stage", arag_root, "test/embedding")
    assert metadata["builder_sha256"] == retrieval._sha256(arag_root / "scripts" / "build_index.py")
    assert metadata["embedding_dimension"] == 2
    assert metadata["chunk_count"] == metadata["sentence_count"] == 1
    assert str(tmp_path) not in json.dumps(metadata)
    installed = tmp_path / "installed"
    (tmp_path / "stage").rename(installed)
    retriever = retrieval.CaptionRetriever(installed)
    assert retriever.chunks["v1"]["view"] == chunks[0]["view"]
    assert retriever.search("red building")[0]["view_id"] == "v1"


def test_build_rejects_duplicate_ids_before_upstream_loading(tmp_path):
    chunks = [{"id": "same", "text": "This is a sufficiently long sentence"}] * 2
    with pytest.raises(ValueError, match="Duplicate"):
        retrieval.build_index(chunks, tmp_path, tmp_path / "missing", "test/embedding")


def varied_retriever(tmp_path, monkeypatch):
    rows = [
        ("a", ["A red brick warehouse beside a tree", "A staircase leads to the roof"]),
        ("b", ["A blue building behind a metal fence"]),
        ("c", ["红色建筑旁边有一棵树和一个蓝色路牌"]),
        ("d", ["A gray concrete building with a staircase"]),
    ]
    retriever = retrieval.CaptionRetriever(
        make_index(tmp_path, rows, [[0.1, 0.9], [0.9, 0.1], [-0.5, 0.2], [0.4, -0.8], [0.9, 0.1]])
    )
    calls = []

    def encode(texts, **kwargs):
        calls.append((list(texts), kwargs))
        vectors = {"red tree": [0.65, 0.33], "blue fence": [-0.3, 0.5],
                   "红色建筑": [0.4, -0.7], "staircase": [0.9, 0.11], "unknown": [0.1, 0.1]}
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)

    monkeypatch.setattr(retriever.model, "encode", encode)
    return retriever, calls


@pytest.mark.parametrize("batch_size", [1, 2, 3, 128])
def test_bulk_search_matches_single_rankings_scores_and_sentence_evidence(tmp_path, monkeypatch, batch_size):
    retriever, calls = varied_retriever(tmp_path, monkeypatch)
    queries = ["red tree", "blue fence", "红色建筑", "staircase", "unknown", "red tree"]
    expected = [retriever.search(query, top_k=3) for query in queries]
    calls.clear()
    actual = retriever.search_many(queries, top_k=3, batch_size=batch_size)
    assert actual == expected
    assert actual[0] == actual[-1]
    assert [query for batch, _ in calls for query in batch] == queries
    assert len(calls) == (len(queries) + batch_size - 1) // batch_size
    assert all(len(batch) <= batch_size for batch, _ in calls)
    assert all(kwargs == {"batch_size": batch_size, "normalize_embeddings": True} for _, kwargs in calls)


def test_bulk_score_matrices_obey_memory_batch_limit(tmp_path, monkeypatch):
    retriever, calls = varied_retriever(tmp_path, monkeypatch)
    queries = ["red tree", "blue fence", "红色建筑", "staircase", "unknown"]
    expected = [retriever.search(query) for query in queries]
    calls.clear()
    # Force two score rows at a time while encoding the complete query batch.
    monkeypatch.setattr(retrieval, "_MAX_SCORE_BYTES", len(retriever.sentences) * 4 * 2)
    original_einsum = np.einsum
    sizes = []

    def tracked_einsum(subscripts, sentences, vectors, **kwargs):
        result = original_einsum(subscripts, sentences, vectors, **kwargs)
        sizes.append(result.shape)
        return result

    monkeypatch.setattr(np, "einsum", tracked_einsum)
    assert retriever.search_many(queries, batch_size=128) == expected
    assert len(calls) == 1
    assert sizes == [(2, 5), (2, 5), (1, 5)]


def test_bulk_empty_input_never_encodes_and_tuples_are_supported(tmp_path, monkeypatch):
    retriever, calls = varied_retriever(tmp_path, monkeypatch)
    assert retriever.search_many([]) == []
    assert calls == []
    assert retriever.search_many(("red tree",)) == [retriever.search("red tree")]


@pytest.mark.parametrize(
    "queries,kwargs",
    [
        ("red tree", {}),
        (None, {}),
        (["red tree", ""], {}),
        (["red tree", " \t\n"], {}),
        (["red tree", 42], {}),
        (["red tree"], {"top_k": 0}),
        (["red tree"], {"top_k": True}),
        (["red tree"], {"batch_size": 0}),
        (["red tree"], {"batch_size": -1}),
        (["red tree"], {"batch_size": True}),
        (["red tree"], {"batch_size": 1.5}),
    ],
)
def test_bulk_rejects_invalid_input_before_any_encoder_work(tmp_path, monkeypatch, queries, kwargs):
    retriever, calls = varied_retriever(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        retriever.search_many(queries, **kwargs)
    assert calls == []


@pytest.mark.parametrize(
    "encoded",
    [
        [[1, 0]],  # The encoder silently dropped the second query.
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0], [float("nan"), 0]],
        [[1, 0], [0, 0]],
    ],
)
def test_bulk_validates_every_query_embedding(tmp_path, monkeypatch, encoded):
    retriever, _ = varied_retriever(tmp_path, monkeypatch)
    monkeypatch.setattr(retriever.model, "encode", lambda *args, **kwargs: encoded)
    with pytest.raises(ValueError, match="Query embedding"):
        retriever.search_many(["red tree", "blue fence"])


def test_bulk_retains_more_than_twenty_views_and_max_sentence_pooling(tmp_path):
    rows = [("crowded", [f"A reference building sentence number {i}" for i in range(100)])]
    rows += [(f"v{i:03}", [f"A separate reference building number {i}"]) for i in range(30)]
    retriever = retrieval.CaptionRetriever(make_index(tmp_path, rows))
    results = retriever.search_many(["building", "unmentioned"], top_k=31, batch_size=2)
    assert [len(row) for row in results] == [31, 31]
    assert [len({candidate["view_id"] for candidate in row}) for row in results] == [31, 31]
    assert results == [retriever.search(query, top_k=31) for query in ["building", "unmentioned"]]
