"""Build a modular, extensible *text-retrieval* mini‑pipeline
that runs fully offline and showcases real OOP techniques you
can reuse in agent or RAG systems.
"""

from inputs import CORPUS
from observers import EventBus
from mixins import LogMixin, TimeMixin
from abcs import Preprocessor, Embedder, VectorIndex, Retriever, Reranker, Answerer
from utilities import tokenize, cosine


# from __future__ import annotations
from typing import Dict, List, Iterable, Tuple, Callable, Any
from functools import lru_cache
from collections import Counter

# ---- Preprocessors ----
class BasicPreprocessor(Preprocessor):
    STOPWORDS = {"the","a","an","and","or","to","of","in","is","are","that","with","by"}
    def clean(self, text: str) -> str:
        words = tokenize(text)
        filtered = [w for w in words if w not in self.STOPWORDS]
        return " ".join(filtered)

class NoopPreprocessor(Preprocessor):
    def clean(self, text: str) -> str:
        return text

# ---- TF-IDF Embedder (from scratch) ----
class TfidfEmbedder(Embedder, LogMixin):
    def __init__(self):
        self.vocab: Dict[str,int] = {}
        self.idf: List[float] = []
    def fit(self, corpus: Iterable[str]) -> None:
        docs = [tokenize(doc) for doc in corpus]
        vocab = {}
        for doc in docs:
            for t in doc:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab
        N = len(docs)
        df = [0]*len(vocab)
        for doc in docs:
            seen = set(doc)
            for t in seen:
                df[vocab[t]] += 1
        import math
        self.idf = [math.log((N+1)/(df_i+1))+1 for df_i in df]
        self.log(f"Fitted TF-IDF on {N} docs, |vocab|={len(vocab)}")
    def encode(self, text: str) -> List[float]:
        toks = tokenize(text)
        tf = Counter(toks)
        vec = [0.0]*len(self.vocab)
        for t, c in tf.items():
            if t in self.vocab:
                idx = self.vocab[t]
                vec[idx] = (c / len(toks)) * self.idf[idx]
        return vec

# ---- Adapter example ----
class LegacyCountVectorizer:
    # Pretend this is a third‑party component with a different API.
    def fit_corpus(self, docs: List[str]):
        self.vocab = {}
        for d in docs:
            for t in tokenize(d):
                self.vocab.setdefault(t, len(self.vocab))
    def vectorize(self, text: str) -> List[float]:
        v = [0.0]*len(self.vocab)
        for t in tokenize(text):
            if t in self.vocab:
                v[self.vocab[t]] += 1.0
        return v

class CountVectorAdapter(Embedder):
    def __init__(self, legacy: LegacyCountVectorizer):
        self.legacy = legacy
    def fit(self, corpus: Iterable[str]) -> None:
        self.legacy.fit_corpus(list(corpus))
    def encode(self, text: str) -> List[float]:
        return self.legacy.vectorize(text)

# ---- Vector index ----
class BruteForceIndex(VectorIndex):
    def __init__(self):
        self.vectors: Dict[str, List[float]] = {}
    def add(self, doc_id: str, vector: List[float]):
        self.vectors[doc_id] = vector
    def search(self, query_vec: List[float], k: int = 3) -> List[Tuple[str, float]]:
        scores = []
        for doc_id, v in self.vectors.items():
            similarity = cosine(query_vec, v)
            scores.append((doc_id, similarity))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

# ---- Retriever ----
class SimpleRetriever(Retriever):
    def __init__(self, pre: Preprocessor, emb: Embedder, index: VectorIndex, bus: EventBus | None = None):
        self.pre, self.emb, self.index, self.bus = pre, emb, index, bus
    @lru_cache(maxsize=256)
    def _encode_cached(self, text: str) -> List[float]:
        return self.emb.encode(text)
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        clean = self.pre.clean(query)
        if self.bus: self.bus.publish("query.start", {"query": query, "clean": clean})
        qv = self._encode_cached(clean)
        hits = self.index.search(qv, k=k)
        if self.bus: self.bus.publish("query.done", {"query": query, "hits": hits})
        return hits

# ---- Reranker (optional; here just identity passthrough) ----
class IdentityReranker(Reranker):
    def rerank(self, query: str, items: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        return items

# ---- Answerer ----
class TemplateAnswerer(Answerer):
    def __init__(self, corpus: Dict[str,str]):
        self.corpus = corpus
    def answer(self, query: str, context_docs: List[str]) -> str:
        bullets = "\n".join(f"- {doc[:160]}..." for doc in context_docs)
        return f"Answering: '{query}'\n\nTop context:\n{bullets}"



BUS = EventBus()
BUS.subscribe("query.start", lambda e: print(f"[EVENT] query.start clean='{e['clean']}'"))
BUS.subscribe("query.done",  lambda e: print(f"[EVENT] query.done hits={e['hits']}"))

pre = BasicPreprocessor()
emb = TfidfEmbedder()
emb.fit(CORPUS.values())

index = BruteForceIndex()
for doc_id, text in CORPUS.items():
    index.add(doc_id, emb.encode(pre.clean(text)))

retriever = SimpleRetriever(pre, emb, index, bus=BUS)
reranker  = IdentityReranker()
answerer  = TemplateAnswerer(CORPUS)

print("Index ready. Documents:", list(CORPUS.keys()))



def run_pipeline(query: str, k: int = 3):
    hits = retriever.retrieve(query, k=k)
    doc_ids = [doc_id for doc_id, _ in reranker.rerank(query, hits)]
    contexts = [CORPUS[i] for i in doc_ids]
    return answerer.answer(query, contexts)

print(run_pipeline("How does TF-IDF work and why cosine similarity?"))

