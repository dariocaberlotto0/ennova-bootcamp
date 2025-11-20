import unittest
import numpy as np
from typing import List
from text_retrival import tokenize, cosine, retrival2
from observers import EventBus

def make_embedder(kind: str) -> Embedder:
    kind = kind.lower()
    if kind == "tfidf":
        return TfidfEmbedder()
    elif kind == "count":
        return CountVectorAdapter(LegacyCountVectorizer())
    raise ValueError(f"Unknown embedder kind: {kind}")

# Demo swap:
emb2 = make_embedder("count")
emb2.fit(CORPUS.values())
index2 = BruteForceIndex()
for i, t in CORPUS.items():
    index2.add(i, emb2.encode(pre.clean(t)))
retriever2 = SimpleRetriever(pre, emb2, index2)
print([x for x in retriever2.retrieve("mixins for logging and timing", k=3)])

BUS = EventBus()
BUS.subscribe("tests.start", lambda e: print(f"[EVENT] tests.start"))
BUS.subscribe("tests.done",  lambda e: print(f"[EVENT] tests.done"))

class TestMiniPipeline(unittest.TestCase):
    def test_tokenize(self):
        test1 = tokenize("Hello, world! This is a test to see if tokenize method works. or. to fix, it.")
        self.assertEqual(test1, ["hello", "world", "this", "is", "a", "test", "to", "see", "if", "tokenize", "method", "works", "or", "to", "fix", "it"])
    def test_cosine_unit(self):
        A: List[float] = [1, 2, 3]
        B: List[float] = [4, 5, 6]
        test2 = cosine(A, B)
        cos_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
        self.assertEqual(test2, cos_sim)

    def test_retrieval_nonempty(self):
        test3 = [x for x in retriever2.retrieve("mixins for logging and timing", k=3)]
        self.assertTrue(len(test3) > 0)

    BUS.publish("tests.start", {})
    unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestMiniPipeline));
    BUS.publish("tests.done", {})