from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

# ---------- ABCs ----------
class Preprocessor(ABC):
    @abstractmethod
    def clean(self, text: str) -> str: ...

class Embedder(ABC):
    @abstractmethod
    def fit(self, corpus: Iterable[str]) -> None: ...
    @abstractmethod
    def encode(self, text: str) -> List[float]: ...

class VectorIndex(ABC):
    @abstractmethod
    def add(self, doc_id: str, vector: List[float]): ...
    @abstractmethod
    def search(self, query_vec: List[float], k: int = 3) -> List[Tuple[str, float]]: ...

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]: ...

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, items: List[Tuple[str, float]]) -> List[Tuple[str, float]]: ...

class Answerer(ABC):
    @abstractmethod
    def answer(self, query: str, context_docs: List[str]) -> str: ...
