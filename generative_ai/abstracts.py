from abc import ABC, abstractmethod

# Abstract base class for Generative AI clients
class GenerativeAIClient(ABC):
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        ...