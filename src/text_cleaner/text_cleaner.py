# Practical session to apply clean-code patterns in small utilities.

from dataclasses import dataclass
from typing import Iterable

@dataclass
class TextCleaner:
    stopwords: Iterable[str]

    # Remove stopwords from the input text
    def clean(self, text: str) -> str:
        words = text.split()
        cleaned_words = [word for word in words if word.lower() not in self.stopwords]
        return ' '.join(cleaned_words)
    
    # Normalize text by removing punctuation and filtering short words
    def normalize(self, text: str, min_len: int = 1, strip_punct: bool = True) -> str:
        if strip_punct:
            import string
            # Remove punctuation
            # text = ''.join(char for char in text if char not in string.punctuation)
            text = ''.join(filter(lambda x: x not in string.punctuation, text))
        words = text.split()
        # Filter words by minimum length
        # normalized_words = [word for word in words if len(word) >= min_len]
        normalized_words = filter(lambda x: len(x) >= min_len, words)
        return ' '.join(normalized_words)

def main():
    cleaner = TextCleaner(stopwords={"the", "is", "a"})

    response1 = cleaner.clean("This is a simple Example for the Lab")
    print(response1)  # Output: "This simple Example for Lab"

    response2 = cleaner.normalize("Hello, World! 123 is a nice number.", min_len=2)
    print(response2)  # Output: "Hello World 123 is nice number"

if __name__ == "__main__":
    main()