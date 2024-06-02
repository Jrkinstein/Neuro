from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple

# Функция для токенизации текста
def tokenize(text: str) -> List[str]:
    return text.lower().split()

# Функция для построения словаря и преобразования текста в индексы
def build_vocab(texts: List[str]) -> Tuple[dict, dict]:
    vectorizer = CountVectorizer(tokenizer=tokenize)
    vectorizer.fit(texts)
    word_to_index = vectorizer.vocabulary_
    index_to_word = {i: word for word, i in word_to_index.items()}
    return word_to_index, index_to_word

def text_to_indices(text: str, word_to_index: dict) -> List[int]:
    tokens = tokenize(text)
    return [word_to_index.get(token, 0) for token in tokens]
