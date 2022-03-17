from typing import List
from gensim.models import Word2Vec
import numpy as np


def train_model(corpus_file: str, **hyperparameters) -> Word2Vec:
    """
    Accepts a file of the text corpus in LineSentence format (every line corresponds to a new sentence in the corpus)
    Trains a word2vec model with standard settings (plus the specified hyperparameters) and returns it
    """
    return Word2Vec(corpus_file=corpus_file, min_count=10, **hyperparameters)


def most_similar(word: str, n: int, model: Word2Vec) -> List[str]:
    """
    Returns the n most similar words to word in the model's vocabulary
    """
    return [word for word, _ in model.wv.most_similar(positive=[word], topn=n)]
