"""Latent Dirichlet Allocation

Patrick Wang, 2021
"""
from typing import List

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import numpy as np
import random


def lda_gen(vocabulary: List[str], alpha: np.ndarray, beta: np.ndarray, xi: int) -> List[str]:
    #generate the length of the topic
    l = np.random.poisson(xi)

    #generate list of topics and topic distribution proportion from alpha
    x = np.random.dirichlet(alpha)
    t_list = [0,1,2]
    list_of_topics = []
    for i in range(l):
        random_number = random.choices(t_list, x)
        list_of_topics.append(random_number[0])
    #print(list_of_topics)
    
    #generate words given topic
    strings = []
    for i in list_of_topics:
        word = random.choices(vocabulary, beta[i])
        strings.append(word[0])
    return strings


def test():
    """Test the LDA generator."""
    vocabulary = [
        "bass", "pike", "deep", "tuba", "horn", "catapult",
    ]
    beta = np.array([
        [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.1, 0.0, 0.3, 0.3],
        [0.3, 0.0, 0.2, 0.3, 0.2, 0.0]
    ])
    alpha = np.array([0.2, 0.2, 0.2])
    xi = 50
    documents = [
        lda_gen(vocabulary, alpha, beta, xi)
        for _ in range(100)
    ]

    # Create a corpus from a list of texts
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    model = LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=3,
    )
    print(model.alpha)
    print(model.show_topics())


if __name__ == "__main__":
    test()