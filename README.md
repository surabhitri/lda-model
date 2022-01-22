# lda-model

## Q1
Implemented the latent Dirichlet allocation (LDA) model to generate
a corpus from a given set of parameters. Built a function lda_gen()
that takes four arguments:
1. vocabulary - list (of length V ) of strings
2. alpha - topic distribution parameter vector, numpy array of size (k,)
3. beta - topic-word matrix, numpy array of size (k, V )
4. xi - Poisson parameter (scalar) for document size distribution
and returns:
1. words - list of words (strings) in a document

## Q2
Use the provided script to generate a corpus of documents and apply LDA parameter inference with gensim’s solver. Showed the inferred beta vectors
and indicated how they map to the true topics above.
