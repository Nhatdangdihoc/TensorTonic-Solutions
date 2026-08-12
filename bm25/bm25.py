import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    N = len(docs)
    avgdl = np.mean([len(doc) for doc in docs])

    # Document frequency: số document chứa term
    df = Counter()
    for doc in docs:
        for term in set(doc):
            df[term] += 1

    scores = []

    for doc in docs:
        tf = Counter(doc)
        dl = len(doc)

        score = 0.0

        for term in query_tokens:
            if term not in tf:
                continue

            # IDF
            idf = math.log(
                (N - df[term] + 0.5) / (df[term] + 0.5) + 1
            )

            freq = tf[term]

            # BM25 term score
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (
                1 - b + b * dl / avgdl
            )

            score += idf * numerator / denominator

        scores.append(score)

    return np.array(scores)