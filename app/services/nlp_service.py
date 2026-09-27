import time
import re
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import gensim
from gensim import corpora, models, similarities

class NLPService:
    @staticmethod
    def _simple_tokenize(text: str) -> List[str]:
        clean_text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [token.strip() for token in clean_text.split() if len(token.strip()) > 1]
        return tokens

    @classmethod
    def compute_tfidf_similarity(
        cls,
        query: str,
        documents: List[Dict[str, str]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        doc_texts = [doc["text"] for doc in documents]
        tokenized_docs = [cls._simple_tokenize(text) for text in doc_texts]
        
        dictionary = corpora.Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_docs]
        
        if len(dictionary) == 0:
            return [
                {
                    "doc_id": doc["doc_id"],
                    "text": doc["text"],
                    "similarity_score": 0.0
                }
                for doc in documents[:top_k]
            ]

        tfidf_model = models.TfidfModel(corpus)
        index = similarities.MatrixSimilarity(tfidf_model[corpus], num_features=len(dictionary))

        query_tokens = cls._simple_tokenize(query)
        query_bow = dictionary.doc2bow(query_tokens)
        query_tfidf = tfidf_model[query_bow]

        sims = index[query_tfidf]
        ranked_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            results.append({
                "doc_id": documents[idx]["doc_id"],
                "text": documents[idx]["text"],
                "similarity_score": round(float(sims[idx]), 4)
            })

        return results
