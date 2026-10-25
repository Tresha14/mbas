"""TF-IDF + logistic / ridge regression baseline for the answer-quality model (no neural network).

Features: TF-IDF of the answer (word 1-2 grams + char 3-5 grams), TF-IDF of the question, and the cosine
similarity between question+key points and answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge


class TfidfBaseline:
    def __init__(self):
        self.vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=20000, sublinear_tf=True)
        self.vq = TfidfVectorizer(ngram_range=(1, 2), min_df=1)

    def _x(self, t: pd.DataFrame, fit: bool = False) -> sp.csr_matrix:
        a, q = t.transcript.fillna("").tolist(), t.question_text.fillna("").tolist()
        if fit:
            self.vw.fit(a + q)
            self.vc.fit(a)
            self.vq.fit(q)
        wa, wq = self.vw.transform(a), self.vw.transform(q)
        sim = np.asarray(wa.multiply(wq).sum(1))
        return sp.hstack([wa, self.vc.transform(a), self.vq.transform(q), sp.csr_matrix(sim * 5)]).tocsr()

    def fit(self, t: pd.DataFrame) -> "TfidfBaseline":
        x = self._x(t, fit=True)
        self.rel = Ridge(alpha=1.0).fit(x, t.relevance)
        self.q = LogisticRegression(max_iter=2000, C=4.0).fit(x, t.quality_1to5.astype(int))
        self.f = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(x, t.follow_up_needed.astype(int))
        self.r = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(x, t.follow_up_reason)
        return self

    def predict(self, t: pd.DataFrame) -> dict:
        x = self._x(t)
        return {"relevance": np.clip(self.rel.predict(x), 0, 1), "quality": self.q.predict(x),
                "quality_expected": self.q.predict_proba(x) @ self.q.classes_,
                "follow_up": self.f.predict_proba(x)[:, 1], "reason": self.r.predict(x)}
