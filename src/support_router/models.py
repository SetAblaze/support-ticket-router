"""Конфигурации моделей, обучение и откалиброванный RouterBundle."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, normalize
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from .config import SEED
from .text import preprocess, preprocess_batch


def make_model(name="lr_word (baseline)", seed=SEED):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                           lowercase=False, sublinear_tf=True)
    feat = word
    if name.startswith('semantic_'):
        from .semantic_encoder import FrozenSentenceEncoder
        feat = FrozenSentenceEncoder()
    if name == "linsvc_wc_C1":
        feat = FeatureUnion([
            ("w", word),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                  max_features=50000, lowercase=False, sublinear_tf=True))])
    if name.startswith("lr_") or name in ['semantic_lr_C1','semantic_aug_lr_C1']:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed,
                                 solver="lbfgs")
    elif name.startswith("sgd_"):
        alpha = {"sgd_w_a1e-4": 1e-4, "sgd_w_a5e-5": 5e-5, "sgd_w_a2e-5": 2e-5}[name]
        clf = SGDClassifier(loss="modified_huber", alpha=alpha, max_iter=3000,
                            class_weight="balanced", random_state=seed)
    elif name == "nb_word":
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB(alpha=.1)
    elif name == 'semantic_svc_C1':
        clf = LinearSVC(C=1.,class_weight='balanced',dual='auto',max_iter=10000,random_state=seed)
    else:
        c = {"linsvc_w_C0.5": .5, "linsvc_w_C1": 1., "linsvc_w_C3": 3.,
             "linsvc_wc_C1": 1.}[name]
        clf = LinearSVC(C=c, class_weight="balanced", dual="auto", max_iter=10000,
                        random_state=seed)
    return Pipeline([("preprocess", FunctionTransformer(preprocess_batch, validate=False)),
                     ("features", feat), ("clf", clf)])

CONFIG_NAMES = ["lr_word (baseline)", "sgd_w_a1e-4", "sgd_w_a5e-5", "sgd_w_a2e-5",
                "linsvc_w_C0.5", "linsvc_w_C1", "linsvc_w_C3", "linsvc_wc_C1",
                "semantic_lr_C1", "semantic_svc_C1", "semantic_aug_lr_C1"]

def fit_model(name, frame, leave_intent_out=False):
    texts, labels = frame.text.tolist(), frame.category.tolist()
    fit_params = {}
    if name == 'semantic_aug_lr_C1':
        from .category_examples import eligible_examples
        additions = eligible_examples(frame,leave_intent_out=leave_intent_out)
        if additions:
            texts += [x['text'] for x in additions]
            labels += [x['category'] for x in additions]
            fit_params['clf__sample_weight'] = np.r_[np.ones(len(frame)),[x['sample_weight'] for x in additions]]
    with threadpool_limits(limits=1):
        return make_model(name).fit(texts, labels, **fit_params)

def checked_proba(model, texts):
    if not hasattr(model, "predict_proba"):
        raise ValueError("MODEL_NO_PROBA")
    classes = np.asarray(model.classes_)
    p = np.asarray(model.predict_proba(texts), dtype=float)
    if p.shape != (len(texts), len(classes)) or len(set(classes)) != len(classes):
        raise ValueError("MODEL_BAD_PROBA_SHAPE")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError("MODEL_INVALID_PROBA")
    return p

@dataclass
class RouterBundle:
    model: Any
    base_model: Any
    reference_matrix: Any
    model_name: str
    policy: dict = field(default_factory=dict)
    model_id: str = field(default_factory=lambda: "router-" + uuid.uuid4().hex[:16])
    metadata: dict = field(default_factory=dict)
    reference_texts: Any = None
    reference_categories: Any = None
    novelty_vectorizer: Any = None
    novelty_matrix: Any = None

    @property
    def classes_(self):
        return self.model.classes_

    def predict_proba(self, texts):
        return self.model.predict_proba(list(texts))

    def predict(self, texts):
        p = checked_proba(self, list(texts))
        return np.asarray(self.classes_)[p.argmax(axis=1)]

    def similarity(self, texts):
        texts = list(texts)
        if not texts:
            return np.empty(0, dtype=float)
        if self.novelty_vectorizer is not None and self.novelty_matrix is not None:
            x = normalize(self.novelty_vectorizer.transform(texts),norm='l2',copy=False)
            reference = self.novelty_matrix
        else:
            x = normalize(self.base_model[:-1].transform(texts),norm='l2',copy=True)
            reference = self.reference_matrix
        # Dense or sparse matrices supported; process bounded batches.
        scores = []
        for start in range(0,len(texts),64):
            batch=x[start:start+64]
            product = batch @ reference.T
            if sparse.issparse(product):
                scores.extend(product.max(axis=1).toarray().ravel().tolist())
            else:
                scores.extend(np.asarray(product).max(axis=1).ravel().tolist())
        return np.asarray(scores,dtype=float)

def fit_calibrated_bundle(name, train, calibration):
    if set(train.group) & set(calibration.group):
        raise AssertionError("Calibration groups overlap fitting groups")
    if set(train.category) != set(calibration.category):
        raise ValueError("Calibration must contain each fitted category")
    base = fit_model(name, train)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid", ensemble="auto")
    with threadpool_limits(limits=1):
        calibrated.fit(calibration.text.tolist(), calibration.category.tolist())
    ref = normalize(base[:-1].transform(train.text.tolist()), norm="l2", copy=True)
    if sparse.issparse(ref):ref=ref.tocsr()
    # Novelty uses an independent train-fitted char space, not MiniLM confidence.
    novelty = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5),
                              preprocessor=preprocess,lowercase=False,
                              min_df=1,max_features=90000,sublinear_tf=True)
    novelty_matrix = normalize(novelty.fit_transform(train.text.tolist()),copy=False).tocsr()
    # References contain ONLY real train rows, not synthetic extensions.
    return RouterBundle(calibrated,base,ref,name,
                        reference_texts=train.text.tolist(),
                        reference_categories=train.category.tolist(),
                        novelty_vectorizer=novelty,novelty_matrix=novelty_matrix)
