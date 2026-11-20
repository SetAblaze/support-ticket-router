"""Вклад признаков в линейную часть модели (не причинное объяснение)."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import normalize


def local_explanation(bundle, text, top_n=6):
    """Contributions to the BASE linear score; calibrated probability is separate."""
    if bundle.model_name.startswith('semantic_'):
        x = normalize(bundle.base_model[:-1].transform([text]),norm='l2')
        scores=(x @ bundle.reference_matrix.T).ravel()
        indices=np.argsort(-scores)[:min(top_n,3)]
        return {'status':'SEMANTIC_NEIGHBOURS_NOT_CAUSAL','class':str(bundle.predict([text])[0]),
                'examples':[{'text':bundle.reference_texts[i],'category':bundle.reference_categories[i],
                             'similarity':float(scores[i])} for i in indices]}
    clf = bundle.base_model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return {"status": "UNAVAILABLE_FOR_THIS_CLASSIFIER"}
    x = bundle.base_model[:-1].transform([text]).tocsr()
    names = bundle.base_model.named_steps["features"].get_feature_names_out()
    cls = str(bundle.predict([text])[0]); ci = list(clf.classes_).index(cls)
    weights = clf.coef_[ci, x.indices] * x.data
    order = np.argsort(-np.abs(weights))[:top_n]
    return {"status": "BASE_SCORE_CONTRIBUTIONS_NOT_CAUSAL", "class": cls,
            "intercept": float(clf.intercept_[ci]),
            "terms": [{"term": str(names[x.indices[i]]), "contribution": float(weights[i])} for i in order]}
