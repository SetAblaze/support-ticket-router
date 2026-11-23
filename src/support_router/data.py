"""Очистка датасета, группы похожих фраз и разбиение без утечки."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize

from .config import CATEGORIES, LINGUISTIC_FLAGS, SEED
from .text import parse_linguistic_flags, preprocess, text_id


def clean_frame(raw: pd.DataFrame):
    expected = {"utterance", "category", "intent", "flags"}
    if not expected.issubset(raw.columns):
        raise ValueError(f"Missing columns: {sorted(expected - set(raw.columns))}")
    d = raw[list(sorted(expected))].copy()
    if d[["utterance", "category", "intent"]].isna().any().any():
        raise ValueError("Missing text/labels: resolve or quarantine explicitly before fitting")
    if not all(isinstance(t, str) for t in d.utterance):
        raise ValueError("utterance must contain strings")
    d["text"] = d.utterance.map(preprocess)
    if d.text.eq("").any():
        raise ValueError("Empty normalized training text: resolve explicitly")
    if set(d.category) != set(CATEGORIES):
        raise ValueError("Unexpected category set; do not silently change taxonomy")
    conflicts = d.groupby("text").agg(categories=("category", "nunique"),
                                      intents=("intent", "nunique"))
    if (conflicts.categories > 1).any() or (conflicts.intents > 1).any():
        raise ValueError("Conflicting labels after normalization: manual resolution required")
    # Preserve ALL generation flags represented by duplicates, rather than an arbitrary first.
    flags = d.groupby("text")["flags"].agg(
        lambda xs: "".join(sorted({flag for x in xs for flag in parse_linguistic_flags(x)})))
    counts = d.text.value_counts()
    d = d.drop_duplicates("text", keep="first").reset_index(drop=True)
    d["flags"] = d.text.map(flags)
    d["source_duplicate_count"] = d.text.map(counts).astype(int)
    d["row_id"] = d.text.map(text_id)
    d["n_words"] = d.text.str.split().str.len()
    short = d[d.n_words < 3].copy()
    info = {"raw_rows": len(raw), "clean_rows": len(d),
            "deduplicated_rows": len(raw) - len(d), "short_rows_retained": len(short),
            "flags_policy": "union_of_unique_parsed_flags_for_identical_normalized_text",
            "unrecognized_flag_codes": sorted(set("".join(flags)) - set(LINGUISTIC_FLAGS)),
            "short_policy": "retained_in_main_splits_and_reported_as_a_slice"}
    return d, short, info

def lexical_groups(texts, threshold=.85, batch_size=64):
    """Connected-components grouping with character AND word n-grams.

    This is a split-building heuristic, fitted on the full unsplit corpus by
    design. It is NEVER reused for model features/evaluation. Strings in a
    component stay together even when a different label occurs; report these.
    The thresholds are frozen a priori, not tuned on test results.
    """
    texts = list(texts)
    if not texts:
        raise ValueError('No cleaned texts to group')
    word_threshold = max(.90, float(threshold))
    char_threshold = float(threshold)
    spaces = [
        (TfidfVectorizer(preprocessor=preprocess, lowercase=False,
                         ngram_range=(1, 2), min_df=1, max_features=60000,
                         sublinear_tf=True), word_threshold, 'word_1_2'),
        (TfidfVectorizer(preprocessor=preprocess, lowercase=False,
                         analyzer='char_wb', ngram_range=(3, 5), min_df=1,
                         max_features=90000, sublinear_tf=True), char_threshold, 'char_wb_3_5'),
    ]
    from collections import Counter
    from difflib import SequenceMatcher
    safe_fillers = {'while','whlie','please','kindly','just','really','there',
                    'my','an','a','the','some','any'}
    def typo_bridge(one,two):
        a,b=preprocess(one).split(),preprocess(two).split()
        if min(len(a),len(b))<4 or abs(len(a)-len(b))>1:return False
        counts_a,counts_b=Counter(a),Counter(b)
        common=counts_a&counts_b
        if sum(common.values())<4:return False
        extra_a=list((counts_a-common).elements())
        extra_b=list((counts_b-common).elements())
        if len(extra_a)==1 and len(extra_b)==1:
            x,y=extra_a[0],extra_b[0]
            return min(len(x),len(y))>=4 and SequenceMatcher(None,x,y).ratio()>=.76
        if len(extra_a)==1 and not extra_b:return extra_a[0] in safe_fillers
        if len(extra_b)==1 and not extra_a:return extra_b[0] in safe_fillers
        return False
    rows, cols = [], []
    per_view = {}
    for vectorizer, cut, label in spaces:
        matrix = vectorizer.fit_transform(texts)
        matrix = normalize(matrix, copy=False)
        edge_count = 0
        for start in range(0, len(texts), batch_size):
            sims = (matrix[start:start + batch_size] @ matrix.T).tocsr()
            floor = .48 if label == 'word_1_2' else cut
            sims.data[sims.data < floor] = 0
            sims.eliminate_zeros()
            coo = sims.tocoo()
            mask = coo.row + start < coo.col
            candidate_rows = coo.row[mask] + start
            candidate_cols = coo.col[mask]
            candidate_values = coo.data[mask]
            for i,j,cosine in zip(candidate_rows,candidate_cols,candidate_values):
                if cosine>=cut or (label=='word_1_2' and typo_bridge(texts[int(i)],texts[int(j)])):
                    rows.append(int(i));cols.append(int(j));edge_count+=1
        per_view[label] = {'cosine_threshold': cut, 'undirected_edges': edge_count,
                           'vocabulary': len(vectorizer.vocabulary_)}
        del matrix
    # Undirected connected components, no dense N x N similarity matrix.
    matrix = sparse.coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                               shape=(len(texts), len(texts))).tocsr()
    count, groups = connected_components(matrix, directed=False)
    sizes = pd.Series(groups).value_counts()
    return groups, {
        'method': 'connected_components_char_wb_3_5_OR_word_1_2',
        'threshold': float(threshold), 'views': per_view,
        'n_groups': int(count), 'largest_group': int(sizes.max()),
        'largest_fraction': float(sizes.max()/len(texts)),
        'edges_undirected_before_dedup': len(rows),
        'semantic_equivalence_verified': False,
        'warning': 'Heuristic groups can overmerge; inspect largest clusters and cross-label groups',
    }

def balanced_group_holdout(frame, fraction, seed=SEED, candidates=96):
    """Select by label/size balance only, never by downstream metric."""
    labels = sorted(frame.category.unique())
    totals = frame.category.value_counts().reindex(labels).to_numpy(float)
    best = None
    splitter = GroupShuffleSplit(n_splits=candidates, test_size=fraction, random_state=seed)
    for a, b in splitter.split(frame, frame.category, frame.group):
        tr, te = frame.iloc[a], frame.iloc[b]
        if set(tr.category) != set(labels) or set(te.category) != set(labels):
            continue
        rates = te.category.value_counts().reindex(labels, fill_value=0).to_numpy() / totals
        cost = float(np.mean((rates - fraction)**2) + (len(te)/len(frame)-fraction)**2)
        if best is None or cost < best[0]:
            best = cost, a, b
    if best is None:
        raise ValueError("Cannot split whole groups while retaining all categories; inspect groups")
    return frame.iloc[best[1]].copy(), frame.iloc[best[2]].copy()

def make_splits(frame):
    remaining, test = balanced_group_holdout(frame, .15, SEED)
    fit_cal, val = balanced_group_holdout(remaining, .15/.85, SEED+1)
    train, calibration = balanced_group_holdout(fit_cal, .14/.70, SEED+2)
    splits = {"train": train, "calibration": calibration, "val": val, "test": test}
    assert_split_integrity(splits)
    return splits

def assert_split_integrity(splits):
    checks = {}
    keys = list(splits)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            item = {col: len(set(splits[a][col]) & set(splits[b][col]))
                    for col in ["text", "row_id", "group"]}
            checks[f"{a}__{b}"] = item
            if any(item.values()):
                raise AssertionError(f"Split overlap {a}/{b}: {item}")
    for name, df in splits.items():
        if set(df.category) != set(CATEGORIES):
            raise AssertionError(f"Missing categories in {name}")
    return checks

def cross_split_similarity(train, other, batch_size=64):
    """Train-only TF-IDF diagnostic; max of word and typo-sensitive char similarity.

    Similarity values are descriptive: do not move test rows or tune the
    grouping threshold based on these diagnostics.
    """
    texts_train = train.text.tolist()
    texts_eval = other.text.tolist()
    maxima = np.zeros(len(other), dtype=float)
    best_ix = np.zeros(len(other), dtype=int)
    views = [('word', TfidfVectorizer(preprocessor=preprocess, lowercase=False,
                   ngram_range=(1, 2), min_df=1, max_features=60000, sublinear_tf=True)),
             ('char', TfidfVectorizer(preprocessor=preprocess, lowercase=False,
                   analyzer='char_wb', ngram_range=(3, 5), min_df=1,
                   max_features=90000, sublinear_tf=True))]
    for label, vec in views:
        a = normalize(vec.fit_transform(texts_train),copy=False)
        b = normalize(vec.transform(texts_eval),copy=False)
        for start in range(0, len(other), batch_size):
            similarity = (b[start:start+batch_size] @ a.T).tocsr()
            values = np.asarray(similarity.max(axis=1).toarray()).ravel()
            indices = np.asarray(similarity.argmax(axis=1)).ravel()
            positions = np.arange(start, min(start+batch_size, len(other)))
            mask = values > maxima[positions]
            maxima[positions[mask]] = values[mask]
            best_ix[positions[mask]] = indices[mask]
    examples = [{'cosine_max_views':float(score),
                 'train_text':texts_train[int(best_ix[j])],
                 'heldout_text':texts_eval[j]}
                for j, score in enumerate(maxima) if score >= .85]
    summary = {'n':len(maxima),'mean':float(maxima.mean()) if len(maxima) else None,
               'p95':float(np.quantile(maxima,.95)) if len(maxima) else None,
               'n_ge_085':int((maxima>=.85).sum()),
               'n_ge_090':int((maxima>=.90).sum()),
               'method':'train-only max(word_1_2, char_wb_3_5)',
               'warning':'Residual lexical similarity remains possible; never prove semantic disjointness'}
    return summary, pd.DataFrame(sorted(examples, key=lambda r:-r['cosine_max_views'])[:40],
                                columns=['cosine_max_views','train_text','heldout_text'])


def group_diagnostics(frame, top_n=10, max_pairwise=40):
    """Describe lexical groups without changing the split.

    Large connected components can merge different intents through a chain of
    similar phrasings.  This report makes that risk visible; it is descriptive,
    not a new grouping threshold.
    """
    groups = list(frame.groupby("group"))
    sizes = frame.group.value_counts()
    cross_category = [str(g) for g, part in groups if part.category.nunique() > 1]
    cross_intent = [str(g) for g, part in groups if part.intent.nunique() > 1]
    largest = []
    for group, part in sorted(groups, key=lambda item: -len(item[1]))[:top_n]:
        texts = part.text.tolist()
        sample = texts[:max_pairwise]
        mean_similarity = minimum_similarity = None
        if len(sample) >= 2:
            vectorizer = TfidfVectorizer(preprocessor=preprocess, lowercase=False,
                                         analyzer="char_wb", ngram_range=(3, 5),
                                         min_df=1, sublinear_tf=True)
            matrix = normalize(vectorizer.fit_transform(sample))
            similarity = (matrix @ matrix.T).toarray()
            values = similarity[np.triu_indices(len(sample), 1)]
            if len(values):
                mean_similarity = float(values.mean())
                minimum_similarity = float(values.min())
        largest.append({
            "group": str(group), "size": len(part),
            "categories": sorted(part.category.unique().tolist()),
            "intents": sorted(part.intent.unique().tolist()),
            "mean_pairwise_char_similarity": mean_similarity,
            "min_pairwise_char_similarity": minimum_similarity,
            "example_texts": texts[:5],
        })
    return {
        "n_groups": int(len(groups)),
        "size_min": int(sizes.min()), "size_median": float(sizes.median()),
        "size_p95": float(sizes.quantile(.95)), "size_max": int(sizes.max()),
        "largest_fraction": float(sizes.max() / len(frame)),
        "groups_with_multiple_categories": len(cross_category),
        "groups_with_multiple_intents": len(cross_intent),
        "cross_category_group_ids": cross_category[:20],
        "cross_intent_group_ids": cross_intent[:20],
        "largest_groups": largest,
        "method": "connected_components over word and char TF-IDF views",
        "warning": ("Group overlap is not semantic independence; inspect the largest "
                    "components and cross-label groups before claiming a clean split"),
    }


def leakage_diagnostics(splits):
    """Exact overlap counts for every pair of splits (all should be zero)."""
    keys = list(splits)
    report = {}
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            report[f"{left}__{right}"] = {
                column: len(set(splits[left][column]) & set(splits[right][column]))
                for column in ("text", "row_id", "group")
            }
    return {"pairs": report,
            "all_zero": all(value == 0 for pair in report.values() for value in pair.values()),
            "note": "text is already normalized by preprocess; zero overlap is necessary, not sufficient"}
