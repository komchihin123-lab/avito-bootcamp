"""Эксперимент: BM25 без учёта / с учётом локации (валидация)."""
import numpy as np, pandas as pd
from common import *

q, items, pairs = load_split("val")
idx = BM25Index({"title_st": 3.0, "params_st": 1.0, "desc_st": 1.0}).fit(items)
S = idx.score(q.q_st)
ids = items.item_id.values
loc = items.item_location_id.values

# P(item_loc | search_loc) по train
cnt = pairs.groupby(["search_location_id", "item_location_id"]).size().rename("n").reset_index()
tot = cnt.groupby("search_location_id").n.transform("sum")
cnt["p"] = cnt.n / tot
ploc = {(a, b): p for a, b, p in zip(cnt.search_location_id, cnt.item_location_id, cnt.p)}

res = {"bm25": [], "bm25_sameloc": []}
for alpha in [0.5, 1, 2, 4]:
    res[f"bm25+loc{alpha}"] = []
for i in range(len(q)):
    r = S.getrow(i)
    ri, rv = r.indices, r.data
    res["bm25"].append(ids[topk_from_sparse_row(ri, rv, 50)[0]])
    sl = q.search_location_id[i]
    same = loc[ri] == sl
    res["bm25_sameloc"].append(ids[topk_from_sparse_row(ri[same], rv[same], 50)[0]])
    pl = np.array([ploc.get((sl, l), 0.0) for l in loc[ri]])
    for alpha in [0.5, 1, 2, 4]:
        sc = rv + alpha * np.log(pl + 1e-3)
        res[f"bm25+loc{alpha}"].append(ids[topk_from_sparse_row(ri, sc, 50)[0]])
for k, v in res.items():
    print(k, round(recall_at_k(v, q.rel), 4))
print("any bm25 hit (all matches):", np.mean([len(set(ids[S.getrow(i).indices]) & set(r)) > 0 for i, r in enumerate(q.rel)]))
