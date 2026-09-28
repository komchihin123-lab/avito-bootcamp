"""Эксперимент: полнота кандидатов BM25+локация на разных глубинах (потолок для ранжирования)."""
import numpy as np, pandas as pd
from common import *

q, items, pairs = load_split("val")
idx = BM25Index({"title_st": 3.0, "params_st": 1.0, "desc_st": 1.0}).fit(items)
S = idx.score(q.q_st).tocoo()

# матрица P(item_loc | search_loc)
sl_codes = {l: i for i, l in enumerate(pd.unique(pd.concat([pairs.search_location_id, q.search_location_id])))}
il_codes = {l: i for i, l in enumerate(pd.unique(pd.concat([pairs.item_location_id, items.item_location_id])))}
P = np.zeros((len(sl_codes), len(il_codes)), np.float32)
c = pairs.groupby(["search_location_id", "item_location_id"]).size()
a = c.index.get_level_values(0).map(sl_codes); b = c.index.get_level_values(1).map(il_codes)
P[a, b] = c.values
P /= np.maximum(P.sum(1, keepdims=True), 1)

qs = q.search_location_id.map(sl_codes).values
dl = items.item_location_id.map(il_codes).values
sc = S.data + 2 * np.log(P[qs[S.row], dl[S.col]] + 1e-3)
df = pd.DataFrame({"r": S.row, "c": S.col, "s": sc}).sort_values(["r", "s"], ascending=[True, False])
df["rk"] = df.groupby("r").cumcount()
ids = items.item_id.values
for K in [50, 100, 200, 500, 1000, 3000]:
    top = df[df.rk < K].groupby("r").c.apply(lambda x: ids[x.values])
    preds = [top.get(i, []) for i in range(len(q))]
    print(K, round(recall_at_k(preds, q.rel, k=K), 4))
