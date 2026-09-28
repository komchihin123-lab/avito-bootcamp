"""Смотрим на промахи: какие таргеты не попадают в топ-1000 BM25+loc."""
import numpy as np, pandas as pd
exec(open('exp_headroom.py',encoding='utf-8').read().split("ids = items.item_id.values")[0])
ids = items.item_id.values
top = df[df.rk < 1000].groupby("r").c.apply(lambda x: set(ids[x.values]))
it = items.set_index("item_id")
rng = np.random.default_rng(0); shown=0
for i in rng.permutation(len(q)):
    got = top.get(i, set())
    miss = [r for r in q.rel[i] if r not in got]
    if not miss: continue
    r = it.loc[miss[0]]
    print(f"Q: {q.search_query[i]!r} | {q.search_infm_params_text[i]} | sloc={q.search_location_id[i]} cat={q.search_category[i]}")
    print(f"   T: {r.item_title_raw!r} | {str(r.item_infm_params_text)[:80]} | iloc={r.item_location_id} mc={r.item_microcat_id} | {str(r.item_description_raw)[:120]!r}")
    shown += 1
    if shown >= 25: break
