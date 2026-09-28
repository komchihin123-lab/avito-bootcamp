"""
Шаг 4. Кандидатогенерация + признаки для ранжирования.

Для каждого запроса считаем по ВСЕМУ корпусу (батчами запросов, плотные матрицы) несколько сигналов:
  * BM25 по полям (всё вместе с весами / заголовок / описание / параметры) и покрытие слов запроса;
  * локационный приор log P(локация объявления | локация поиска), посчитанный по train,
    и расстояние (км) от центра локации поиска до объявления;
  * "микрокатегорийный" приор P(микрокатегория | запрос): находим похожие запросы из train
    (kNN по TF-IDF словесных стемов + символьных n-грамм) и усредняем, в каких микрокатегориях
    выбирали объявления по ним;
  * история объявления: сколько раз его выбирали в train по похожим запросам и вообще;
  * косинусная близость эмбеддингов дообученной модели (если посчитаны).
Из нескольких комбинаций этих скоров берём топы -> объединение = кандидаты (~500-700 на запрос).
Для кандидатов сохраняем признаки и метку (для val/rank), дальше их ранжирует LightGBM (ranker.py).

Запуск: python features.py val|rank|bench
"""
import os, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from common import BM25Index, load_split, WORK

MODE = sys.argv[1] if len(sys.argv) > 1 else "val"
B = 150  # размер батча запросов
K_BM25, K_EMB, K_MC, K_HIST = 300, 300, 100, 50
LOG_EPS = 1e-4


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def topk_idx(M, k):
    """Индексы top-k по строкам плотной матрицы (без сортировки внутри топа)."""
    k = min(k, M.shape[1])
    return np.argpartition(-M, k - 1, axis=1)[:, :k]


def main():
    q, items, pairs = load_split(MODE)
    N, NQ = len(items), len(q)
    log(MODE, "queries", NQ, "items", N, "stat pairs", len(pairs))

    # ------------------------------------------------------------------ BM25
    idx_full = BM25Index({"title_st": 3.0, "params_st": 1.0, "desc_st": 1.0}).fit(items)
    idx_title = BM25Index({"title_st": 1.0}).fit(items)
    idx_desc = BM25Index({"desc_st": 1.0}).fit(items)
    idx_par = BM25Index({"params_st": 1.0}).fit(items)
    # бинарные матрицы "терм есть в поле" для покрытия запроса
    def binmat(idx, col):
        m = idx.cv.transform(items[col]).T.tocsr().astype(np.float32)
        m.data[:] = 1
        return m
    Tb = binmat(idx_full, "title_st")
    Ab = binmat(idx_full, "title_st") + binmat(idx_full, "params_st") + binmat(idx_full, "desc_st")
    Ab.data[:] = 1
    # тяжёлые текстовые колонки больше не нужны — освобождаем память
    items = items.drop(columns=["item_description_raw", "desc_st", "item_infm_params_text"])
    import gc; gc.collect()
    log("bm25 built")

    # ------------------------------------------------------------------ локации
    sl_codes = {l: i for i, l in enumerate(pd.unique(pd.concat([pairs.search_location_id, q.search_location_id])))}
    il_codes = {l: i for i, l in enumerate(pd.unique(pd.concat([pairs.item_location_id, items.item_location_id])))}
    P = np.zeros((len(sl_codes), len(il_codes)), np.float32)
    c = pairs.groupby(["search_location_id", "item_location_id"]).size()
    P[c.index.get_level_values(0).map(sl_codes), c.index.get_level_values(1).map(il_codes)] = c.values
    sloc_cnt = P.sum(1)
    P = P / np.maximum(sloc_cnt[:, None], 1)
    logP = np.log(P + LOG_EPS).astype(np.float32)
    qs = q.search_location_id.map(sl_codes).values
    dl = items.item_location_id.map(il_codes).values

    # центры локаций: для локаций объявлений — медиана координат объявлений,
    # для локаций поиска без объявлений (регионы) — медиана координат выбранных объявлений
    all_items_geo = pd.concat([
        pd.read_parquet(f"{WORK}/train_items.parquet", columns=["item_id", "item_location_id", "item_latitude", "item_longitude"]),
        items[["item_id", "item_location_id", "item_latitude", "item_longitude"]]]).drop_duplicates("item_id")
    cen = all_items_geo.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()
    pg = pairs[["search_location_id", "item_id"]].merge(all_items_geo[["item_id", "item_latitude", "item_longitude"]], on="item_id")
    cen_s = pg.groupby("search_location_id")[["item_latitude", "item_longitude"]].median()
    qlat = q.search_location_id.map(cen.item_latitude).fillna(q.search_location_id.map(cen_s.item_latitude)).values
    qlon = q.search_location_id.map(cen.item_longitude).fillna(q.search_location_id.map(cen_s.item_longitude)).values
    ilat, ilon = items.item_latitude.values, items.item_longitude.values
    # "плотность" корпуса в локации поиска
    loc_items_cnt = items.item_location_id.value_counts()
    q_loc_density = np.log1p(q.search_location_id.map(loc_items_cnt).fillna(0).values)

    # ------------------------------------------------------------------ kNN по запросам train
    # уникальные тексты запросов train -> распределение по микрокатегориям и по объявлениям
    tq_all = pd.read_parquet(f"{WORK}/train_queries.parquet", columns=["gid", "q_st", "search_query"])
    pr = pairs.merge(tq_all[["gid", "q_st"]], on="gid")
    texts = pd.Index(pr.q_st.unique())
    t_idx = texts.get_indexer(pr.q_st)
    mcs = pd.Index(pd.unique(pd.concat([pr.item_microcat_id, items.item_microcat_id])))
    TM = sp.csr_matrix((np.ones(len(pr), np.float32), (t_idx, mcs.get_indexer(pr.item_microcat_id))),
                       shape=(len(texts), len(mcs)))
    TM = normalize(TM, norm="l1")
    corpus_ids = pd.Index(items.item_id)
    in_corp = corpus_ids.get_indexer(pr.item_id)
    m = in_corp >= 0
    TI = sp.csr_matrix((np.ones(m.sum(), np.float32), (t_idx[m], in_corp[m])), shape=(len(texts), N))
    # популярность объявления в train (всего кликов)
    item_pop = np.bincount(in_corp[m], minlength=N).astype(np.float32)
    # "удалённость" микрокатегории: доля кликов в другую локацию
    pr_remote = (pr.search_location_id != pr.item_location_id).groupby(pr.item_microcat_id).mean()
    mc_remote = items.item_microcat_id.map(pr_remote).fillna(pr_remote.mean()).values.astype(np.float32)
    item_mc = mcs.get_indexer(items.item_microcat_id)

    vec_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+", sublinear_tf=True, min_df=1)
    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)
    Xt = normalize(sp.hstack([vec_w.fit_transform(texts), vec_c.fit_transform(texts)]).tocsr())
    Xq = normalize(sp.hstack([vec_w.transform(q.q_st), vec_c.transform(q.q_st)]).tocsr())
    XtT = Xt.T.tocsr()
    log("knn index built", len(texts))

    # ------------------------------------------------------------------ эмбеддинги
    use_emb = os.path.exists(f"{WORK}/emb_items.npy")
    if use_emb:
        E = np.load(f"{WORK}/emb_items.npy")
        eid = pd.read_parquet(f"{WORK}/emb_items_ids.parquet").item_id
        Ei = E[pd.Index(eid).get_indexer(items.item_id)].astype(np.float32)
        if MODE == "bench":
            Eq = np.load(f"{WORK}/emb_queries_bench.npy"); qid = pd.read_parquet(f"{WORK}/emb_queries_bench_ids.parquet").query_id
        else:
            Eq = np.load(f"{WORK}/emb_queries_train.npy"); qid = pd.read_parquet(f"{WORK}/emb_queries_train_ids.parquet").query_id
        Eq = Eq[pd.Index(qid).get_indexer(q.query_id)].astype(np.float32)
        log("embeddings loaded", Ei.shape)

    # ------------------------------------------------------------------ параметры поиска vs параметры объявления
    stop = {"вид", "услуг", "тип", "и", "в", "на"}
    qp_tokens = [[t for t in s.split() if t not in stop] for s in q.qparams_st]
    par_cv = idx_par.cv
    Pb = binmat(idx_par, "params_st")

    rel = q["rel"].values if "rel" in q else None
    out = []
    for s in range(0, NQ, B):
        e = min(NQ, s + B)
        bs = np.arange(s, e)
        Qm = idx_full.query_matrix(q.q_st.values[s:e])
        qlen = np.maximum(np.asarray(Qm.sum(1)).ravel(), 1)
        bm_full = (Qm @ idx_full.D).toarray()
        bm_title = (idx_title.query_matrix(q.q_st.values[s:e]) @ idx_title.D).toarray()
        bm_desc = (idx_desc.query_matrix(q.q_st.values[s:e]) @ idx_desc.D).toarray()
        bm_par = (idx_par.query_matrix(q.q_st.values[s:e]) @ idx_par.D).toarray()
        cov_t = (Qm @ Tb).toarray() / qlen[:, None]
        cov_a = (Qm @ Ab).toarray() / qlen[:, None]
        lp = logP[qs[bs]][:, dl]
        # kNN соседи (top-20)
        sims = (Xq[s:e] @ XtT).toarray()
        nn = topk_idx(sims, 20)
        nn_s = np.take_along_axis(sims, nn, 1)
        nn_w = nn_s ** 3
        W = sp.csr_matrix((nn_w.ravel(), (np.repeat(np.arange(e - s), nn.shape[1]), nn.ravel())), shape=(e - s, len(texts)))
        pmc_q = normalize(W @ TM, norm="l1").toarray()                      # B x n_mc
        pmc = pmc_q[:, item_mc]                                               # B x N
        hist = (W @ TI).toarray()                                             # B x N
        nn_max = nn_s.max(1)
        # совпадение фильтров поиска с параметрами объявления
        qpm = np.zeros((e - s, N), np.float32)
        qp_len = np.zeros(e - s, np.float32)
        for j, i in enumerate(bs):
            if qp_tokens[i]:
                v = par_cv.transform([" ".join(qp_tokens[i])])
                v.data[:] = 1
                qp_len[j] = max(v.sum(), 1)
                qpm[j] = (v @ Pb).toarray().ravel() / qp_len[j]
        if use_emb:
            es = Eq[s:e] @ Ei.T

        # ---- источники кандидатов
        cands = [topk_idx(bm_full + 2 * lp, K_BM25),
                 topk_idx(np.log(pmc + 1e-4) + lp + 0.3 * bm_full + np.log1p(hist), K_MC),
                 topk_idx(hist + 1e-6 * (lp + 10), K_HIST)]
        if use_emb:
            cands.append(topk_idx(20 * es + 2 * lp, K_EMB))
        cand = np.concatenate(cands, 1)

        for j, i in enumerate(bs):
            c = np.unique(cand[j])
            f = {
                "qi": np.full(len(c), i, np.int32), "ii": c.astype(np.int32),
                "bm_full": bm_full[j, c], "bm_title": bm_title[j, c], "bm_desc": bm_desc[j, c], "bm_par": bm_par[j, c],
                "cov_t": cov_t[j, c], "cov_a": cov_a[j, c],
                "logploc": lp[j, c], "same_loc": (items.item_location_id.values[c] == q.search_location_id.values[i]).astype(np.int8),
                "dist": np.log1p(haversine(qlat[i], qlon[i], ilat[c], ilon[c])),
                "pmc": pmc[j, c], "hist": hist[j, c], "item_pop": np.log1p(item_pop[c]), "mc_remote": mc_remote[c],
                "qpm": qpm[j, c],
            }
            if use_emb:
                f["emb"] = es[j, c]
            fd = pd.DataFrame(f)
            # query-level
            fd["qlen"] = qlen[j]; fd["qp_len"] = qp_len[j]; fd["nn_max"] = nn_max[j]
            fd["sloc_cnt"] = np.log1p(sloc_cnt[qs[i]]); fd["q_loc_density"] = q_loc_density[i]
            fd["pmc_max"] = pmc_q[j].max()
            # относительные (внутри запроса) признаки
            fd["bm_full_rel"] = fd.bm_full / (bm_full[j].max() + 1e-6)
            fd["bm_full_rank"] = fd.bm_full.rank(ascending=False)
            fd["comb1"] = fd.bm_full + 2 * fd.logploc
            fd["comb1_rank"] = fd.comb1.rank(ascending=False)
            if use_emb:
                fd["emb_rel"] = fd.emb - es[j].max()
                fd["emb_rank"] = fd.emb.rank(ascending=False)
                fd["comb2_rank"] = (20 * fd.emb + 2 * fd.logploc).rank(ascending=False)
            if rel is not None:
                rs = set(rel[i])
                fd["label"] = np.isin(items.item_id.values[c], list(rs)).astype(np.int8)
            out.append(fd)
        log(f"batch {s}-{e}")

    F = pd.concat(out, ignore_index=True)
    # признаки объявления
    it = items[["item_rating", "item_rating_reviews_count", "item_price", "item_is_phone_hidden",
                "item_is_message_forbidden", "item_category_id", "title_st"]].iloc[F.ii.values]
    F["rating"] = it.item_rating.values
    F["reviews"] = np.log1p(it.item_rating_reviews_count.fillna(0).values)
    F["price"] = np.log1p(it.item_price.fillna(-1).clip(lower=0).values)
    F["phone_hidden"] = it.item_is_phone_hidden.astype(float).values
    F["msg_forbidden"] = it.item_is_message_forbidden.astype(float).values
    F["item_cat"] = it.item_category_id.values
    F["title_len"] = it.title_st.str.count(" ").values + 1
    F["search_cat"] = q.search_category.values[F.qi.values]
    F.to_parquet(f"{WORK}/feat_{MODE}.parquet")
    if rel is not None:
        # потолок полноты кандидатов
        hit = F.groupby("qi").label.sum().reindex(range(NQ), fill_value=0).values
        nrel = np.array([len(r) for r in rel])
        log("candidates per query", len(F) / NQ, "candidate recall", np.mean(hit / nrel))


if __name__ == "__main__":
    main()
