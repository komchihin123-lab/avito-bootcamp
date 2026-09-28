"""
Шаг 1. Подготовка данных.

Что делаем:
  * нормализуем тексты (нижний регистр, ё->е, токенизация, стемминг Snowball через PyStemmer);
  * собираем "запросы" из train: группа = (текст запроса, локация, фильтры, категория, флаг доставки),
    релевантные объявления группы = все item_id, которые выбрали по этому запросу;
  * откладываем валидационные группы (~2500), чтобы локально мерить Recall@50,
    имитируя бенчмарк: в бенчмарке каждый текст запроса встречается один раз,
    и ~37% текстов бенчмарка встречаются в train — подбираем такую же долю;
  * валидационный корпус = benchmark_items + объявления, выбранные по валидационным запросам
    (их признаки берём из строк train).

Все промежуточные результаты складываются в папку work/.
"""
import os
import re
import numpy as np
import pandas as pd
import Stemmer

DATA = "dataset"
WORK = "work"
os.makedirs(WORK, exist_ok=True)
RNG = np.random.default_rng(42)

ITEM_COLS = ["item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
             "item_category_id", "item_microcat_id", "item_price", "item_rating",
             "item_rating_reviews_count", "item_location_id", "item_latitude", "item_longitude",
             "item_is_phone_hidden", "item_is_message_forbidden"]
QUERY_KEY = ["search_query", "search_location_id", "search_infm_params_text",
             "search_category", "search_is_delivery_search"]

_stemmer = Stemmer.Stemmer("russian")
_tok_re = re.compile(r"[a-zа-я0-9]+")


def norm_tokens(s: str):
    """Нормализация строки -> список стемов."""
    if not isinstance(s, str) or not s:
        return []
    s = s.lower().replace("ё", "е")
    return _stemmer.stemWords(_tok_re.findall(s))


def stem_series(ser: pd.Series, max_chars=None) -> pd.Series:
    """Стемминг колонки с кэшированием одинаковых строк (запросы/заголовки часто повторяются)."""
    ser = ser.fillna("").astype(str)
    if max_chars:
        ser = ser.str.slice(0, max_chars)
    uniq = pd.unique(ser)
    mp = {u: " ".join(norm_tokens(u)) for u in uniq}
    return pd.Series([mp[x] for x in ser], index=ser.index, dtype=object)


def clean_items(df: pd.DataFrame) -> pd.DataFrame:
    df = df[ITEM_COLS].copy()
    # decimal -> float
    for c in ["item_price", "item_latitude", "item_longitude"]:
        df[c] = pd.to_numeric(df[c].astype("float64"), errors="coerce")
    df["title_st"] = stem_series(df.item_title_raw)
    df["params_st"] = stem_series(df.item_infm_params_text)
    df["desc_st"] = stem_series(df.item_description_raw, max_chars=1500)
    return df


def main():
    # ---------------- train ----------------
    tr = pd.read_parquet(f"{DATA}/train.parquet")
    tr["search_infm_params_text"] = tr["search_infm_params_text"].fillna("")
    tr = tr.drop_duplicates(QUERY_KEY + ["item_id"]).reset_index(drop=True)
    tr["gid"] = tr.groupby(QUERY_KEY, sort=False).ngroup()
    print("train rows", len(tr), "groups", tr.gid.nunique())

    # ------------- валидационный сплит -------------
    # берём по одной группе на текст (как в бенчмарке), причём 37% — тексты, которые
    # ещё встречаются в остальном train, 63% — "уникальные" тексты.
    grp = tr.groupby("gid").agg(q=("search_query", "first")).reset_index()
    text_ngroups = grp.q.value_counts()
    grp["n_text"] = grp.q.map(text_ngroups)
    one_per_text = grp.sample(frac=1, random_state=1).drop_duplicates("q")
    N_VAL = 2500
    n_seen = int(N_VAL * 0.37)
    seen = one_per_text[one_per_text.n_text > 1].sample(n_seen, random_state=2)
    unseen = one_per_text[one_per_text.n_text == 1].sample(N_VAL - n_seen, random_state=3)
    val_gids = set(seen.gid) | set(unseen.gid)
    tr["is_val"] = tr.gid.isin(val_gids)
    print("val groups", len(val_gids), "val rows", tr.is_val.sum())

    # отдельный сплит для обучения ранжирующей модели (так же, по одной группе на текст)
    N_RANK = 15000
    pool = one_per_text[~one_per_text.gid.isin(val_gids)]
    n_seen = int(N_RANK * 0.37)
    rk = pd.concat([pool[pool.n_text > 1].sample(n_seen, random_state=4),
                    pool[pool.n_text == 1].sample(N_RANK - n_seen, random_state=5)])
    tr["is_rank"] = tr.gid.isin(set(rk.gid))
    print("rank groups", tr[tr.is_rank].gid.nunique())

    # ------------- объявления -------------
    bench_items = pd.read_parquet(f"{DATA}/benchmark_items.parquet")
    bench_items = clean_items(bench_items)
    bench_items["src"] = "bench"
    bench_items.to_parquet(f"{WORK}/bench_items.parquet")
    import gc; gc.collect()

    # объявления из train (уникальные по item_id) — нужны для валид. корпуса и для обучения
    tr_items = clean_items(tr.drop_duplicates("item_id"))
    tr_items["src"] = "train"
    tr_items.to_parquet(f"{WORK}/train_items.parquet")

    # валидационный корпус: bench_items + таргеты валидации, которых нет в bench
    val_targets = tr.loc[tr.is_val, "item_id"].unique()
    extra = tr_items[tr_items.item_id.isin(val_targets) & ~tr_items.item_id.isin(bench_items.item_id)]
    val_corpus = pd.concat([bench_items, extra], ignore_index=True)
    val_corpus.to_parquet(f"{WORK}/val_items.parquet")
    print("val corpus", len(val_corpus))
    rank_targets = tr.loc[tr.is_rank, "item_id"].unique()
    extra = tr_items[tr_items.item_id.isin(rank_targets) & ~tr_items.item_id.isin(bench_items.item_id)]
    pd.concat([bench_items, extra], ignore_index=True).to_parquet(f"{WORK}/rank_items.parquet")

    # ------------- запросы -------------
    qcols = QUERY_KEY + ["gid", "is_val", "is_rank"]
    tr_q = tr[qcols].drop_duplicates("gid").copy()
    rel = tr.groupby("gid").item_id.apply(list)
    tr_q["rel"] = tr_q.gid.map(rel)
    tr_q["query_id"] = "g" + tr_q.gid.astype(str)
    tr_q["q_st"] = stem_series(tr_q.search_query)
    tr_q["qparams_st"] = stem_series(tr_q.search_infm_params_text)
    tr_q.to_parquet(f"{WORK}/train_queries.parquet")

    bq = pd.read_parquet(f"{DATA}/benchmark_queries.parquet")
    bq["search_infm_params_text"] = bq["search_infm_params_text"].fillna("")
    bq["q_st"] = stem_series(bq.search_query)
    bq["qparams_st"] = stem_series(bq.search_infm_params_text)
    bq.to_parquet(f"{WORK}/bench_queries.parquet")

    # строки train (пары запрос-объявление) без тяжёлых текстов — для статистик
    pair_cols = ["gid", "is_val", "is_rank", "item_id", "search_location_id", "item_location_id",
                 "item_microcat_id", "search_query"]
    tr[pair_cols].to_parquet(f"{WORK}/train_pairs.parquet")

    # проверка: доля текстов валидации, встречающихся в остальном train
    rest_texts = set(tr.loc[~tr.is_val, "search_query"])
    vq = tr_q[tr_q.is_val]
    print("val text seen in train:", vq.search_query.isin(rest_texts).mean())
    print("val targets in bench corpus:", np.mean(np.isin(val_targets, bench_items.item_id)))


if __name__ == "__main__":
    main()
