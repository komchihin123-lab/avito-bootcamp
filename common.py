"""
Общие утилиты: загрузка подготовленных данных, BM25-индекс, метрика Recall@50.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

WORK = "work"


def recall_at_k(pred_lists, rel_lists, k=50):
    """Recall@k, усреднённый по запросам (как в условии)."""
    vals = []
    for p, r in zip(pred_lists, rel_lists):
        r = set(r)
        vals.append(len(set(p[:k]) & r) / len(r))
    return float(np.mean(vals))


def _split(s):
    return s.split()


class BM25Index:
    """
    Простой BM25F: частоты термов по полям складываются с весами
    (заголовок важнее описания), затем применяется стандартная формула BM25.
    Документ-матрица хранится в CSR, скоринг = Q(бинарный) @ D^T.
    """

    def __init__(self, field_weights, k1=1.2, b=0.75, min_df=1):
        self.fw = field_weights
        self.k1, self.b, self.min_df = k1, b, min_df

    def fit(self, df: pd.DataFrame):
        cv = CountVectorizer(analyzer=_split, min_df=self.min_df, dtype=np.float32)
        cv.fit(pd.concat([df[f] for f in self.fw]))
        self.cv = cv
        tf = None
        for f, w in self.fw.items():
            m = cv.transform(df[f]) * w
            tf = m if tf is None else tf + m
        tf = tf.tocsr()
        n = tf.shape[0]
        dl = np.asarray(tf.sum(1)).ravel()
        avgdl = dl.mean()
        df_ = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n - df_ + 0.5) / (df_ + 0.5)).astype(np.float32)
        # BM25-насыщение tf
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[rows] / avgdl)
        tf.data = tf.data * (self.k1 + 1) / denom * self.idf[tf.indices]
        self.D = tf.T.tocsr()  # term x doc — удобно для Q @ D
        return self

    def query_matrix(self, texts):
        q = self.cv.transform(texts)
        q.data[:] = 1.0
        return q.astype(np.float32).tocsr()

    def score(self, texts):
        return (self.query_matrix(texts) @ self.D).tocsr()  # queries x docs (sparse)


def topk_from_sparse_row(row_idx, row_val, k):
    if len(row_val) <= k:
        o = np.argsort(-row_val)
    else:
        part = np.argpartition(-row_val, k)[:k]
        o = part[np.argsort(-row_val[part])]
    return row_idx[o], row_val[o]


def load_split(mode):
    """
    mode='val'   -> валидационные запросы из train + валидационный корпус
    mode='rank'  -> запросы для обучения ранкера + их корпус
    mode='bench' -> запросы бенчмарка + корпус бенчмарка
    Возвращает (queries, items, pairs), где pairs — строки train, по которым разрешено
    считать статистики (без val и rank групп, чтобы не было утечки меток).
    """
    pairs = pd.read_parquet(f"{WORK}/train_pairs.parquet")
    pairs = pairs[~pairs.is_val & ~pairs.is_rank].reset_index(drop=True)
    if mode in ("val", "rank"):
        tq = pd.read_parquet(f"{WORK}/train_queries.parquet")
        q = tq[tq.is_val if mode == "val" else tq.is_rank].reset_index(drop=True)
        items = pd.read_parquet(f"{WORK}/{mode}_items.parquet")
    else:
        q = pd.read_parquet(f"{WORK}/bench_queries.parquet")
        items = pd.read_parquet(f"{WORK}/bench_items.parquet")
    return q, items.reset_index(drop=True), pairs
