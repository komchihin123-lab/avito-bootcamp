"""
Шаг 5. Ранжирование кандидатов LightGBM (lambdarank) и выбор топ-50.

Обучаемся на кандидатах "ранкерного" сплита (15000 запросов из train, отложенных от статистик),
оцениваем Recall@50 на валидации (2500 запросов), затем предсказываем для бенчмарка.

Запуск:
  python ranker.py eval  [suffix]  -> обучение на rank, Recall@50 на val
  python ranker.py bench [suffix]  -> обучение на rank(+val), answer.csv для бенчмарка
suffix — суффикс файлов признаков (например, "_noemb").
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import load_split, recall_at_k, WORK

MODE = sys.argv[1] if len(sys.argv) > 1 else "eval"
SUF = sys.argv[2] if len(sys.argv) > 2 else ""

PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[50], learning_rate=0.05,
              num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambdarank_truncation_level=60, verbose=-1, num_threads=8, seed=0)
N_ROUNDS = 600


import os
# признаки, которые можно исключить (эксперименты с устойчивостью), через запятую
DROP = set(filter(None, os.environ.get("DROP", "").split(",")))


def feat_cols(F):
    return [c for c in F.columns if c not in ("qi", "ii", "label") and c not in DROP]


def to_ds(F):
    F = F.sort_values("qi", kind="stable")
    # запросы без единого релевантного кандидата не дают сигнала для lambdarank
    has = F.groupby("qi").label.transform("max") > 0
    F = F[has]
    return lgb.Dataset(F[feat_cols(F)], F.label, group=F.groupby("qi", sort=True).size().values)


def top50(F, scores, items):
    """Для каждого запроса — 50 item_id с максимальным скором модели."""
    F = F[["qi", "ii"]].copy()
    F["s"] = scores
    F = F.sort_values(["qi", "s"], ascending=[True, False])
    F = F[F.groupby("qi").cumcount() < 50]
    ids = items.item_id.values
    return F.groupby("qi").ii.apply(lambda x: list(ids[x.values]))


def main():
    Fr = pd.read_parquet(f"{WORK}/feat_rank{SUF}.parquet")
    Fv = pd.read_parquet(f"{WORK}/feat_val{SUF}.parquet")
    cols = feat_cols(Fr)
    print("features:", cols)
    if MODE == "eval":
        qv, iv, _ = load_split("val")
        m = lgb.train(PARAMS, to_ds(Fr), N_ROUNDS, valid_sets=[to_ds(Fv)],
                      callbacks=[lgb.log_evaluation(100)])
        for n in [200, 400, 600]:
            if n > m.num_trees():
                continue
            p = top50(Fv, m.predict(Fv[cols], num_iteration=n), iv)
            preds = [p.get(i, []) for i in range(len(qv))]
            print(f"val Recall@50 (trees={n}):", round(recall_at_k(preds, qv.rel), 4))
        # для сравнения — простая эвристика BM25 + локация
        p = top50(Fv, Fv.comb1.values, iv)
        print("baseline comb1:", round(recall_at_k([p.get(i, []) for i in range(len(qv))], qv.rel), 4))
        imp = pd.Series(m.feature_importance("gain"), cols).sort_values(ascending=False)
        print(imp.round(0).head(40))
    else:
        # финальная модель: обучаемся на rank + val (больше данных)
        Fv = Fv.copy(); Fv["qi"] += Fr.qi.max() + 1
        m = lgb.train(PARAMS, to_ds(pd.concat([Fr, Fv], ignore_index=True)), N_ROUNDS)
        Fb = pd.read_parquet(f"{WORK}/feat_bench{SUF}.parquet")
        qb, ib, _ = load_split("bench")
        p = top50(Fb, m.predict(Fb[cols]), ib)
        # если у запроса вдруг меньше 50 кандидатов — просто отдаём сколько есть
        ans = pd.DataFrame({"query_id": qb.query_id.values,
                            "answer": [" ".join(p.get(i, [])) for i in range(len(qb))]})
        ans.to_csv("answer.csv", index=False)
        print("saved answer.csv", ans.shape, "avg items/row", ans.answer.str.split().str.len().mean())


if __name__ == "__main__":
    main()
