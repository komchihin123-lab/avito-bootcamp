"""Проверка answer.csv на соответствие требованиям формата из условия."""
import pandas as pd
a = pd.read_csv("answer.csv", dtype=str, keep_default_na=False)
q = pd.read_parquet("dataset/benchmark_queries.parquet")
it = set(pd.read_parquet("dataset/benchmark_items.parquet", columns=["item_id"]).item_id)
assert list(a.columns) == ["query_id", "answer"], a.columns
assert len(a) == len(q) and set(a.query_id) == set(q.query_id) and a.query_id.is_unique
lists = a.answer.str.split(" ")
assert (lists.str.len() <= 50).all()
assert all(len(set(l)) == len(l) for l in lists)
assert all(x in it for l in lists for x in l)
assert all(len(x) == 16 for l in lists for x in l)
print("OK:", len(a), "rows; items per row min/mean:", lists.str.len().min(), lists.str.len().mean())
