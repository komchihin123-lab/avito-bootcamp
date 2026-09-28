"""
Шаг 3. Считаем эмбеддинги дообученной моделью:
  * все объявления: benchmark_items + объявления из train (нужны для валид./ранкерного корпуса);
  * все запросы: train-группы (val + rank) и запросы бенчмарка.
Результат: work/emb_items.npy (+ work/emb_items_ids.parquet), work/emb_queries_*.npy (float16).
Запуск: .venv_gpu/Scripts/python encode_emb.py  (на CPU тоже работает, но медленно)
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from train_emb import query_text, item_text, mean_pool, Q_LEN, D_LEN

MODEL = os.environ.get("EMB_MODEL", "work/emb_model")
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(dev).eval()


@torch.no_grad()
def encode(texts, L, bs=256):
    # сортируем по длине, чтобы меньше паддинга
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), np.float16)
    for s in range(0, len(texts), bs):
        idx = order[s:s + bs]
        b = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=L, return_tensors="pt").to(dev)
        e = F.normalize(mean_pool(model(**b), b["attention_mask"]).float(), dim=-1)
        out[idx] = e.cpu().numpy().astype(np.float16)
        if (s // bs) % 200 == 0:
            print(s, len(texts), flush=True)
    return out


def main():
    cols = ["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"]
    items = pd.concat([pd.read_parquet("work/bench_items.parquet", columns=cols),
                       pd.read_parquet("work/train_items.parquet", columns=cols)]).drop_duplicates("item_id")
    texts = [item_text(a, b, c) for a, b, c in zip(items.item_title_raw, items.item_infm_params_text, items.item_description_raw)]
    np.save("work/emb_items.npy", encode(texts, D_LEN, bs=64))
    items[["item_id"]].reset_index(drop=True).to_parquet("work/emb_items_ids.parquet")

    tq = pd.read_parquet("work/train_queries.parquet")
    tq = tq[tq.is_val | tq.is_rank]
    np.save("work/emb_queries_train.npy",
            encode([query_text(a, b) for a, b in zip(tq.search_query, tq.search_infm_params_text)], Q_LEN))
    tq[["query_id"]].reset_index(drop=True).to_parquet("work/emb_queries_train_ids.parquet")

    bq = pd.read_parquet("work/bench_queries.parquet")
    np.save("work/emb_queries_bench.npy",
            encode([query_text(a, b) for a, b in zip(bq.search_query, bq.search_infm_params_text)], Q_LEN))
    bq[["query_id"]].to_parquet("work/emb_queries_bench_ids.parquet")


if __name__ == "__main__":
    main()
