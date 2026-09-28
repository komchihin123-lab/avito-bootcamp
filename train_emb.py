"""
Шаг 2. Дообучение bi-encoder'а (запрос -> объявление) на парах из train.

Идея: BM25 не ловит синонимы/перефразы ("очистка ног" -> "Педикюр", "гувернер" -> "Няня").
Дообучаем открытую русскоязычную модель эмбеддингов (по умолчанию intfloat/multilingual-e5-small,
open-source, скачивается с HuggingFace один раз и далее работает локально) контрастивной
функцией потерь с in-batch негативами (InfoNCE / MultipleNegativesRanking):
для запроса правильное объявление должно быть ближе, чем объявления других запросов в батче.

Важно: пары валидационных и "ранкерных" групп в обучение НЕ попадают (иначе утечка).

Выход: work/emb_model/ (веса), далее encode_emb.py считает эмбеддинги.
Запуск (GPU): .venv_gpu/Scripts/python train_emb.py
"""
import os, sys, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

BASE_MODEL = os.environ.get("BASE_MODEL", "intfloat/multilingual-e5-small")
OUT = "work/emb_model"
BATCH = int(os.environ.get("BATCH", 64))
EPOCHS = float(os.environ.get("EPOCHS", 1))
LR = 3e-5
Q_LEN, D_LEN = 32, 96
# e5-модели обучены с префиксами "query: " / "passage: "
Q_PREFIX, D_PREFIX = "query: ", "passage: "
MAX_PER_TEXT = 30  # ограничиваем число пар на один популярный текст запроса
TEMP = 0.05
SEED = 0

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = os.environ.get("FP16", "0") == "1"  # на GTX 16xx fp16 оказался медленнее fp32


def query_text(q, params):
    """Текст запроса для модели: сам запрос + фильтры (если есть)."""
    q = Q_PREFIX + str(q)
    return q + (" | " + params if isinstance(params, str) and params.strip() else "")


def item_text(title, params, desc, max_desc=300):
    """Текст объявления: заголовок + параметры + начало описания."""
    t = str(title or "")
    p = str(params or "")
    d = str(desc or "")[:max_desc].replace("\n", " ")
    return f"{D_PREFIX}{t} | {p} | {d}"


def mean_pool(out, mask):
    h = out.last_hidden_state
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)


def main():
    pairs = pd.read_parquet("work/train_pairs.parquet")
    pairs = pairs[~pairs.is_val & ~pairs.is_rank]
    tq = pd.read_parquet("work/train_queries.parquet", columns=["gid", "search_infm_params_text"])
    pairs = pairs.merge(tq, on="gid")
    # не больше MAX_PER_TEXT пар на один текст запроса
    pairs = pairs.sample(frac=1, random_state=SEED)
    pairs = pairs[pairs.groupby("search_query").cumcount() < MAX_PER_TEXT]
    items = pd.read_parquet("work/train_items.parquet",
                            columns=["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"])
    items = items.set_index("item_id")
    it = items.loc[pairs.item_id]
    qs = [query_text(a, b) for a, b in zip(pairs.search_query, pairs.search_infm_params_text)]
    ds = [item_text(a, b, c) for a, b, c in zip(it.item_title_raw, it.item_infm_params_text, it.item_description_raw)]
    qtexts = pairs.search_query.to_numpy(dtype=object)
    del items, it, pairs, tq
    import gc; gc.collect()
    print("pairs for training:", len(qs), flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModel.from_pretrained(BASE_MODEL).to(dev)
    # матрицу словарных эмбеддингов (250k x 384) замораживаем: экономит память/время,
    # а качество почти не меняется
    model.embeddings.word_embeddings.weight.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    n_steps = int(math.ceil(len(qs) / BATCH) * EPOCHS)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * max(0.0, 1 - s / n_steps))
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda" and FP16))

    def enc(texts, L):
        b = tok(texts, padding=True, truncation=True, max_length=L, return_tensors="pt").to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda" and FP16)):
            return F.normalize(mean_pool(model(**b), b["attention_mask"]).float(), dim=-1)

    model.train()
    step = 0
    order = np.arange(len(qs))
    while step < n_steps:
        np.random.shuffle(order)
        for s in range(0, len(order) - BATCH + 1, BATCH):
            bi = order[s:s + BATCH]
            qe = enc([qs[i] for i in bi], Q_LEN)
            de = enc([ds[i] for i in bi], D_LEN)
            logits = qe @ de.T / TEMP
            # одинаковые тексты запросов в батче — не считаем их негативами друг для друга
            qt = qtexts[bi]
            same = torch.tensor(qt[:, None] == qt[None, :], device=dev) & ~torch.eye(len(bi), dtype=torch.bool, device=dev)
            logits = logits.masked_fill(same, -1e4)
            lbl = torch.arange(len(bi), device=dev)
            loss = (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.T, lbl)) / 2
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % 200 == 0 or step == 20:
                print(f"step {step}/{n_steps} loss {loss.item():.4f}", flush=True)
            if step >= n_steps:
                break
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
