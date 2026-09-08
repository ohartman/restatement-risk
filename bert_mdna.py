#!/usr/bin/env python3
"""Fine-tune ModernBERT on MD&A text to predict restatements, on a 6 GB card.

The one free lever with a published gain over a forest is a language model that
reads the MD&A (+0.04 in Waffo Dzuyo et al. 2026, with an 8B model). This is the
version that fits a laptop GPU: ModernBERT-base (150M parameters, 8k context,
Apache-2.0), the first 2,048 tokens of Item 7, bf16, gradient checkpointing.

Honest by construction:
  train      10-Ks filed 2014-2016 with MD&A text (EDGAR-CORPUS gives all of them)
  select     AUC on 2017-2018 after every quarter epoch; best checkpoint kept
  score      every filing with text, saved aligned to the feature cache; the
             2019-2023 numbers are read once, by fusion_test.py

  python bert_mdna.py train [--epochs 2] [--max-len 2048]
  python bert_mdna.py score
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from relabel import submissions
from text_stack import load_text

MODEL = "answerdotai/ModernBERT-base"
CKPT = Path("data/out/bert_mdna.pt")
SCORES = Path("data/out/bert_score.npy")
TRAIN_YEARS, VAL_YEARS = (2014, 2015, 2016), (2017, 2018)


class Docs(Dataset):
    def __init__(self, ids, labels):
        self.ids, self.labels = ids, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def collate(batch, pad_id):
    ids, labels = zip(*batch)
    L = max(len(x) for x in ids)
    inp = torch.full((len(ids), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids), L), dtype=torch.long)
    for k, x in enumerate(ids):
        inp[k, :len(x)] = torch.tensor(x)
        att[k, :len(x)] = 1
    return inp, att, torch.tensor(labels, dtype=torch.float32)


def tokenize_all(tok, texts, max_len):
    out = []
    for i in range(0, len(texts), 256):
        enc = tok(texts[i:i + 256], truncation=True, max_length=max_len, add_special_tokens=True)
        out.extend(enc["input_ids"])
    return out


@torch.no_grad()
def predict(model, ids, pad_id, device, bs=8):
    model.eval()
    order = np.argsort([len(x) for x in ids])           # length-sorted batches waste less padding
    out = np.zeros(len(ids))
    for i in range(0, len(ids), bs):
        idx = order[i:i + bs]
        inp, att, _ = collate([(ids[j], 0.0) for j in idx], pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inp.to(device), attention_mask=att.to(device)).logits.squeeze(-1)
        out[idx] = logits.float().cpu().numpy()
    model.train()
    return out


def load_digest(adsh):
    """adsh -> regex-retrieved digest (digest_text.py), with its source recorded like load_text does."""
    import json
    text, source = {}, {}
    for l in open("data/raw/digest.jsonl", encoding="utf-8"):
        r = json.loads(l)
        text[r["adsh"]] = r["text"]; source[r["adsh"]] = r["source"]
    load_text.source = source
    print(f"  digests: {len(text):,} ({sum(v == 'corpus' for v in source.values()):,} corpus, {sum(v == 'fetch' for v in source.values()):,} fetch)")
    return text


def load_split(max_len, tok, use_digest=False):
    z = np.load("data/out/features.npz", allow_pickle=True)
    adsh, y, filed = z["adsh"].tolist(), z["y"], z["filed"].astype("datetime64[D]")
    years = filed.astype("datetime64[Y]").astype(int) + 1970
    text = load_digest(adsh) if use_digest else load_text(adsh, submissions())
    have = [i for i, a in enumerate(adsh) if a in text]
    print(f"text for {len(have):,} of {len(adsh):,} filings")
    t0 = time.time()
    ids = tokenize_all(tok, [text[adsh[i]] for i in have], max_len)
    print(f"tokenized in {time.time()-t0:.0f}s; median length {int(np.median([len(x) for x in ids]))} tokens")
    return adsh, y, years, have, ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["train", "score"])
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--tag", default="", help="suffix for checkpoint and score files")
    ap.add_argument("--digest", action="store_true", help="read the regex-retrieved digest instead of the MD&A opening")
    ap.add_argument("--no-checkpointing", action="store_true", help="skip gradient checkpointing (27% faster, needs ~4.4 GB at 512x8)")
    ap.add_argument("--evals-per-epoch", type=int, default=4)
    ap.add_argument("--score-from", type=int, default=0, help="score only filings from this year on (validation and test need 2017+)")
    ap.add_argument("--neg-per-pos", type=float, default=0, help="pilot: keep this many clean filings per restated one in train")
    ap.add_argument("--val-neg-per-pos", type=float, default=0, help="pilot: same for the validation years")
    args = ap.parse_args()
    ckpt = CKPT.with_name(CKPT.stem + args.tag + CKPT.suffix)
    scores_path = SCORES.with_name(SCORES.stem + args.tag + SCORES.suffix)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tok.pad_token_id
    adsh, y, years, have, ids = load_split(args.max_len, tok, use_digest=args.digest)
    have = np.array(have)
    yrs = years[have]
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(device)

    if args.mode == "score":
        model.load_state_dict(torch.load(ckpt, map_location=device))
        sel = np.flatnonzero(yrs >= args.score_from)
        s_sel = predict(model, [ids[i] for i in sel], pad_id, device)
        s = np.full(len(have), np.nan); s[sel] = s_sel
        out = np.full(len(adsh), np.nan); out[have] = s
        np.save(scores_path, out)
        for name, m in (("train 2014-16", np.isin(yrs, TRAIN_YEARS)), ("val 2017-18", np.isin(yrs, VAL_YEARS)),
                        ("test 2019-23", yrs >= 2019)):
            m = m & np.isfinite(s)
            if m.sum() and 0 < y[have][m].sum() < m.sum():
                print(f"  {name:<14} n {m.sum():>6,}  restated {int(y[have][m].sum()):>5,}  AUC {roc_auc_score(y[have][m], s[m]):.3f}")
        print(f"wrote {scores_path}")
        return

    # Corpus-sourced text only for training and validation (see text_stack.load_text).
    src = load_text.source
    corpus = np.array([src.get(adsh[i]) == "corpus" for i in have])
    tr = np.flatnonzero(np.isin(yrs, TRAIN_YEARS) & corpus); va = np.flatnonzero(np.isin(yrs, VAL_YEARS) & corpus)
    print(f"corpus-sourced: {int(corpus.sum()):,} of {len(have):,} filings with text")
    rng = np.random.default_rng(0)
    def subsample(idx, per_pos):
        if per_pos <= 0:
            return idx
        pos = idx[y[have][idx] == 1]; neg = idx[y[have][idx] == 0]
        keep = rng.choice(neg, min(len(neg), int(per_pos * len(pos))), replace=False)
        return np.sort(np.concatenate([pos, keep]))
    tr, va = subsample(tr, args.neg_per_pos), subsample(va, args.val_neg_per_pos)
    ytr, yva = y[have][tr], y[have][va]
    print(f"train {len(tr):,} ({int(ytr.sum())} restated)   validate {len(va):,} ({int(yva.sum())})")
    pos_weight = torch.tensor(min(10.0, (len(ytr) - ytr.sum()) / max(1, ytr.sum())), device=device)
    print(f"positive class weight {pos_weight.item():.1f}")

    if not args.no_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    ds = Docs([ids[i] for i in tr], ytr.tolist())
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=lambda b: collate(b, pad_id), drop_last=True)
    steps_per_epoch = len(dl) // args.accum
    total = int(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)) if s >= warm else (s + 1) / warm)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)
    val_ids = [ids[i] for i in va]

    best, step, t0, running = -1.0, 0, time.time(), 0.0
    done = False
    while not done:
        for k, (inp, att, lab) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=inp.to(device), attention_mask=att.to(device)).logits.squeeze(-1)
            loss = loss_fn(logits.float(), lab.to(device)) / args.accum
            loss.backward()
            running += loss.item()
            if (k + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    print(f"  step {step}/{total}  loss {running/25:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                          f"[{(time.time()-t0)/60:.0f} min]", flush=True)
                    running = 0.0
                if step % eval_every == 0 or step == total:
                    s = predict(model, val_ids, pad_id, device)
                    auc = roc_auc_score(yva, s)
                    per = "  ".join(f"{v}: {roc_auc_score(yva[yrs[va]==v], s[yrs[va]==v]):.3f}" for v in VAL_YEARS)
                    flag = ""
                    if auc > best:
                        best = auc; torch.save(model.state_dict(), ckpt); flag = "  <- saved"
                    print(f"== step {step} ({step/steps_per_epoch:.2f} epochs)  val AUC {auc:.3f}  ({per}){flag}", flush=True)
                if step >= total:
                    done = True
                    break
    print(f"\nbest validation AUC {best:.3f}; checkpoint at {CKPT}")


if __name__ == "__main__":
    main()
