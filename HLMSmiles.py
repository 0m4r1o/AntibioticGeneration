# Hierarchical LSTM (HLM) for SMILES — PyTorch single-file script with hardcoded variables and metrics saving
# ------------------------------------------------------------------------------------------
# Adds saving of training metrics (loss, accuracy) to CSV.


import math
import os
from pathlib import Path
import random
import csv
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Hardcoded configuration
# -----------------------
MODE = 'train'             # 'train' or 'generate'
DATA_PATH = 'data/all_smiles_clean.smi'
SAVE_DIR = 'Results/HLM_SMILES'
MAX_LEN = 128
BLOCK_SIZE = 16
EMB_DIM = 256
LOWER_HID = 512
UPPER_HID = 512
BATCH_SIZE = 64
EPOCHS = 20
LR = 2e-4
DROPOUT = 0.1
SEED = 42
VAL_SPLIT = 0.05
NUM_SAMPLES = 50
TEMPERATURE = 1.0
TOP_K = 0

# -----------------------
# Utils and Tokenization
# -----------------------
SPECIAL = {"PAD": " ", "BOS": "<", "EOS": ">"}


class SmilesTokenizer:
    def __init__(self, smiles_list: List[str]):
        chars = set()
        for s in smiles_list:
            for ch in s.strip():
                chars.add(ch)
        self.vocab = [SPECIAL["PAD"], SPECIAL["BOS"], SPECIAL["EOS"]] + sorted(list(chars))
        self.stoi = {s:i for i,s in enumerate(self.vocab)}
        self.itos = {i:s for i,s in enumerate(self.vocab)}
        self.pad_id = self.stoi[SPECIAL["PAD"]]
        self.bos_id = self.stoi[SPECIAL["BOS"]]
        self.eos_id = self.stoi[SPECIAL["EOS"]]

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = [self.bos_id] + [self.stoi.get(ch, self.pad_id) for ch in text.strip()] + [self.eos_id]
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i == self.eos_id: break
            if i in (self.pad_id, self.bos_id): continue
            out.append(self.itos.get(i, ""))
        return "".join(out)

# -----------------------
# Dataset
# -----------------------
class SmilesDataset(Dataset):
    def __init__(self, smiles: List[str], tok: SmilesTokenizer, max_len: int):
        self.smiles = smiles
        self.tok = tok
        self.max_len = max_len
    def __len__(self):
        return len(self.smiles)
    def __getitem__(self, idx):
        s = self.smiles[idx]
        ids = self.tok.encode(s, self.max_len)
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:],  dtype=torch.long)
        return x, y

# -----------------------
# Hierarchical LSTM Model
# -----------------------
class HierarchicalLM(nn.Module):
    def __init__(self, vocab_size:int, emb_dim:int=256, lower_hid:int=512, upper_hid:int=512, \
                 block_size:int=16, num_layers_lower:int=1, num_layers_upper:int=1, dropout:float=0.1):
        super().__init__()
        self.block_size = block_size
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.upper_lstm = nn.LSTM(input_size=emb_dim, hidden_size=upper_hid, num_layers=num_layers_upper,
                                  batch_first=True, dropout=dropout if num_layers_upper>1 else 0.0)
        self.lower_in = emb_dim + upper_hid
        self.lower_lstm = nn.LSTM(input_size=self.lower_in, hidden_size=lower_hid, num_layers=num_layers_lower,
                                  batch_first=True, dropout=dropout if num_layers_lower>1 else 0.0)
        self.proj = nn.Linear(lower_hid, vocab_size)

    def forward(self, x):
        B, L = x.shape
        E = self.emb(x)
        bs = self.block_size
        num_blocks = (L + bs - 1) // bs
        pad_len = num_blocks*bs - L
        if pad_len > 0:
            last = E[:, -1:, :].expand(B, pad_len, E.size(-1))
            E_blocks = torch.cat([E, last], dim=1)
        else:
            E_blocks = E
        E_blocks = E_blocks.view(B, num_blocks, bs, -1)
        block_repr = E_blocks.mean(dim=2)
        upper_out, _ = self.upper_lstm(block_repr)
        ctx = upper_out.unsqueeze(2).expand(B, num_blocks, bs, upper_out.size(-1)).contiguous()
        ctx = ctx.view(B, num_blocks*bs, upper_out.size(-1))[:, :L, :]
        lower_in = torch.cat([E, ctx], dim=-1)
        lower_out, _ = self.lower_lstm(lower_in)
        logits = self.proj(lower_out)
        return logits

# -----------------------
# Helper Functions
# -----------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_smiles(path: str) -> List[str]:
    smiles = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                smiles.append(s)
    return smiles

@torch.no_grad()
def evaluate(model, loader, device, pad_id:int):
    model.eval()
    tot_loss, tot_tok, correct, total = 0.0, 0, 0, 0
    ce = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        logits = model(xb)
        loss = ce(logits.view(-1, logits.size(-1)), yb.view(-1))
        preds = logits.argmax(dim=-1)
        mask = (yb != pad_id)
        correct += (preds[mask] == yb[mask]).sum().item()
        total += mask.sum().item()
        tot_loss += loss.item(); tot_tok += total
    ppl = math.exp(tot_loss / max(1, total))
    acc = 100.0 * correct / max(1, total)
    return tot_loss / max(1, total), ppl, acc

@torch.no_grad()
def generate(model, tok: SmilesTokenizer, max_len:int, num_samples:int, device, temperature:float=1.0, top_k:int=0):
    model.eval()
    bos = tok.bos_id; eos = tok.eos_id; pad = tok.pad_id
    samples = []
    for _ in range(num_samples):
        seq = torch.full((1, max_len), pad, dtype=torch.long, device=device)
        seq[0,0] = bos
        for t in range(1, max_len):
            logits = model(seq[:, :t])[:, -1, :] / max(1e-6, temperature)
            if top_k > 0:
                v, ix = torch.topk(logits, k=min(top_k, logits.size(-1)))
                probs = torch.zeros_like(logits).scatter_(1, ix, F.softmax(v, dim=-1))
            else:
                probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            seq[0, t] = next_id
            if next_id.item() == eos:
                break
        samples.append(tok.decode(seq[0].tolist()))
    return samples

# -----------------------
# Main Logic
# -----------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if MODE == 'train':
        smiles = load_smiles(DATA_PATH)
        random.shuffle(smiles)
        n_val = max(1, int(len(smiles) * VAL_SPLIT))
        val = smiles[:n_val]
        train = smiles[n_val:]
        tok = SmilesTokenizer(train + val)

        train_ds = SmilesDataset(train, tok, MAX_LEN)
        val_ds = SmilesDataset(val, tok, MAX_LEN)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = HierarchicalLM(vocab_size=len(tok.vocab), emb_dim=EMB_DIM, lower_hid=LOWER_HID,
                               upper_hid=UPPER_HID, block_size=BLOCK_SIZE, dropout=DROPOUT).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        ce = nn.CrossEntropyLoss(ignore_index=tok.pad_id)

        metrics_path = Path(SAVE_DIR, 'training_metrics.csv')
        with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_ppl', 'val_acc'])

        best_val = float('inf')
        for epoch in range(1, EPOCHS+1):
            model.train()
            tot = 0.0
            for xb, yb in train_dl:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                loss = ce(logits.view(-1, logits.size(-1)), yb.view(-1))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
            tr_loss = tot / max(1, len(train_dl))
            va_nll, va_ppl, va_acc = evaluate(model, val_dl, device, tok.pad_id)
            print(f"[E{epoch:03d}] train_loss={tr_loss:.4f}  val_nll/tok={va_nll:.4f}  val_ppl={va_ppl:.2f}  val_acc={va_acc:.2f}%")

            with open(metrics_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, tr_loss, va_nll, va_ppl, va_acc])

            ckpt = {'model_state': model.state_dict(), 'vocab': tok.vocab}
            torch.save(ckpt, Path(SAVE_DIR, 'hlm_last.pt'))
            if va_nll < best_val:
                best_val = va_nll
                torch.save(ckpt, Path(SAVE_DIR, 'hlm_best.pt'))

        print(f"[DONE] Training finished → {SAVE_DIR}")

    elif MODE == 'generate':
        ckpt_path = Path(SAVE_DIR, 'hlm_best.pt')
        if not ckpt_path.exists():
            ckpt_path = Path(SAVE_DIR, 'hlm_last.pt')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        vocab = ckpt['vocab']
        tok = SmilesTokenizer(["C"])
        tok.vocab = vocab
        tok.stoi = {s:i for i,s in enumerate(vocab)}
        tok.itos = {i:s for i,s in enumerate(vocab)}
        tok.pad_id = tok.stoi[SPECIAL['PAD']]
        tok.bos_id = tok.stoi[SPECIAL['BOS']]
        tok.eos_id = tok.stoi[SPECIAL['EOS']]

        model = HierarchicalLM(vocab_size=len(vocab), emb_dim=EMB_DIM, lower_hid=LOWER_HID,
                               upper_hid=UPPER_HID, block_size=BLOCK_SIZE).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        smiles = generate(model, tok, max_len=MAX_LEN, num_samples=NUM_SAMPLES, device=device,
                          temperature=TEMPERATURE, top_k=TOP_K)
        out_path = Path(SAVE_DIR, f"hlm_samples_{NUM_SAMPLES}.smi")
        with open(out_path, 'w', encoding='utf-8') as f:
            for s in smiles:
                f.write(s + "\n")
        print(f"[SAVE] {len(smiles)} SMILES → {out_path}")

if __name__ == '__main__':
    main()
