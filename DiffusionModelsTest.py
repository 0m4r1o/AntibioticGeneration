# smiles_diffusion.py
# Diffusion over SMILES token embeddings with a Transformer denoiser.
# Hardcoded config, verbose logs, checkpointing, and matplotlib plots.

# =========================
# ===== CONFIG (EDIT) =====
# =========================
# MODE = "train"            # "train" or "generate"
MODE = "generate"            # "train" or "generate"

# Data & IO
DATA_PATH = "data/all_smiles_clean.smi"      # one SMILES per line (used only in train)
SAVE_DIR  = "Resuls/Models/Diffusion"        # checkpoints & outputs (auto-created)

# Model
MAX_LEN   = 128
D_MODEL   = 256
NHEAD     = 8
LAYERS    = 6

# Diffusion
TIMESTEPS = 1000
SCHEDULE  = "cosine"      # "cosine" or "linear"

# Training
EPOCHS     = 120         # for real runs, 100 is recommended
BATCH_SIZE = 64
LR         = 2e-4
VAL_SPLIT  = 0.05
SEED       = 42

# Generation
NUM_SAMPLES = 100
DDIM_ETA    = 0.0         # 0.0 = deterministic DDIM

# Verbosity
VERBOSE_EVERY = 50         # print every N batches

# =========================
# ===== IMPLEMENTATION ====
# =========================
import json
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure output directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# -------- Tokenizer --------
SPECIAL_TOKENS = {"PAD": "<pad>", "BOS": "<bos>", "EOS": "<eos>"}

def build_vocab(smiles_list: List[str]) -> List[str]:
    chars = set()
    for s in smiles_list:
        for ch in s.strip():
            chars.add(ch)
    vocab = [SPECIAL_TOKENS["PAD"], SPECIAL_TOKENS["BOS"], SPECIAL_TOKENS["EOS"]] + sorted(list(chars))
    return vocab

class SmilesTokenizer:
    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.stoi = {s: i for i, s in enumerate(vocab)}
        self.itos = {i: s for i, s in enumerate(vocab)}
        self.pad_id = self.stoi[SPECIAL_TOKENS["PAD"]]
        self.bos_id = self.stoi[SPECIAL_TOKENS["BOS"]]
        self.eos_id = self.stoi[SPECIAL_TOKENS["EOS"]]

    def encode(self, text: str, max_len: int):
        ids = [self.bos_id] + [self.stoi.get(ch, self.pad_id) for ch in text.strip()] + [self.eos_id]
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.eos_id: break
            if i in (self.pad_id, self.bos_id): continue
            out.append(self.itos.get(i, ""))
        return "".join(out)

# -------- Dataset --------
class SmilesDataset(Dataset):
    def __init__(self, smiles: List[str], tokenizer: SmilesTokenizer, max_len: int):
        self.smiles = smiles
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        s = self.smiles[idx]
        ids = self.tok.encode(s, self.max_len)
        return torch.tensor(ids, dtype=torch.long)

# -------- Diffusion schedule (DDPM-like) --------
def cosine_beta_schedule(timesteps, s=0.008):
    # https://openreview.net/forum?id=-NEXDKk8gZ
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-5, 0.999)

class DiffusionSchedule:
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine"):
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = np.linspace(1e-4, 0.02, timesteps)
        else:
            raise ValueError("Unknown schedule")
        self.timesteps = timesteps
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

# -------- Model: Transformer denoiser --------
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.lin = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * (-math.log(10000) / (half - 1)))
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.lin(emb)

class Denoiser(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, num_layers: int = 6, max_len: int = 128):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.time_emb = TimeEmbedding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.to_noise = nn.Linear(d_model, d_model)      # predict noise in embedding space
        self.proj = nn.Linear(d_model, vocab_size, bias=False)  # tie to token_emb
        self.proj.weight = self.token_emb.weight

    def forward(self, noisy_emb: torch.Tensor, t: torch.Tensor):
        B, L, D = noisy_emb.shape
        pos_ids = torch.arange(L, device=noisy_emb.device).unsqueeze(0).expand(B, L)
        x = noisy_emb + self.pos_emb(pos_ids) + self.time_emb(t)[:, None, :]
        h = self.transformer(x)
        eps_hat = self.to_noise(h)
        return eps_hat

    def decode_logits(self, emb: torch.Tensor):
        return self.proj(emb)

# -------- Utilities --------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def prepare_data(data_path: str, max_len: int, val_split: float = 0.05) -> Tuple[List[str], List[str]]:
    smiles = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                smiles.append(s)
    random.shuffle(smiles)
    n_val = max(1, int(len(smiles) * val_split))
    val = smiles[:n_val]
    train = smiles[n_val:]
    return train, val

# -------- Training --------
def train_loop():
    print(f"[INFO] Device check: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"[INFO] Save dir: {SAVE_DIR}")
    history_train = []
    history_val = []

    device = get_device()
    seed_everything(SEED)

    # Data
    train_smiles, val_smiles = prepare_data(DATA_PATH, MAX_LEN, VAL_SPLIT)
    vocab = build_vocab(train_smiles + val_smiles)
    tok = SmilesTokenizer(vocab)

    train_ds = SmilesDataset(train_smiles, tok, MAX_LEN)
    val_ds = SmilesDataset(val_smiles, tok, MAX_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"[INFO] Train size: {len(train_ds)} | Val size: {len(val_ds)}")
    print(f"[INFO] Hyperparams: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, "
          f"timesteps={TIMESTEPS}, schedule={SCHEDULE}, d_model={D_MODEL}, layers={LAYERS}, nhead={NHEAD}")

    # Diffusion & model
    sched = DiffusionSchedule(timesteps=TIMESTEPS, schedule=SCHEDULE)
    model = Denoiser(vocab_size=len(vocab), d_model=D_MODEL, nhead=NHEAD, num_layers=LAYERS, max_len=MAX_LEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    # Save config & vocab
    with open(Path(SAVE_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab}, f, ensure_ascii=False, indent=2)
    with open(Path(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "max_len": MAX_LEN, "d_model": D_MODEL, "nhead": NHEAD, "layers": LAYERS,
            "timesteps": TIMESTEPS, "schedule": SCHEDULE, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "val_split": VAL_SPLIT, "seed": SEED
        }, f, ensure_ascii=False, indent=2)

    def diffusion_step(batch_ids):
        with torch.no_grad():
            emb_x = model.token_emb(batch_ids)  # (B,L,D)
        B, L, D = emb_x.shape
        t = torch.randint(0, sched.timesteps, (B,), device=device, dtype=torch.long)
        a_bar = sched.alphas_cumprod.to(device)[t]
        a_bar_sqrt = a_bar.sqrt().view(B,1,1)
        one_minus = (1.0 - a_bar).sqrt().view(B,1,1)
        eps = torch.randn_like(emb_x)
        z_t = a_bar_sqrt * emb_x + one_minus * eps
        eps_hat = model(z_t, t)
        loss = F.mse_loss(eps_hat, eps)
        return loss

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for bi, batch in enumerate(train_dl, 1):
            batch = batch.to(device)
            loss = diffusion_step(batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item()
            if bi % VERBOSE_EVERY == 0:
                print(f"    [Epoch {epoch:03d} | Batch {bi:05d}] loss={loss.item():.4f}")

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                batch = batch.to(device)
                loss = diffusion_step(batch)
                va_loss += loss.item()

        tr_loss /= max(1, len(train_dl))
        va_loss /= max(1, len(val_dl))
        print(f"[Epoch {epoch:03d}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")

        history_train.append(tr_loss)
        history_val.append(va_loss)

        # Checkpoints
        ckpt = {"model_state": model.state_dict(),
                "args": {"max_len": MAX_LEN, "d_model": D_MODEL, "nhead": NHEAD, "layers": LAYERS,
                         "timesteps": TIMESTEPS, "schedule": SCHEDULE},
                "vocab": vocab}
        torch.save(ckpt, Path(SAVE_DIR, "model.pt"))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, Path(SAVE_DIR, "model_best.pt"))

    print("Training complete →", SAVE_DIR)

    # --- Plot & save training curves ---
    try:
        # Loss curve
        plt.figure()
        plt.plot(range(1, len(history_train)+1), history_train, label="train_loss")
        plt.plot(range(1, len(history_val)+1), history_val, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Diffusion Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(SAVE_DIR, "loss_curve.png"))
        plt.close()

        # Save CSV of history
        with open(Path(SAVE_DIR, "training_history.csv"), "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss\n")
            for i, (tr, va) in enumerate(zip(history_train, history_val), 1):
                f.write(f"{i},{tr:.6f},{va:.6f}\n")
        print(f"[INFO] Saved loss curves and history to {SAVE_DIR}")
    except Exception as e:
        print("[WARN] Could not save training plots:", e)

# -------- Sampling --------
@torch.no_grad()
def sample_embeddings(model, sched, shape, device, ddim_eta=0.0):
    B, L, D = shape
    z = torch.randn(shape, device=device)
    for t in reversed(range(sched.timesteps)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        eps = model(z, t_tensor)
        a_bar_t = sched.alphas_cumprod.to(device)[t]
        a_t = sched.alphas.to(device)[t]
        sqrt_one_minus_a_bar_t = sched.sqrt_one_minus_alphas_cumprod.to(device)[t]
        sqrt_a_bar_t = sched.sqrt_alphas_cumprod.to(device)[t]
        x0 = (z - sqrt_one_minus_a_bar_t * eps) / (sqrt_a_bar_t + 1e-8)
        if t > 0:
            a_bar_prev = sched.alphas_cumprod.to(device)[t - 1]
            sigma_t = ddim_eta * (((1 - a_bar_prev) / (1 - a_bar_t) * (1 - a_t)) ** 0.5)
            c = ((1 - a_bar_prev - sigma_t**2) ** 0.5)
            z = (a_bar_prev**0.5) * x0 + c * eps + sigma_t * torch.randn_like(z)
        else:
            z = x0
    return z

@torch.no_grad()
def decode_embeddings_to_smiles(model, tok, emb):
    logits = model.decode_logits(emb)        # (B, L, V)
    ids = logits.argmax(dim=-1).tolist()
    return [tok.decode(seq) for seq in ids]

def generate_loop():
    device = get_device()
    # load best or last checkpoint
    ckpt_path = Path(SAVE_DIR, "model_best.pt")
    if not ckpt_path.exists():
        ckpt_path = Path(SAVE_DIR, "model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    vocab = ckpt["vocab"]
    tok = SmilesTokenizer(vocab)
    model_args = ckpt["args"]
    d_model = model_args.get("d_model", D_MODEL)
    nhead = model_args.get("nhead", NHEAD)
    layers = model_args.get("layers", LAYERS)
    max_len = model_args.get("max_len", MAX_LEN)

    model = Denoiser(vocab_size=len(vocab), d_model=d_model, nhead=nhead, num_layers=layers, max_len=max_len).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sched = DiffusionSchedule(timesteps=model_args.get("timesteps", TIMESTEPS), schedule=model_args.get("schedule", SCHEDULE))
    emb = sample_embeddings(model, sched, (NUM_SAMPLES, max_len, d_model), device=device, ddim_eta=DDIM_ETA)
    smiles = decode_embeddings_to_smiles(model, tok, emb)

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = Path(SAVE_DIR, f"generated_diffusion_{NUM_SAMPLES}.smi")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in smiles:
            f.write(s + "\n")
    print(f"Saved {len(smiles)} SMILES → {out_path}")

# -------- Main --------
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    if MODE == "train":
        train_loop()
    elif MODE == "generate":
        generate_loop()
    else:
        raise ValueError("MODE must be 'train' or 'generate'")
