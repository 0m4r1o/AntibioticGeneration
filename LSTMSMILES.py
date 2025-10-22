#!/usr/bin/env python
# LSTMSMILES.py
# LSTM (with BatchNorm) SMILES generator: arch search, train, evaluate, sample, and transfer learning.

import os
# --- GPU-friendly env toggles BEFORE importing TensorFlow (no logic change) ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")   # reduce TF log noise
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # avoid oneDNN path quirks on Win+TF2.10
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem
try:
    from rdkit import Chem
    RDKit_OK = True
except Exception:
    RDKit_OK = False

# --- Enable GPU memory growth (so TF doesn't grab all VRAM) ---
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        for _g in _gpus:
            tf.config.experimental.set_memory_growth(_g, True)
        print("[TF] GPUs:", _gpus)
    except Exception as _e:
        print("[TF] Could not enable memory growth:", _e)
else:
    print("[TF] No GPU visible; running on CPU.")

# =========================
# HARD-CODED CONFIG
# =========================
generation = "0"
DATA_PATH = "all_smiles_clean.smi"        # path to training SMILES (one per line)
# DATA_PATH = "all_smiles_clean.smi"        # path to training SMILES (one per line)
SAVE_DIR  = f"Results/smiles_lstm_bn_ckpt_{generation}"         # output folder (model, vocab, logs, metrics, samples)
os.makedirs(SAVE_DIR, exist_ok=True)

# Training
MAX_EPOCHS          = 100
BATCH_SIZE          = 128
VAL_SPLIT           = 0.1
SEED                = 42
SHUFFLE_BUFFER      = 10000
EARLY_STOP_PATIENCE = 6
LEARNING_RATE       = 3e-4
MODEL_FILENAME = "model.keras"  # Adjust if needed
# Tokens & sequence
BOS = "<"
EOS = ">"
PAD = " "       # single space; used as left-pad (index 0)

# Tiny search space (fast sanity-check)
SEARCH_MAX_EPOCHS = 10
CANDIDATES = [
    # (embedding_dim, lstm1, lstm2, dropout)
    (128, 256, 256, 0.2),
    (128, 384, 256, 0.3),
    (192, 384, 384, 0.3),
    (256, 512, 512, 0.35),
]

# Generation
GEN_MAX_LEN      = 128
GEN_NUM_SAMPLES  = 50
GEN_TEMPERATURE  = 0.9
GEN_TOP_P        = 0.95  # nucleus sampling

# =========================
# UTILITIES
# =========================
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_smiles_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines

def clean_and_tag(smiles: List[str]) -> List[str]:
    # Add BOS/EOS; defer length control to data pipeline
    return [f"{BOS}{s}{EOS}" for s in smiles]

class Vocab:
    def __init__(self, char_to_idx: Dict[str, int], idx_to_char: Dict[int, str], vocab_size: int, max_len: int):
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size  = vocab_size
        self.max_len     = max_len

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "char_to_idx": self.char_to_idx,
                "idx_to_char": {str(k): v for k, v in self.idx_to_char.items()},
                "vocab_size" : self.vocab_size,
                "max_len"    : self.max_len
            }, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        idx_to_char = {int(k): v for k, v in d["idx_to_char"].items()}
        return Vocab(d["char_to_idx"], idx_to_char, d["vocab_size"], d["max_len"])

def build_vocab(smiles_tagged: List[str]) -> Vocab:
    # Ensure PAD is index 0
    chars = sorted(set("".join(smiles_tagged + [PAD])))
    if PAD in chars:
        chars.remove(PAD)
    chars = [PAD] + chars
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    lengths = [len(s) for s in smiles_tagged]
    # Use robust max_len (95th percentile) to avoid over-long padding
    max_len = int(np.clip(np.percentile(lengths, 95), 32, 256))
    return Vocab(char_to_idx, idx_to_char, len(chars), max_len)

def encode(smile: str, vocab: Vocab) -> List[int]:
    return [vocab.char_to_idx.get(c, 0) for c in smile]

def make_inputs_targets(smiles_tagged: List[str], vocab: Vocab) -> Tuple[np.ndarray, np.ndarray]:
    """
    Teacher forcing next-char prediction.
      input:  [BOS, ..., last-1] left-padded to (max_len-1)
      target: [ ..., EOS]        left-padded to (max_len-1)
    """
    X, Y = [], []
    for s in smiles_tagged:
        s = s[:vocab.max_len]          # cap at max_len
        inp = s[:-1]
        tgt = s[1:]

        x_ids = encode(inp, vocab)
        y_ids = encode(tgt, vocab)

        need = (vocab.max_len - 1) - len(x_ids)
        if need > 0:
            x_ids = [0] * need + x_ids
            y_ids = [0] * need + y_ids
        else:
            x_ids = x_ids[:(vocab.max_len - 1)]
            y_ids = y_ids[:(vocab.max_len - 1)]

        X.append(x_ids)
        Y.append(y_ids)

    return np.array(X, dtype=np.int32), np.array(Y, dtype=np.int32)

def make_tf_dataset(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(min(len(X), SHUFFLE_BUFFER), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# =========================
# MODEL (LSTM + BatchNorm)
# =========================
def build_model(vocab_size: int, max_len: int,
                embed_dim: int, lstm1: int, lstm2: int, dropout: float,
                lr: float = LEARNING_RATE) -> tf.keras.Model:
    """
    Embedding → LSTM → BatchNorm → Dropout → LSTM → BatchNorm → Dropout → TimeDistributed(Dense vocab)
    """
    inputs = tf.keras.Input(shape=(max_len - 1,), dtype="int32")
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)(inputs)

    # --- Only change here: recurrent_dropout set to 0.0 to enable cuDNN fast path ---
    x = tf.keras.layers.LSTM(lstm1, return_sequences=True, recurrent_dropout=0.0)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # --- Same here (second LSTM) ---
    x = tf.keras.layers.LSTM(lstm2, return_sequences=True, recurrent_dropout=0.0)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size))(x)

    model = tf.keras.Model(inputs, logits, name="smiles_lstm_bn")
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

# =========================
# TRAIN / SEARCH / EVAL
# =========================
def train_model(model: tf.keras.Model, train_ds, val_ds, max_epochs=MAX_EPOCHS, ckpt_path=None):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(SAVE_DIR, "training_log.csv"), append=False),
    ]
    if ckpt_path:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                ckpt_path, monitor="val_loss", save_best_only=True, save_weights_only=False
            )
        )
    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        verbose=2,
        callbacks=callbacks
    )
    print("Finished training!")
    return history

def evaluate_model(model: tf.keras.Model, val_ds):
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    return {"val_loss": float(val_loss), "val_acc": float(val_acc)}

def simple_arch_search(X: np.ndarray, Y: np.ndarray, vocab: Vocab) -> Tuple[tuple, dict]:
    """Train a few epochs on several candidates; pick best val_loss."""
    print("[SEARCH] Starting tiny architecture search over", len(CANDIDATES), "candidates...")
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int((1.0 - VAL_SPLIT) * n)
    tr_idx, va_idx = idx[:split], idx[split:]
    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_va, Y_va = X[va_idx], Y[va_idx]
    ds_tr = make_tf_dataset(X_tr, Y_tr, batch_size=BATCH_SIZE, shuffle=True)
    ds_va = make_tf_dataset(X_va, Y_va, batch_size=BATCH_SIZE, shuffle=False)

    best_cfg = None
    best_val = float("inf")
    logs = {}

    for cfg in CANDIDATES:
        embed_dim, lstm1, lstm2, dropout = cfg
        print(f"[SEARCH] Candidate: embed={embed_dim}, lstm=({lstm1},{lstm2}), dropout={dropout}")
        model = build_model(vocab.vocab_size, vocab.max_len, embed_dim, lstm1, lstm2, dropout)
        h = model.fit(ds_tr, validation_data=ds_va, epochs=SEARCH_MAX_EPOCHS, verbose=1)
        val_loss = h.history["val_loss"][-1]
        logs[str(cfg)] = float(val_loss)
        print(f"[SEARCH]   -> val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_cfg = cfg

    print(f"[SEARCH] Best: {best_cfg} with val_loss={best_val:.4f}")
    return best_cfg, logs

# =========================
# SAMPLING
# =========================
def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def _nucleus_sample(probs: np.ndarray, top_p: float) -> int:
    idx = np.argsort(probs)[::-1]
    sorted_p = probs[idx]
    cumsum = np.cumsum(sorted_p)
    cutoff = np.searchsorted(cumsum, top_p)
    cutoff = max(1, cutoff)
    chosen = idx[:cutoff]
    renorm = sorted_p[:cutoff] / np.sum(sorted_p[:cutoff])
    return int(np.random.choice(chosen, p=renorm))

def _sample_next(logits: np.ndarray, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))
    scaled = logits / max(1e-6, temperature)
    probs = _softmax(scaled)
    if top_p < 1.0:
        return _nucleus_sample(probs, top_p)
    return int(np.random.choice(len(probs), p=probs))

def generate_one(model: tf.keras.Model, vocab: Vocab,
                 max_len: int = GEN_MAX_LEN,
                 temperature: float = GEN_TEMPERATURE,
                 top_p: float = GEN_TOP_P) -> str:
    seq = [vocab.char_to_idx[BOS]]
    for _ in range(max_len):
        x = seq[-(vocab.max_len - 1):]
        need = (vocab.max_len - 1) - len(x)
        if need > 0:
            x = [0] * need + x
        x = np.array(x, dtype=np.int32)[None, :]  # (1, L)
        logits = model.predict(x, verbose=0)[0, -1]  # last-step dist
        idx = _sample_next(logits, temperature, top_p)
        ch  = vocab.idx_to_char.get(idx, PAD)
        seq.append(idx)
        if ch == EOS:
            break
    # decode, strip BOS/EOS/PAD
    chars = [vocab.idx_to_char.get(i, PAD) for i in seq]
    return "".join(c for c in chars if c not in (BOS, EOS, PAD))

def generate_many(model: tf.keras.Model, vocab: Vocab, n: int = GEN_NUM_SAMPLES,
                  max_len: int = GEN_MAX_LEN, temperature: float = GEN_TEMPERATURE, top_p: float = GEN_TOP_P) -> List[str]:
    out = []
    for i in range(1, n + 1):
        s = generate_one(model, vocab, max_len=max_len, temperature=temperature, top_p=top_p)
        print(f"[GEN] [{i}/{n}] {s}")
        out.append(s)
    return out

# =========================
# SMILES STATS (validity/uniqueness)
# =========================


def smiles_stats(smiles_list: List[str]):
    """
    validation % = (# valid SMILES) / total * 100
    uniqueness % = (# unique canonical among valid) / (# valid) * 100
    returns dict with clean_unique_smiles (canonical, deduped)
    """
    if not smiles_list:
        return {
            "total": 0, "valid_count": 0, "validation_pct": 0.0,
            "unique_count": 0, "uniqueness_pct": 0.0, "clean_unique_smiles": []
        }
    if not RDKit_OK:
        # Fallback when RDKit isn't installed
        uniq = sorted(set(s for s in smiles_list if s))
        return {
            "total": len(smiles_list),
            "valid_count": len(uniq),
            "validation_pct": 100.0 * len(uniq) / max(1, len(smiles_list)),
            "unique_count": len(uniq),
            "uniqueness_pct": 100.0,
            "clean_unique_smiles": uniq
        }

    valid_canonical = []
    for s in smiles_list:
        s = (s or "").strip()
        if not s:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)  # canonical
        valid_canonical.append(can)

    total = len(smiles_list)
    valid_count = len(valid_canonical)
    validation_pct = (valid_count / total * 100.0) if total else 0.0

    seen = set()
    clean_unique = []
    for can in valid_canonical:
        if can not in seen:
            seen.add(can)
            clean_unique.append(can)

    unique_count = len(clean_unique)
    uniqueness_pct = (unique_count / valid_count * 100.0) if valid_count else 0.0

    return {
        "total": total,
        "valid_count": valid_count,
        "validation_pct": round(validation_pct, 2),
        "unique_count": unique_count,
        "uniqueness_pct": round(uniqueness_pct, 2),
        "clean_unique_smiles": clean_unique
    }

# =========================
# TRANSFER LEARNING
# =========================
def finetune_from_folder(base_dir: str, new_smiles: List[str], epochs: int = 5, lr: float = 2e-4):
    """
    Load saved vocab+model from base_dir, fine-tune on new SMILES list, and save a new model file.
    """
    vocab = Vocab.load(os.path.join(base_dir, "vocab.json"))
    tagged = clean_and_tag(new_smiles)
    X, Y = make_inputs_targets(tagged, vocab)
    ds = make_tf_dataset(X, Y, batch_size=BATCH_SIZE, shuffle=True)

    model_path = os.path.join(base_dir, "model.keras")
    model = tf.keras.models.load_model(model_path, compile=False)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    print(f"[FT] Fine-tuning on {len(new_smiles)} SMILES...")
    model.fit(ds, epochs=epochs, verbose=1)
    out_path = os.path.join(base_dir, "model_finetuned.keras")
    model.save(out_path)
    print("[FT] Fine-tuned model saved to", out_path)

# =========================
# MAIN
# =========================
def main_train_and_generate():
    set_seed()

    # 1) Load data
    base_smiles = load_smiles_lines(DATA_PATH)
    tagged = clean_and_tag(base_smiles)

    # 2) Vocab & encode
    vocab = build_vocab(tagged)
    X, Y = make_inputs_targets(tagged, vocab)
    print(f"[DATA] Vocab size = {vocab.vocab_size}, max_len = {vocab.max_len}, samples = {len(X)}")

    # 3) Split train/val
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int((1.0 - VAL_SPLIT) * n)
    tr_idx, va_idx = idx[:split], idx[split:]
    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_va, Y_va = X[va_idx], Y[va_idx]
    train_ds = make_tf_dataset(X_tr, Y_tr, batch_size=BATCH_SIZE, shuffle=True)
    val_ds   = make_tf_dataset(X_va, Y_va, batch_size=BATCH_SIZE, shuffle=False)

    # 4) Tiny architecture search
    best_cfg, logs = simple_arch_search(X, Y, vocab)
    print("[SEARCH LOGS]", logs)

    # 5) Build best model & train (with EarlyStopping + CSVLogger)
    embed_dim, lstm1, lstm2, dropout = best_cfg
    model = build_model(vocab.vocab_size, vocab.max_len, embed_dim, lstm1, lstm2, dropout)
    ckpt_path = os.path.join(SAVE_DIR, "model.keras")
    history = train_model(model, train_ds, val_ds, max_epochs=MAX_EPOCHS, ckpt_path=ckpt_path)

    # 6) Save vocab + config
    vocab_path = os.path.join(SAVE_DIR, "vocab.json")
    vocab.save(vocab_path)
    with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "embed_dim": embed_dim, "lstm1": lstm1, "lstm2": lstm2, "dropout": dropout,
            "seed": SEED, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE
        }, f, indent=2)
    print("[SAVE] Model + vocab saved to:", SAVE_DIR)

    # 7) Evaluate (loss/acc) and save metrics
    train_loss = float(history.history["loss"][-1])
    train_acc  = float(history.history.get("sparse_categorical_accuracy", [None])[-1])
    eval_res   = evaluate_model(model, val_ds)
    val_loss, val_acc = eval_res["val_loss"], eval_res["val_acc"]

    print("\n[METRICS]")
    print(f"  train_loss: {train_loss:.4f}")
    if train_acc is not None:
        print(f"  train_acc : {train_acc:.4f}")
    print(f"  val_loss  : {val_loss:.4f}")
    print(f"  val_acc   : {val_acc:.4f}")

    with open(os.path.join(SAVE_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }, f, indent=2)

    # 8) Generate & write clean unique SMILES
    print("[GEN] Sampling...")
    samples = generate_many(model, vocab, n=GEN_NUM_SAMPLES,
                            max_len=GEN_MAX_LEN, temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P)

    gen_info = smiles_stats(samples)
    cleaned_smi = gen_info["clean_unique_smiles"]

    print("\n[GEN STATS]")
    print(f"  Generated: {len(samples)}")
    print(f"  validity  : {gen_info['validation_pct']}%  ({gen_info['valid_count']}/{gen_info['total']})")
    print(f"  uniqueness: {gen_info['uniqueness_pct']}%  ({len(cleaned_smi)}/{gen_info['valid_count']})")

    out_path = os.path.join(SAVE_DIR, f"Clean_Generation_{generation}.smi")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in cleaned_smi:
            f.write(s + "\n")
    print("[GEN] Clean unique SMILES written to:", out_path)
    # --- LOSS CURVE ---
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train_loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="val_loss", linewidth=2)
    plt.title("Model Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi=150)
    plt.show()

    # --- ACCURACY CURVE ---
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["sparse_categorical_accuracy"], label="train_acc", linewidth=2)
    plt.plot(history.history["val_sparse_categorical_accuracy"], label="val_acc", linewidth=2)
    plt.title("Model Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "accuracy_curve.png"), dpi=150)
    plt.show()
    #########################################################################
    #RETRAIN
    ########################################################################
def retrain_model_simple(
    base_dir: str,
    smiles_list: list,
    output_dir: str,
    epochs: int = 5,
    lr: float = 2e-4,
    batch_size: int = 128,
    ):
    """
    Load model + vocab from base_dir, retrain on smiles_list, and save to output_dir.
    No tracking or saving of train/val accuracy or loss.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- 1) Load vocab and model
    vocab_path = os.path.join(base_dir, "vocab.json")
    model_path = os.path.join(base_dir, MODEL_FILENAME)

    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"vocab.json not found in {base_dir}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"{MODEL_FILENAME} not found in {base_dir}")

    vocab = Vocab.load(vocab_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    # --- 2) Prepare data
    tagged = clean_and_tag(smiles_list)
    X, Y = make_inputs_targets(tagged, vocab)
    ds = make_tf_dataset(X, Y, batch_size=batch_size, shuffle=True)

    # --- 3) Compile and retrain
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss)
    print(f"[FT] Retraining on {len(smiles_list)} SMILES for {epochs} epochs...")
    model.fit(ds, epochs=epochs, verbose=1)

    # --- 4) Save fine-tuned model
    model_out = os.path.join(output_dir, "model.keras")
    vocab.save(os.path.join(output_dir, "vocab.json"))
    model.save(model_out)
    print(f"[FT] Fine-tuned model saved to: {model_out}")

    return model_out

def generate_smiles_from_folder(
    model_dir: str,
    num_samples: int = 100,
    max_len: int = 128,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> list:
    """
    Load model+vocab from `model_dir`, generate `num_samples` SMILES, and
    return a list of valid, canonical, unique SMILES (filtered).
    Requires the following in your codebase:
      - class Vocab with .load(path) and fields char_to_idx/idx_to_char/max_len
      - generate_one(model, vocab, max_len, temperature, top_p)  # uses your LSTM
    """
    vocab_path = os.path.join(model_dir, "vocab.json")
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"vocab.json not found in {model_dir}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"{MODEL_FILENAME} not found in {model_dir}")

    # 1) load vocab + model
    vocab = Vocab.load(vocab_path)
    # prefer the saved max_len from vocab if caller didn't override
    max_len = max_len or vocab.max_len

    model = tf.keras.models.load_model(model_path, compile=False)

    # 2) generate raw strings
    raw = []
    for i in tqdm(range(1, num_samples + 1)):
        s = generate_one(model, vocab, max_len=max_len, temperature=temperature, top_p=top_p)
        raw.append(s)

    # 3) validity + uniqueness filtering
    # if RDKit_OK:
    #     canonical = []
    #     seen = set()
    #     for s in raw:
    #         s = (s or "").strip()
    #         if not s:
    #             continue
    #         mol = Chem.MolFromSmiles(s)
    #         if mol is None:
    #             continue
    #         can = Chem.MolToSmiles(mol)  # canonical SMILES
    #         if can not in seen:
    #             seen.add(can)
    #             canonical.append(can)
    #     return canonical
    else:
        # fallback: just unique, non-empty strings
        out = []
        seen = set()
        for s in raw:
            s = (s or "").strip()
            out.append(s)
            # if s and s not in seen:
            #     seen.add(s)
            #     out.append(s)
        return out


if __name__ == "__main__":
    # 1) Regular training + initial generation
    Train = False
    generation = "1"
    from FilterSmiles import GetMetrics

    if Train:
        main_train_and_generate()
    else:
        # 2) generate smiles from a model
        T= 1.2
        # for T in [0.8,0.9,1.0,1.1,1.2]:
        out = generate_smiles_from_folder(
            model_dir=f"Results/Models/smiles_lstm_bn_ckpt_{generation}",
            num_samples=1200,
            max_len=128,
            temperature=T,
            top_p=0.9,
        )
        print(f"Metrics for {T} are : {GetMetrics(out)}")

            # print(f"Returned {len(out)} valid & unique SMILES")
        with open(f"Results/Generation/SMILES_gen_{generation}.smi","w") as outf:
            for s in out:
                outf.write(f"{s}\n")


        # 2) Immediately load the saved model and fine-tune on new inpu
        # new_smiles = out
        # retrain_model_simple(
        #     base_dir="Results/smiles_lstm_bn_ckpt_0",
        #     smiles_list=new_smiles,
        #     output_dir="Results/smiles_lstm_bn_ckpt_1",
        #     epochs=5,
        #     lr=2e-4,
        # )
        #
