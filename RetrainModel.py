"""
Auto pipeline to:
1. Take top 100 SMILES by score from Excel
2. Generate 35 new ones for diversity (keep ~20)
3. Retrain gen-0 model on the combined dataset
4. Save fine-tuned model as gen-1
"""

import os
import random
from pathlib import Path
import pandas as pd
from LSTMSMILES import generate_smiles_from_folder, smiles_stats, retrain_model_simple
from FilterSmiles import FilterSmilesLip


# --- Hardcoded parameters ---
GENERATION = "0"
NEXTGENERATION = "1"
EXCEL_PATH = f"Results/Curated/scores_output_gen_{GENERATION}.xlsx"
SMILES_COL = "SMILES"
SCORE_COL = "Scores"
BASE_MODEL_DIR = f"Results/Models/smiles_lstm_bn_ckpt_0"
IN_MODEL_DIR = f"Results/Models/smiles_lstm_bn_ckpt_{GENERATION}"
OUT_MODEL_DIR = f"Results/Models/smiles_lstm_bn_ckpt_{NEXTGENERATION}"

N_TOP = 100
N_GEN = 40
# N_KEEP = 20
EPOCHS = 30
LR = 2e-4


def pick_top_from_excel(excel_path, smiles_col, score_col, n_top):
    df = pd.read_excel(excel_path)
    df_sorted = df.sort_values(score_col, ascending=False).dropna(subset=[smiles_col])
    top_df = df_sorted.head(n_top)[[smiles_col, score_col]].reset_index(drop=True)
    top_smiles = [str(s).strip() for s in top_df[smiles_col].tolist() if str(s).strip()]
    return top_smiles, top_df


def select_diversity_additions(base_model_dir, already, n_generate):
    """Generate SMILES with the base model and keep unique/valid ones not already in top set."""
    gen_raw = generate_smiles_from_folder(
        model_dir=base_model_dir,
        num_samples=n_generate,
        max_len=128,
        temperature=1.2,
        top_p=0.9
    )
    # stats = smiles_stats(gen_raw)
    candidates = FilterSmilesLip(gen_raw)
    # candidates = [s for s in stats.get("clean_unique_smiles", []) if s and (s not in already)]
    # random.shuffle(candidates)
    return candidates


def main():
    os.makedirs(OUT_MODEL_DIR, exist_ok=True)

    # 1. Select top 100
    top_smiles, top_df = pick_top_from_excel(EXCEL_PATH, SMILES_COL, SCORE_COL, N_TOP)
    print(f"[TOP] Selected {len(top_smiles)} SMILES from Excel (top {N_TOP}).")

    # 2. Generate diversity
    already = set(top_smiles)
    add_smiles = select_diversity_additions(BASE_MODEL_DIR, already, N_GEN)
    print(f"[DIV] Generated {N_GEN}, kept {len(add_smiles)} after filtering and dedup.")

    # 3. Combine
    combined = list(top_smiles) + list(add_smiles)
    print(f"[DATA] Combined training set size = {len(combined)}")

    # 4. Save intermediate files
    pd.Series(top_smiles).to_csv(Path(OUT_MODEL_DIR, "top100.smi"), index=False, header=False)
    pd.Series(add_smiles).to_csv(Path(OUT_MODEL_DIR, "diversity.smi"), index=False, header=False)
    pd.Series(combined).to_csv(Path(OUT_MODEL_DIR, "finetune_set.smi"), index=False, header=False)
    top_df.to_excel(Path(OUT_MODEL_DIR, "top100_with_scores.xlsx"), index=False)

    # 5. Retrain
    retrain_model_simple(
        base_dir=IN_MODEL_DIR,
        smiles_list=combined,
        output_dir=OUT_MODEL_DIR,
        epochs=EPOCHS,
        lr=LR,
        batch_size=128,
    )

    print(f"[DONE] Model retrained and saved to {OUT_MODEL_DIR}")


if __name__ == "__main__":
    main()
