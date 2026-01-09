# src/music/audio_similarity.py
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


AUDIO_COLS_DEFAULT = [
    "sp_danceability",
    "sp_energy",
    "sp_valence",
    "sp_acousticness",
    "sp_instrumentalness",
    "sp_loudness",
    "sp_tempo",
]


def _norm_genre(x) -> str:
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().lower().split())


def sample_pairs_by_genre(genres: np.ndarray, max_pairs: int, seed: int):
    rng = random.Random(seed)

    by_g = {}
    for i, g in enumerate(genres):
        if g:
            by_g.setdefault(g, []).append(i)

    unique_g = list(by_g.keys())
    if len(unique_g) < 2:
        return [], []

    same_pairs = []
    for g in unique_g:
        idxs = by_g[g]
        if len(idxs) < 2:
            continue
        for _ in range(min(max_pairs, len(idxs) * 2)):
            i, j = rng.sample(idxs, 2)
            if i > j:
                i, j = j, i
            same_pairs.append((i, j))

    diff_pairs = []
    for _ in range(max_pairs * 2):
        g1, g2 = rng.sample(unique_g, 2)
        i = rng.choice(by_g[g1])
        j = rng.choice(by_g[g2])
        if i > j:
            i, j = j, i
        diff_pairs.append((i, j))

    same_pairs = list(dict.fromkeys(same_pairs))[:max_pairs]
    diff_pairs = list(dict.fromkeys(diff_pairs))[:max_pairs]
    return same_pairs, diff_pairs


def main():
    in_path = os.getenv("MUSIC_AUDIO_SIM_INPUT", "cache/formatted/tracks_refined.csv")

    max_rows = int(os.getenv("MUSIC_AUDIO_SIM_MAX", "400"))
    max_pairs = int(os.getenv("MUSIC_AUDIO_SIM_PAIRS", "20000"))
    seed = int(os.getenv("MUSIC_SEED", "42"))

    # Robustness knobs
    min_features = int(os.getenv("MUSIC_AUDIO_SIM_MIN_FEATURES", "4"))
    min_per_genre = int(os.getenv("MUSIC_AUDIO_SIM_MIN_PER_GENRE", "8"))
    min_rows_needed = int(os.getenv("MUSIC_AUDIO_SIM_MIN_ROWS", "30"))
    impute = os.getenv("MUSIC_AUDIO_SIM_IMPUTE", "1") == "1"

    # NEW: drop columns with too much missingness
    max_missing_rate = float(os.getenv("MUSIC_AUDIO_SIM_MAX_MISSING_RATE", "0.60"))
    # e.g. 0.60 means we drop columns with >60% NaNs in the filtered set

    audio_cols_req = os.getenv("MUSIC_AUDIO_COLS", ",".join(AUDIO_COLS_DEFAULT)).split(",")
    audio_cols_req = [c.strip() for c in audio_cols_req if c.strip()]

    p = Path(in_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing input file: {in_path}. Run formatted_prepare.py first (or set MUSIC_AUDIO_SIM_INPUT)."
        )

    df = pd.read_csv(p)

    # Determine available audio cols
    audio_cols = [c for c in audio_cols_req if c in df.columns]
    if len(audio_cols) < 1:
        raise RuntimeError(f"No requested audio columns found in {in_path}.")

    if "genre" not in df.columns:
        raise ValueError("Missing 'genre' column in input. Cannot compare same-vs-different genre.")

    # Clean genre
    df["genre_norm"] = df["genre"].map(_norm_genre)

    # Convert audio cols to numeric
    for c in audio_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    Path("reports").mkdir(exist_ok=True)

    # Filter: keep rows with a genre
    before_total = len(df)
    df = df[df["genre_norm"] != ""].copy()
    after_genre = len(df)

    # --- Drop audio columns that are all-NaN or too-missing in THIS filtered dataset ---
    missing_rate = {c: float(df[c].isna().mean()) for c in audio_cols}
    audio_cols_kept = [
        c for c in audio_cols
        if missing_rate[c] < 1.0 and missing_rate[c] <= max_missing_rate
    ]
    dropped_cols = [c for c in audio_cols if c not in audio_cols_kept]
    audio_cols = audio_cols_kept

    # Save missingness/debug report
    miss_report = {
        "input": str(p),
        "rows_total": int(before_total),
        "rows_after_genre_filter": int(after_genre),
        "missing_rate_by_audio_col_after_genre_filter": missing_rate,
        "max_missing_rate_threshold": max_missing_rate,
        "audio_cols_kept": audio_cols,
        "audio_cols_dropped": dropped_cols,
        "impute_enabled": bool(impute),
    }
    Path("reports/audio_similarity_missingness.json").write_text(
        json.dumps(miss_report, indent=2), encoding="utf-8"
    )

    if len(audio_cols) < min_features:
        raise RuntimeError(
            "Not enough usable audio columns after dropping too-missing columns.\n"
            f"Kept ({len(audio_cols)}): {audio_cols}\n"
            f"Dropped ({len(dropped_cols)}): {dropped_cols}\n"
            "Tip: increase MUSIC_AUDIO_SIM_MAX_MISSING_RATE (e.g. 0.8) or lower MUSIC_AUDIO_SIM_MIN_FEATURES."
        )

    # Optional: median impute remaining missing values
    if impute:
        for c in audio_cols:
            med = df[c].median(skipna=True)
            if pd.notna(med):
                df[c] = df[c].fillna(med)

    # Require at least min_features non-null per row (final guard)
    nonnull_count = df[audio_cols].notna().sum(axis=1)
    df = df[nonnull_count >= min_features].copy()

    # Final guard: drop any remaining NaNs in kept audio cols
    df = df.dropna(subset=audio_cols).copy()

    # Filter to genres with enough tracks
    vc = df["genre_norm"].value_counts()
    keep_genres = vc[vc >= min_per_genre].index.tolist()
    df = df[df["genre_norm"].isin(keep_genres)].copy()

    if len(df) < min_rows_needed or df["genre_norm"].nunique() < 2:
        raise RuntimeError(
            "Not enough usable rows/genre diversity AFTER filtering.\n"
            f"Usable rows: {len(df)}\n"
            f"Unique genres: {df['genre_norm'].nunique()}\n"
            f"Try: set MUSIC_AUDIO_SIM_MIN_PER_GENRE=3 and MUSIC_AUDIO_SIM_MIN_ROWS=15.\n"
            "Also check reports/audio_similarity_missingness.json."
        )

    # Sample rows for speed
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    X = df[audio_cols].astype(float).values
    X = StandardScaler().fit_transform(X)

    # Cosine similarity matrix (should be NaN-free now)
    S = cosine_similarity(X)

    genres = df["genre_norm"].values
    same_pairs, diff_pairs = sample_pairs_by_genre(genres, max_pairs=max_pairs, seed=seed)
    if not same_pairs or not diff_pairs:
        raise RuntimeError("Could not sample enough same/different genre pairs. Check genre diversity.")

    same_sims = np.array([S[i, j] for (i, j) in same_pairs], dtype=float)
    diff_sims = np.array([S[i, j] for (i, j) in diff_pairs], dtype=float)

    summary = {
        "input": str(p),
        "rows_used": int(len(df)),
        "unique_genres_used": int(df["genre_norm"].nunique()),
        "top_genres_used": df["genre_norm"].value_counts().head(10).to_dict(),
        "audio_cols_used": audio_cols,
        "audio_cols_dropped": dropped_cols,
        "min_features_per_row": int(min_features),
        "min_per_genre": int(min_per_genre),
        "impute_missing_audio": bool(impute),
        "max_missing_rate_threshold": float(max_missing_rate),
        "pairs_same_genre": int(len(same_pairs)),
        "pairs_diff_genre": int(len(diff_pairs)),
        "mean_sim_same_genre": float(np.mean(same_sims)),
        "mean_sim_diff_genre": float(np.mean(diff_sims)),
        "median_sim_same_genre": float(np.median(same_sims)),
        "median_sim_diff_genre": float(np.median(diff_sims)),
        "delta_mean_same_minus_diff": float(np.mean(same_sims) - np.mean(diff_sims)),
        "seed": int(seed),
        "note": (
            "Same-modality similarity check using standardized audio feature vectors "
            "(cosine similarity). Hypothesis: same-genre pairs have higher similarity."
        ),
        "debug_missingness_file": "reports/audio_similarity_missingness.json",
    }

    out_json = Path("reports/audio_similarity_report.json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure()
    plt.hist(same_sims, bins=30, alpha=0.7, label="Same genre")
    plt.hist(diff_sims, bins=30, alpha=0.7, label="Different genre")
    plt.xlabel("Cosine similarity (audio feature vectors)")
    plt.ylabel("Count")
    plt.title("Audio similarity: same-genre vs different-genre pairs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("reports/audio_similarity_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("=== Audio similarity (same-modality) summary ===")
    print(json.dumps(summary, indent=2))
    print("Saved:")
    print(f" - {out_json}")
    print(" - reports/audio_similarity_hist.png")
    print(" - reports/audio_similarity_missingness.json")


if __name__ == "__main__":
    main()
