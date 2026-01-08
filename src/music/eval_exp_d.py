import os, json, random
from pathlib import Path

import numpy as np
import chromadb
from chromadb.utils import embedding_functions as ef
import matplotlib.pyplot as plt


def parse_ks(s: str):
    out = []
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return sorted(set(out))


def main():
    chroma_path = os.getenv("CHROMA_PATH", "chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "music_exp_d_text_audio_v2")
    max_eval = int(os.getenv("MUSIC_EVAL_MAX", "200"))
    ks = parse_ks(os.getenv("MUSIC_EVAL_KS", "1,5,10,20"))
    seed = int(os.getenv("MUSIC_SEED", "42"))

    random.seed(seed)
    np.random.seed(seed)

    client = chromadb.PersistentClient(path=chroma_path)
    embed = ef.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    col = client.get_or_create_collection(collection_name, embedding_function=embed)

    # --- Load all items once ---
    data = col.get(include=["metadatas", "documents"])
    ids = data.get("ids", [])
    metas = data.get("metadatas", [])
    docs = data.get("documents", [])

    if not ids:
        raise RuntimeError(f"[ERROR] Collection '{collection_name}' is empty at {chroma_path}")

    # --- Build lists + a pair_id -> audio_id mapping from what's ACTUALLY in Chroma ---
    text_items = []
    pair_to_audio_id = {}

    n_text = n_audio = 0
    for _id, meta, doc in zip(ids, metas, docs):
        meta = meta or {}
        modality = meta.get("modality")
        pair_id = meta.get("pair_id")

        if modality == "text":
            n_text += 1
            if pair_id:
                text_items.append((_id, pair_id, doc))
        elif modality == "audio":
            n_audio += 1
            if pair_id:
                pair_to_audio_id[pair_id] = _id

    print(f"[DEBUG] items in collection: total={len(ids)} text={n_text} audio={n_audio} pairs_with_audio={len(pair_to_audio_id)}")

    if n_audio == 0:
        raise RuntimeError("[ERROR] No audio items found (modality='audio'). Exploitation indexing didn't store audio modality.")

    if not text_items:
        raise RuntimeError("[ERROR] No text items found (modality='text').")

    # sample
    if len(text_items) > max_eval:
        text_items = random.sample(text_items, max_eval)

    # --- Evaluate Text -> Audio retrieval ---
    ranks = []
    hits_at_k = {k: 0 for k in ks}
    skipped_no_pair = 0
    skipped_missing_audio = 0

    n_results = min(500, n_audio) # limit to reasonable number

    for _text_id, pair_id, doc in text_items:
        if not pair_id:
            skipped_no_pair += 1
            continue

        expected_audio_id = pair_to_audio_id.get(pair_id)
        if not expected_audio_id:
            skipped_missing_audio += 1
            continue

        # ✅ KEY FIX: query ONLY audio candidates
        res = col.query(
            query_texts=[doc],
            n_results=n_results,
            where={"modality": "audio"},
            include=["metadatas", "documents", "distances"],  # no "ids" here
        )

        cand_ids = (res.get("ids") or [[]])[0]

        try:
            r = cand_ids.index(expected_audio_id) + 1
        except ValueError:
            r = None

        if r is not None:
            ranks.append(r)
            for k in ks:
                if r <= k:
                    hits_at_k[k] += 1

    if not ranks:
        # extra debug help
        some_pairs = list(pair_to_audio_id.items())[:5]
        print("[DEBUG] sample pair_to_audio_id:", some_pairs)
        raise RuntimeError(
            "[ERROR] No valid ranks computed.\n"
            "Most likely: your query isn't returning audio items, OR pair_id mismatch.\n"
            "This eval already filters modality='audio' — so next check exploitation_index.py stores pair_id consistently."
        )

    ranks_arr = np.array(ranks)

    recall_at_k = {k: hits_at_k[k] / len(ranks) for k in ks}
    mrr = float(np.mean(1.0 / ranks_arr))
    med_rank = float(np.median(ranks_arr))
    mean_rank = float(np.mean(ranks_arr))

    summary = {
        "experiment": "D",
        "collection": collection_name,
        "chroma_path": chroma_path,
        "queries_sampled": int(len(text_items)),
        "queries_scored": int(len(ranks)),
        "skipped_no_pair": int(skipped_no_pair),
        "skipped_missing_audio": int(skipped_missing_audio),
        "ks": ks,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "median_rank": med_rank,
        "mean_rank": mean_rank,
        "rank_min": int(np.min(ranks_arr)),
        "rank_max": int(np.max(ranks_arr)),
    }

    Path("reports").mkdir(exist_ok=True)
    out_json = Path("reports/exp_d_eval_metrics.json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure()
    plt.plot(list(recall_at_k.keys()), list(recall_at_k.values()), marker="o")
    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title("Experiment D: Text → Audio (semantic bins)")
    plt.grid(True, alpha=0.3)
    plt.savefig("reports/exp_d_recall_at_k.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(ranks_arr, bins=min(30, max(10, int(np.max(ranks_arr)))))
    plt.xlabel("Rank of correct audio pair")
    plt.ylabel("Count")
    plt.title("Experiment D: Rank distribution (Text → Audio)")
    plt.grid(True, alpha=0.3)
    plt.savefig("reports/exp_d_rank_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("=== Experiment D evaluation summary ===")
    print(json.dumps(summary, indent=2))
    print("Saved:")
    print(f" - {out_json}")
    print(" - reports/exp_d_recall_at_k.png")
    print(" - reports/exp_d_rank_hist.png")


if __name__ == "__main__":
    main()
