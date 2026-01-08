import os, json, re
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import boto3
import pandas as pd
from PIL import Image
from botocore.exceptions import ClientError


# -----------------------------
# S3 helpers
# -----------------------------
def s3():
    load_dotenv()
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD"),
        region_name=os.getenv("MINIO_REGION", "us-east-1"),
    )


def ensure_bucket(name: str):
    cli = s3()
    try:
        cli.head_bucket(Bucket=name)
    except Exception:
        try:
            cli.create_bucket(Bucket=name)
        except Exception:
            pass


def download(bucket: str, key: str, local: Path) -> bool:
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3().download_file(bucket, key, str(local))
        return True
    except Exception as e:
        print(f"[WARN] download failed: s3://{bucket}/{key} -> {local}: {e}")
        return False


def upload(bucket: str, local: Path, key: str, content_type: str | None = None) -> bool:
    extra = {"ContentType": content_type} if content_type else {}
    try:
        s3().upload_file(str(local), bucket, key, ExtraArgs=extra)
        return True
    except ClientError as e:
        print(f"[WARN] upload failed for {local} -> s3://{bucket}/{key}: {e}")
        return False
    except Exception as e:
        print(f"[WARN] upload failed for {local} -> s3://{bucket}/{key}: {e}")
        return False


# -----------------------------
# Formatting helpers
# -----------------------------
def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _safe_year(release_date):
    """
    Spotify release_date can be 'YYYY', 'YYYY-MM', 'YYYY-MM-DD'
    """
    if not isinstance(release_date, str) or not release_date:
        return None
    try:
        return int(release_date.split("-")[0])
    except Exception:
        return None


def refine_spotify(df: pd.DataFrame) -> pd.DataFrame:
    refined_rows = []

    for _, r in df.iterrows():
        album_name = r.get("spotify_album_name")
        release_date = r.get("spotify_release_date")
        year = _safe_year(release_date)

        dur_ms = r.get("spotify_duration_ms")
        dur_sec = round(float(dur_ms) / 1000.0, 2) if pd.notna(dur_ms) else None

        # genre: prefer original genre, else spotify_genres
        genre = r.get("genre", None)
        if genre is None or (isinstance(genre, float) and pd.isna(genre)) or str(genre).strip() == "":
            genre = r.get("spotify_genres", None)

        refined_rows.append({
            "track_name": r.get("track_name", ""),
            "artist_name": r.get("artist_name", ""),
            "album_name": album_name,
            "genre": genre,
            "year": year,
            "duration_sec": dur_sec,
            "cover_url": r.get("spotify_cover_url"),

            # audio features (may be null)
            "sp_danceability": r.get("sp_danceability"),
            "sp_energy": r.get("sp_energy"),
            "sp_valence": r.get("sp_valence"),
            "sp_speechiness": r.get("sp_speechiness"),
            "sp_acousticness": r.get("sp_acousticness"),
            "sp_instrumentalness": r.get("sp_instrumentalness"),
            "sp_liveness": r.get("sp_liveness"),
            "sp_loudness": r.get("sp_loudness"),
            "sp_tempo": r.get("sp_tempo"),
        })

    return pd.DataFrame(refined_rows)


def load_manifest() -> dict:
    """
    Manifest is written by landing_ingest.py at:
      cache/landing/manifest.json
    """
    p = Path("cache/landing/manifest.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def covers_to_convert_from_manifest(manifest: dict) -> list[str]:
    """
    New landing_ingest.py writes:
      "covers_in_run": ["<albumid>.jpg", ...]
    Return that list, otherwise empty.
    """
    covers = manifest.get("covers_in_run")
    if isinstance(covers, list):
        return [c for c in covers if isinstance(c, str) and c.lower().endswith(".jpg")]
    return []


def norm_key(x):
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().lower().split())

def load_kaggle_audio(kaggle_path: str) -> pd.DataFrame:
    k = pd.read_csv(kaggle_path)

    # keep only what we need
    keep = ["artist_name", "track_name", "danceability", "loudness",
            "acousticness", "instrumentalness", "valence", "energy"]
    k = k[[c for c in keep if c in k.columns]].copy()

    k["k_track"] = k["track_name"].map(norm_key)
    k["k_artist"] = k["artist_name"].map(norm_key)

    # dedupe to avoid exploding merges
    k = k.drop_duplicates(subset=["k_track", "k_artist"])

    # rename to avoid collisions
    return k.rename(columns={
        "danceability": "k_danceability",
        "loudness": "k_loudness",
        "acousticness": "k_acousticness",
        "instrumentalness": "k_instrumentalness",
        "valence": "k_valence",
        "energy": "k_energy",
    })

# -----------------------------
# MAIN
# -----------------------------
def main():
    load_dotenv()
    b_in = os.getenv("S3_BUCKET_LANDING", "landing")
    b_out = os.getenv("S3_BUCKET_FORMATTED", "formatted")
    ensure_bucket(b_out)

    # 1) Load enriched landing data (prefer local if it already exists)
    local_enriched = Path("cache/landing/tracks_enriched.csv")
    if not local_enriched.exists():
        ok = download(b_in, "music/persistent_landing/tracks_enriched.csv", local_enriched)
        if not ok:
            raise FileNotFoundError("tracks_enriched.csv not found locally and cannot be downloaded from S3.")

    df = pd.read_csv(local_enriched)

    # ---- Fill audio features from Kaggle dataset (fast, no Spotify needed) ----
    kaggle_csv = "cache/formatted/tcc_ceds_music.csv"
    if Path(kaggle_csv).exists():
        k = load_kaggle_audio(kaggle_csv)

        df["k_track"] = df["track_name"].map(norm_key)
        df["k_artist"] = df["artist_name"].map(norm_key)

        df = df.merge(k, how="left", left_on=["k_track", "k_artist"], right_on=["k_track", "k_artist"])

        # If Spotify sp_* is missing, fill from Kaggle
        def fill(col_sp, col_k):
            if col_sp not in df.columns:
                df[col_sp] = np.nan
            df[col_sp] = df[col_sp].where(df[col_sp].notna(), df[col_k])

        fill("sp_danceability", "k_danceability")
        fill("sp_loudness", "k_loudness")
        fill("sp_acousticness", "k_acousticness")
        fill("sp_instrumentalness", "k_instrumentalness")
        fill("sp_valence", "k_valence")
        fill("sp_energy", "k_energy")

        # cleanup helper cols
        df = df.drop(columns=[c for c in ["k_track","k_artist","k_danceability","k_loudness","k_acousticness",
                                        "k_instrumentalness","k_valence","k_energy"] if c in df.columns])
    else:
        print(f"[WARN] Kaggle file not found at {kaggle_csv}, skipping audio fill.")


    # 2) Refine
    df_refined = refine_spotify(df)

    out_dir = Path("cache/formatted")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = out_dir / "tracks_refined.parquet"
    out_csv = out_dir / "tracks_refined.csv"

    df_refined.to_parquet(out_parquet, index=False)
    df_refined.to_csv(out_csv, index=False)

    upload(b_out, out_parquet, "music/tracks_refined.parquet", content_type="application/octet-stream")
    upload(b_out, out_csv, "music/tracks_refined.csv", content_type="text/csv")

    # 3) Covers: ONLY those from this run (manifest-driven)
    manifest = load_manifest()
    wanted_covers = covers_to_convert_from_manifest(manifest)

    converted = 0
    kept = 0

    landing_covers_dir = Path("cache/landing/album_covers")
    formatted_covers_dir = Path("cache/formatted/covers")
    formatted_covers_dir.mkdir(parents=True, exist_ok=True)

    for fname in wanted_covers:
        # Prefer local cover first
        local_jpg = landing_covers_dir / fname
        if not local_jpg.exists():
            # fallback: try to download from S3 (non-fatal)
            download(b_in, f"music/persistent_landing/covers/{fname}", local_jpg)

        if not local_jpg.exists():
            continue  # couldn't get it

        try:
            im = Image.open(local_jpg).convert("RGB").resize((256, 256))
            out_png = formatted_covers_dir / (Path(fname).with_suffix(".png").name)
            im.save(out_png)
            converted += 1
            kept += 1

            upload(b_out, out_png, f"music/covers/{out_png.name}", content_type="image/png")

        except Exception as e:
            print(f"[WARN] cover convert failed for {local_jpg}: {e}")

    summary = {
        "formatted_rows": int(len(df_refined)),
        "covers_requested_from_manifest": int(len(wanted_covers)),
        "covers_converted": int(converted),
        "output_files": [
            "music/tracks_refined.parquet",
            "music/tracks_refined.csv",
        ],
    }

    (out_dir / "format_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    upload(b_out, out_dir / "format_summary.json", "music/summary/format_summary.json", content_type="application/json")

    print("=== Formatted summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
