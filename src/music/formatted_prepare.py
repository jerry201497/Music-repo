import os, json, re, ast
from pathlib import Path
from dotenv import load_dotenv
import boto3
import pandas as pd
from PIL import Image

# -----------------------------
# S3 / MinIO helpers
# -----------------------------
def s3():
    load_dotenv()
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD"),
        region_name="us-east-1",
    )

def download(bucket, key, local):
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    s3().download_file(bucket, key, str(local))

def upload(bucket, local, key, content_type=None):
    extra = {"ContentType": content_type} if content_type else {}
    s3().upload_file(str(local), bucket, key, ExtraArgs=extra)

def ensure_bucket(name):
    c = s3()
    try:
        c.head_bucket(Bucket=name)
    except Exception:
        c.create_bucket(Bucket=name)

# -----------------------------
# Text normalization
# -----------------------------
def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def parse_genres(val):
    """
    spotify_genres from landing may be:
    - list
    - stringified list: "['pop','rock']"
    - comma separated string: "pop, rock"
    - None / NaN
    Returns first genre if available.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None

    if isinstance(val, list):
        return val[0] if val else None

    if isinstance(val, str):
        v = val.strip()
        if not v:
            return None
        # try list literal
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, list) and parsed:
                return parsed[0]
        except Exception:
            pass
        # fallback comma split
        parts = [p.strip() for p in v.split(",") if p.strip()]
        return parts[0] if parts else None

    return None

# -----------------------------
# Refine Spotify fields
# -----------------------------
def refine_spotify(df: pd.DataFrame) -> pd.DataFrame:
    refined_rows = []

    for _, r in df.iterrows():
        track_name  = normalize_text(r.get("track_name", ""))
        artist_name = normalize_text(r.get("artist_name", ""))
        album_name  = normalize_text(r.get("album_name", ""))

        # Prefer Spotify cover from landing (already extracted)
        cover_url = r.get("spotify_cover_url", None)
        if pd.isna(cover_url):
            cover_url = None

        # Duration (ms -> sec)
        dur_ms = r.get("spotify_duration_ms", None)
        if dur_ms is None or (isinstance(dur_ms, float) and pd.isna(dur_ms)):
            duration_sec = None
        else:
            try:
                duration_sec = round(float(dur_ms) / 1000.0, 2)
            except Exception:
                duration_sec = None

        # Year: prefer landing "year", else spotify release year if present
        year = r.get("year", None)
        if year is None or (isinstance(year, float) and pd.isna(year)):
            # try spotify_release_date if exists
            rd = r.get("spotify_release_date", None)
            if isinstance(rd, str) and rd:
                try:
                    year = int(rd.split("-")[0])
                except Exception:
                    year = None
            else:
                year = None
        else:
            try:
                year = int(year)
            except Exception:
                year = None

        # Genre unified: prefer spotify_genres, fallback to original genre
        sp_genre = parse_genres(r.get("spotify_genres", None))
        base_genre = normalize_text(r.get("genre", "")) or None
        genre_unified = sp_genre or base_genre

        refined_rows.append({
            # --- columns REQUIRED by Trusted Zone ---
            "track_name": track_name,
            "artist_name": artist_name,
            "album_name": album_name,

            # --- refined / final columns you want ---
            "genre": genre_unified,
            "year": year,
            "duration_sec": duration_sec,
            "cover_url": cover_url,
        })

    return pd.DataFrame(refined_rows)

# -----------------------------
# MAIN FORMATTED PIPELINE
# -----------------------------
def main():
    load_dotenv()
    b_in  = os.getenv("S3_BUCKET_LANDING",  "landing_zone")
    b_out = os.getenv("S3_BUCKET_FORMATTED","formatted_zone")
    ensure_bucket(b_out)

    # 1) Load enriched landing data (CORRECT source)
    download(
        b_in,
        "music/persistent_landing/tracks_enriched.csv",
        "cache/landing/tracks_enriched.csv"
    )
    df = pd.read_csv("cache/landing/tracks_enriched.csv")

    # 2) Refine Spotify + normalize
    df_refined = refine_spotify(df)

    # 3) Save refined table (Parquet + CSV)
    out_dir = Path("cache/formatted")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parq = out_dir / "tracks_refined.parquet"
    out_csv  = out_dir / "tracks_refined.csv"

    df_refined.to_parquet(out_parq, index=False)
    df_refined.to_csv(out_csv, index=False)

    upload(b_out, out_parq, "music/tracks_refined.parquet",
           content_type="application/octet-stream")
    upload(b_out, out_csv, "music/tracks_refined.csv",
           content_type="text/csv")

    # 4) Covers conversion (if any were downloaded in landing)
    res = s3().list_objects_v2(
        Bucket=b_in,
        Prefix="music/persistent_landing/covers/"
    )
    converted = 0
    for obj in (res.get("Contents") or []):
        key = obj["Key"]
        if not key.lower().endswith(".jpg"):
            continue

        local = Path("cache/landing/covers") / Path(key).name
        download(b_in, key, local)

        im = Image.open(local).convert("RGB").resize((256, 256))
        outp = Path("cache/formatted/covers") / Path(local).with_suffix(".png").name
        outp.parent.mkdir(parents=True, exist_ok=True)
        im.save(outp)

        upload(b_out, outp, f"music/covers/{outp.name}",
               content_type="image/png")
        converted += 1

    # 5) Summary
    summary = {
        "formatted_rows": int(len(df_refined)),
        "covers_converted": converted,
        "output_files": [
            "music/tracks_refined.parquet",
            "music/tracks_refined.csv",
        ]
    }

    summ_path = out_dir / "format_summary.json"
    summ_path.write_text(json.dumps(summary, indent=2))

    upload(b_out, summ_path, "music/summary/format_summary.json",
           content_type="application/json")

    print("=== Formatted summary ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
