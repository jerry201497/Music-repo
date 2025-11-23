import os, json
from pathlib import Path
from dotenv import load_dotenv
import boto3
import pandas as pd
from PIL import Image
import imagehash
from botocore.exceptions import ClientError

def s3():
    load_dotenv()
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD"),
    )

def ensure_bucket(name):
    c = s3()
    try:
        c.head_bucket(Bucket=name)
    except ClientError:
        c.create_bucket(Bucket=name)

def download(bucket, key, local):
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    s3().download_file(bucket, key, str(local))

def upload(bucket, local, key, ct=None):
    extra = {"ContentType": ct} if ct else {}
    s3().upload_file(str(local), bucket, key, ExtraArgs=extra)

def existing_cols(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    load_dotenv()
    b_in = os.getenv("S3_BUCKET_FORMATTED", "formatted_zone")
    b_out = os.getenv("S3_BUCKET_TRUSTED", "trusted_zone")

    # NEW: make sure trusted bucket exists
    ensure_bucket(b_out)

    # -------- Load refined formatted data --------
    download(b_in, "music/tracks_refined.parquet",
             "cache/formatted/tracks_refined.parquet")
    df = pd.read_parquet("cache/formatted/tracks_refined.parquet")

    before = len(df)

    # -------- Robust column mapping --------
    col_song   = existing_cols(df, ["song_name", "track_name"])
    col_artist = existing_cols(df, ["artist_name", "artist"])
    col_album  = existing_cols(df, ["album", "album_name"])

    if not col_song or not col_artist:
        raise ValueError(
            f"Trusted step cannot find song/artist columns. "
            f"Columns present: {list(df.columns)}"
        )

    dedupe_cols = [col_song, col_artist]
    if col_album:
        dedupe_cols.append(col_album)

    # -------- QC rules --------
    df[col_song] = df[col_song].fillna("").astype(str)
    df[col_artist] = df[col_artist].fillna("").astype(str)
    if col_album:
        df[col_album] = df[col_album].fillna("").astype(str)

    df = df.drop_duplicates(subset=dedupe_cols)
    df = df[(df[col_song] != "") & (df[col_artist] != "")]

    after = len(df)

    # -------- Save trusted table --------
    Path("cache/trusted").mkdir(parents=True, exist_ok=True)
    trusted_path = Path("cache/trusted/tracks_qc.parquet")
    df.to_parquet(trusted_path, index=False)
    upload(b_out, trusted_path, "music/tracks_qc.parquet")

    # -------- QC for covers --------
    res = s3().list_objects_v2(Bucket=b_in, Prefix="music/covers/")
    kept, dropped = 0, 0
    seen = set()

    for obj in (res.get("Contents") or []):
        key = obj["Key"]
        if not key.lower().endswith(".png"):
            continue

        local = Path("cache/formatted/covers") / Path(key).name
        download(b_in, key, local)

        h = str(imagehash.average_hash(Image.open(local)))
        if h in seen:
            dropped += 1
            continue

        seen.add(h)
        kept += 1
        upload(b_out, local, f"music/covers_qc/{Path(local).name}", ct="image/png")

    # -------- Report --------
    report = {
        "tracks_before": before,
        "tracks_after": after,
        "tracks_dropped_duplicates": before - after,
        "covers_kept": kept,
        "covers_dropped": dropped,
        "dedupe_columns_used": dedupe_cols
    }

    Path("reports").mkdir(exist_ok=True)
    report_path = Path("reports/music_quality_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    upload(b_out, report_path, "music/reports/quality_report.json",
           ct="application/json")

    print("=== Trusted summary ===")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
