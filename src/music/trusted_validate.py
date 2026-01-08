import os, json, re
from pathlib import Path
from dotenv import load_dotenv
import boto3
import pandas as pd
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
    try:
        s3().upload_file(str(local), bucket, key, ExtraArgs=extra)
    except Exception as e:
        print(f"[WARN] upload failed for {local} -> s3://{bucket}/{key}: {e}")

def norm_text(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"“”‘’´`]", "", s)
    s = re.sub(r"[^\w\s,&\-]", "", s)
    return s

def load_formatted_local_first():
    p_parq = Path("cache/formatted/tracks_refined.parquet")
    p_csv  = Path("cache/formatted/tracks_refined.csv")

    if p_parq.exists():
        return pd.read_parquet(p_parq), "local_parquet"
    if p_csv.exists():
        return pd.read_csv(p_csv), "local_csv"

    return None, "missing"

def main():
    load_dotenv()
    b_in  = os.getenv("S3_BUCKET_FORMATTED", "formatted_zone")
    b_out = os.getenv("S3_BUCKET_TRUSTED", "trusted_zone")
    ensure_bucket(b_out)

    # --- Load formatted (LOCAL FIRST) ---
    df, src = load_formatted_local_first()
    if df is None:
        # fallback to S3 only if local is missing
        download(b_in, "music/tracks_refined.parquet", "cache/formatted/tracks_refined.parquet")
        df = pd.read_parquet("cache/formatted/tracks_refined.parquet")
        src = "s3_parquet"

    print(f"[TRUSTED] loaded formatted from: {src} (rows={len(df)}, cols={len(df.columns)})")

    # Require correct columns
    if "track_name" not in df.columns or "artist_name" not in df.columns:
        raise ValueError(f"Expected track_name/artist_name not found. Columns: {list(df.columns)}")

    before = len(df)

    # Normalize + filter empty
    df["__track"]  = df["track_name"].apply(norm_text)
    df["__artist"] = df["artist_name"].apply(norm_text)
    df = df[(df["__track"] != "") & (df["__artist"] != "")]

    # Dedupe on track+artist (album optional; don’t kill everything if album missing)
    df = df.drop_duplicates(subset=["__track", "__artist"])

    after = len(df)

    df = df.drop(columns=["__track", "__artist"])

    # Save locally
    Path("cache/trusted").mkdir(parents=True, exist_ok=True)
    outp = Path("cache/trusted/tracks_qc.parquet")
    df.to_parquet(outp, index=False)

    # Best-effort upload
    upload(b_out, outp, "music/tracks_qc.parquet")

    report = {
        "tracks_before": int(before),
        "tracks_after": int(after),
        "tracks_dropped": int(before - after),
        "dedupe_columns_used": ["track_name", "artist_name"],
        "source_loaded": src,
    }

    Path("reports").mkdir(exist_ok=True)
    report_path = Path("reports/music_quality_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    upload(b_out, report_path, "music/reports/quality_report.json", ct="application/json")

    print("=== Trusted summary ===")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
