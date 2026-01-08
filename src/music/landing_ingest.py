import os
import json
import time
import random
import hashlib
from pathlib import Path

import pandas as pd
import requests
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException


# -----------------------------
# S3 / MinIO helpers
# -----------------------------
def s3_resource():
    load_dotenv()
    return boto3.resource(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD"),
        region_name=os.getenv("MINIO_REGION", "us-east-1"),
    )


def s3_client():
    return s3_resource().meta.client


def ensure_bucket(name: str) -> None:
    cli = s3_client()
    try:
        cli.head_bucket(Bucket=name)
    except ClientError:
        cli.create_bucket(Bucket=name)


def upload_small(local_path: Path, bucket: str, key: str, ct: str | None = None, max_attempts: int = 10) -> bool:
    """
    Upload small files (csv/json) using put_object (NOT upload_file), with backoff.
    This avoids boto3's transfer manager internal retries that can throw S3UploadFailedError.
    """
    body = local_path.read_bytes()
    extra = {"ContentType": ct} if ct else {}

    for attempt in range(1, max_attempts + 1):
        try:
            s3_client().put_object(Bucket=bucket, Key=key, Body=body, **extra)
            return True

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")

            # permissions: don't crash pipeline
            if code in {"AccessDenied", "AllAccessDisabled"}:
                print(f"[WARN] upload denied: {local_path} -> s3://{bucket}/{key} ({code})")
                return False

            # throttling: exponential backoff + jitter
            if code in {"SlowDown", "SlowDownWrite", "Throttling"}:
                sleep_s = min(15.0, (2 ** (attempt - 1)) * 0.5 + random.random())
                print(f"[WARN] S3 throttling ({code}) attempt {attempt}/{max_attempts}, sleeping {sleep_s:.2f}s")
                time.sleep(sleep_s)
                continue

            print(f"[WARN] S3 put_object failed ({code}) for {local_path} -> s3://{bucket}/{key}: {e}")
            return False

        except Exception as e:
            print(f"[WARN] S3 put_object failed for {local_path} -> s3://{bucket}/{key}: {e}")
            return False

    print(f"[WARN] upload_small gave up after {max_attempts}: {local_path} -> s3://{bucket}/{key}")
    return False


# -----------------------------
# Spotify
# -----------------------------
def spotify_client():
    load_dotenv()
    auth = SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    )
    return spotipy.Spotify(
        client_credentials_manager=auth,
        requests_timeout=12,
        retries=2,
    )


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _cache_key(track: str, artist: str) -> str:
    return hashlib.md5(f"{_norm(track)}||{_norm(artist)}".encode("utf-8")).hexdigest()


def search_track(sp: spotipy.Spotify, track_name: str, artist_name: str) -> dict:
    """
    One Spotify search call (most expensive part).
    Returns metadata needed for later batch audio_features.
    """
    q = f"track:{track_name} artist:{artist_name}"
    res = sp.search(q, limit=1, type="track")
    items = res.get("tracks", {}).get("items", [])
    if not items:
        return {}

    t = items[0]
    album = t.get("album", {}) or {}

    # cover (largest)
    cover_url = None
    imgs = album.get("images") or []
    if imgs:
        cover_url = sorted(imgs, key=lambda x: x.get("width", 0), reverse=True)[0].get("url")

    artist_ids = [a.get("id") for a in (t.get("artists") or []) if a.get("id")]

    return {
        "spotify_track_id": t.get("id"),
        "spotify_artist_ids": artist_ids,
        "spotify_album_id": album.get("id"),
        "spotify_album_name": album.get("name"),
        "spotify_release_date": album.get("release_date"),
        "spotify_duration_ms": t.get("duration_ms"),
        "spotify_cover_url": cover_url,
        "spotify_popularity": t.get("popularity"),
    }


def batch_audio_features(sp: spotipy.Spotify, track_ids: list[str]) -> tuple[dict[str, dict], bool]:
    """
    Returns: (features_by_track_id, blocked)
    blocked=True if Spotify returns 403 and we should stop calling it.
    """
    feats: dict[str, dict] = {}
    blocked = False

    # Spotify API allows up to 100 ids per call
    for i in range(0, len(track_ids), 100):
        chunk = [tid for tid in track_ids[i:i + 100] if tid]
        if not chunk:
            continue
        try:
            af_list = sp.audio_features(chunk) or []
        except SpotifyException as e:
            if getattr(e, "http_status", None) == 403:
                print("[SPOTIFY] audio_features blocked (403). Skipping audio features for the rest of the run.")
                blocked = True
                break
            print(f"[SPOTIFY] audio_features error: {e}")
            break
        except Exception as e:
            print(f"[SPOTIFY] audio_features unexpected error: {e}")
            break

        for af in af_list:
            if af and af.get("id"):
                feats[af["id"]] = af

        time.sleep(0.02)

    return feats, blocked


def download_cover(session: requests.Session, url: str | None, out_path: Path) -> bool:
    if not url:
        return False
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            return False
        r = session.get(url, timeout=12)
        if r.status_code == 200:
            out_path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


# -----------------------------
# MAIN
# -----------------------------
def main():
    load_dotenv()

    # ---------- Speed defaults (target: < 4 minutes) ----------
    # Default FAST=1 unless you explicitly set MUSIC_FAST=0
    fast_mode = os.getenv("MUSIC_FAST", "1") == "1"

    # Default sample size tuned for speed.
    # If you want bigger, set MUSIC_SAMPLE_N=300 etc.
    sample_n = int(os.getenv("MUSIC_SAMPLE_N", "120"))

    # Covers are slow and not needed for pipeline testing: default 0 in fast mode
    max_covers = int(os.getenv("MUSIC_MAX_COVERS", "0" if fast_mode else "80"))

    # Audio features often 403s for some apps/environments: default off in fast mode
    try_audio = os.getenv("MUSIC_TRY_AUDIO", "0" if fast_mode else "1") == "1"

    bucket = os.getenv("S3_BUCKET_LANDING", "landing")
    ensure_bucket(bucket)

    csv_path = Path("data/music/raw/tracks.csv")
    if not csv_path.exists():
        raise FileNotFoundError("Missing input: data/music/raw/tracks.csv")

    df = pd.read_csv(csv_path)

    keep = [c for c in ["track_name", "artist_name", "album_name", "genre", "year"] if c in df.columns]
    df = df[keep].dropna(subset=["track_name", "artist_name"]).drop_duplicates().reset_index(drop=True)

    # Hard sample (main speed control)
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=42).reset_index(drop=True)

    # ---------- Cache (so reruns are very fast) ----------
    landing_dir = Path("cache/landing")
    landing_dir.mkdir(parents=True, exist_ok=True)
    cache_path = landing_dir / "spotify_search_cache.json"

    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(cache, dict):
                cache = {}
        except Exception:
            cache = {}
    else:
        cache = {}

    sp = spotify_client()

    # ---------- 1) Spotify search (cached) ----------
    meta_rows = []
    uncached = 0

    # Use itertuples (faster than iterrows)
    for idx, row in tqdm(list(enumerate(df.itertuples(index=False))), total=len(df), desc="Spotify: search"):
        track = str(getattr(row, "track_name"))
        artist = str(getattr(row, "artist_name"))
        k = _cache_key(track, artist)

        if k in cache:
            meta_rows.append(cache[k])
            continue

        uncached += 1
        try:
            m = search_track(sp, track, artist)
            cache[k] = m
            meta_rows.append(m)
        except Exception as e:
            print(f"[SPOTIFY] search error for {track} - {artist}: {e}")
            cache[k] = {}
            meta_rows.append({})

        # tiny jitter every 25 uncached requests (keeps speed but reduces bans)
        if uncached % 25 == 0:
            time.sleep(0.05 + random.random() * 0.05)

    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    meta_df = pd.DataFrame(meta_rows)
    expected_meta_cols = [
        "spotify_track_id", "spotify_artist_ids", "spotify_album_id",
        "spotify_album_name", "spotify_release_date", "spotify_duration_ms",
        "spotify_cover_url", "spotify_popularity",
    ]
    for c in expected_meta_cols:
        if c not in meta_df.columns:
            meta_df[c] = None

    df_en = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)

    # ---------- 2) Audio features (batched) ----------
    audio_cols = {
        "sp_danceability": "danceability",
        "sp_energy": "energy",
        "sp_valence": "valence",
        "sp_speechiness": "speechiness",
        "sp_acousticness": "acousticness",
        "sp_instrumentalness": "instrumentalness",
        "sp_liveness": "liveness",
        "sp_loudness": "loudness",
        "sp_tempo": "tempo",
    }
    for out_col in audio_cols.keys():
        df_en[out_col] = None

    blocked = False
    if try_audio:
        track_ids = [tid for tid in df_en["spotify_track_id"].dropna().astype(str).tolist() if tid]
        feats_by_id, blocked = batch_audio_features(sp, track_ids)

        if not blocked and feats_by_id:
            # Fill features quickly
            tid_series = df_en["spotify_track_id"].astype(str)
            for idx, tid in enumerate(tid_series.tolist()):
                af = feats_by_id.get(tid)
                if not af:
                    continue
                for out_col, in_key in audio_cols.items():
                    df_en.at[idx, out_col] = af.get(in_key)

    # ---------- 3) Covers (optional + capped) ----------
    covers_dir = landing_dir / "album_covers"
    covers_dir.mkdir(parents=True, exist_ok=True)

    covers_downloaded = 0
    cover_files: list[str] = []

    if max_covers > 0:
        unique_albums = (
            df_en[["spotify_album_id", "spotify_cover_url"]]
            .dropna(subset=["spotify_album_id", "spotify_cover_url"])
            .drop_duplicates(subset=["spotify_album_id"])
            .head(max_covers)
            .reset_index(drop=True)
        )

        sess = requests.Session()
        for _, r in tqdm(unique_albums.iterrows(), total=len(unique_albums), desc="Covers: download"):
            album_id = str(r["spotify_album_id"])
            url = str(r["spotify_cover_url"])
            outp = covers_dir / f"{album_id}.jpg"
            did = download_cover(sess, url, outp)
            if outp.exists():
                cover_files.append(outp.name)
            covers_downloaded += int(did)

    # ---------- 4) Save enriched CSV locally ----------
    out_csv = landing_dir / "tracks_enriched.csv"
    df_en.to_csv(out_csv, index=False)

    # ---------- 5) Upload to S3 (non-fatal) ----------
    upload_small(out_csv, bucket, "music/persistent_landing/tracks_enriched.csv", ct="text/csv")
    for fname in cover_files:
        p = covers_dir / fname
        upload_small(landing_dir / "manifest.json", bucket, "music/persistent_landing/_manifests/manifest.json", ct="application/json")

    # ---------- 6) Manifest (downstream must use this) ----------
    manifest = {
        "records": int(len(df_en)),
        "uncached_spotify_searches": int(uncached),
        "covers_downloaded": int(covers_downloaded),
        "covers_in_run": cover_files,           # IMPORTANT: use this in formatted/trusted/index steps
        "audio_features_blocked": bool(blocked),
        "fast_mode": bool(fast_mode),
        "sample_n": int(sample_n),
        "max_covers": int(max_covers),
        "try_audio": bool(try_audio),
    }

    (landing_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    upload_small(
        landing_dir / "manifest.json",
        bucket,
        "music/persistent_landing/_manifests/manifest.json",
        ct="application/json",
    )

    print("=== Landing summary ===")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
