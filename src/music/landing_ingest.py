
import os, json, time, requests
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

def s3():
    load_dotenv()
    return boto3.resource(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD"),
        region_name="us-east-1",
    )

def ensure_bucket(name):
    s3r = s3()
    try:
        s3r.meta.client.head_bucket(Bucket=name)
    except ClientError:
        s3r.create_bucket(Bucket=name)

def upload_file(bucket, local_path, key, content_type=None):
    extra = {"ContentType": content_type} if content_type else {}
    s3().Bucket(bucket).upload_file(str(local_path), key, ExtraArgs=extra)

def spotify_client():
    load_dotenv()
    auth = SpotifyClientCredentials(
        os.getenv("SPOTIFY_CLIENT_ID"), os.getenv("SPOTIFY_CLIENT_SECRET")
    )
    return spotipy.Spotify(client_credentials_manager=auth)

def get_spotify_enrichment(sp, row):
    q = f"track:{row['track_name']} artist:{row['artist_name']}"
    try:
        res = sp.search(q, limit=1, type="track")
        items = res.get("tracks", {}).get("items", [])
        if not items:
            return {}

        t = items[0]
        album = t.get("album", {})

        # ---- Artists & genres ----
        artist_ids = [a["id"] for a in t.get("artists", []) if a.get("id")]
        genres = []
        artist_cover_url = None

        for aid in artist_ids:
            try:
                a = sp.artist(aid)
                genres.extend(a.get("genres", []))

                # fallback cover from artist image (largest)
                imgs = a.get("images", [])
                if imgs and not artist_cover_url:
                    # Spotify returns sorted desc by size often, but we sort to be safe
                    artist_cover_url = sorted(imgs, key=lambda x: x.get("width", 0))[-1]["url"]

                time.sleep(0.05)
            except Exception:
                pass

        # ---- Cover: album first ----
        cover_url = None
        album_images = album.get("images", [])
        if album_images:
            # take largest
            cover_url = sorted(album_images, key=lambda x: x.get("width", 0))[-1]["url"]

        # ---- Fallback if album has no image ----
        if not cover_url:
            cover_url = artist_cover_url

        return {
            "spotify_track_id": t.get("id"),
            "spotify_artist_ids": artist_ids,
            "spotify_album_id": album.get("id"),
            "spotify_cover_url": cover_url,
            "spotify_genres": list(sorted(set(genres))) if genres else None,
            "spotify_popularity": t.get("popularity"),
            # extra fields you might want later:
            "spotify_album_name": album.get("name"),
            "spotify_release_date": album.get("release_date"),
            "spotify_duration_ms": t.get("duration_ms"),
        }

    except Exception:
        return {}

def download_cover(url, out_path):
    if not url: return False
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False

def main():
    load_dotenv()
    bucket = os.getenv("S3_BUCKET_LANDING", "landing_zone")
    ensure_bucket(bucket)

    csv_path = Path("data/music/raw/tracks.csv")
    if not csv_path.exists():
        raise FileNotFoundError("Missing input: data/music/raw/tracks.csv")
    df = pd.read_csv(csv_path)
    df = df.sample(3000, random_state=42) # for quicker testing; remove in prod
    keep = [c for c in ["track_name","artist_name","album_name","genre","year"] if c in df.columns]
    df = df[keep].dropna(subset=["track_name","artist_name"]).drop_duplicates()

    sp = spotify_client()
    enrich_rows = []
    covers_downloaded = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Enriching via Spotify"):
        e = get_spotify_enrichment(sp, row)
        enrich_rows.append(e)

    df_en = pd.concat([df.reset_index(drop=True), pd.DataFrame(enrich_rows)], axis=1)

    covers_dir = Path("cache/landing/album_covers")
    for i, r in df_en.iterrows():
        cid = r.get("spotify_album_id") or r.get("spotify_track_id") or f"cover_{i}"
        url = r.get("spotify_cover_url")
        ok = download_cover(url, covers_dir/f"{cid}.jpg")
        covers_downloaded += int(ok)

    landing_dir = Path("cache/landing")
    landing_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = landing_dir/"tracks_enriched.csv"
    df_en.to_csv(meta_csv, index=False)

    upload_file(bucket, meta_csv, "music/persistent_landing/tracks_enriched.csv", content_type="text/csv")

    for p in covers_dir.glob("*.jpg"):
        upload_file(bucket, p, f"music/persistent_landing/covers/{p.name}", content_type="image/jpeg")

    manifest = {"records": len(df_en), "covers_downloaded": covers_downloaded}
    (landing_dir/"manifest.json").write_text(json.dumps(manifest, indent=2))
    upload_file(bucket, landing_dir/"manifest.json", "music/persistent_landing/_manifests/manifest.json", content_type="application/json")

    print("=== Landing summary ===")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
