# test_spotify_enrichment.py
"""
Quick proof script:
- Confirms Spotify API credentials work
- Performs a track search
- Prints key metadata fields (track id, album, release date, popularity, cover URL)
- Optionally attempts audio_features (may be blocked with 403)
"""

import os
import json

from src.music.landing_ingest import spotify_client, search_track, batch_audio_features


def main():
    # ---- Choose a test query (override with env vars if you want) ----
    track = os.getenv("SPOTIFY_TEST_TRACK", "Billie Jean")
    artist = os.getenv("SPOTIFY_TEST_ARTIST", "Michael Jackson")

    print("=== Spotify enrichment smoke test ===")
    print(f"Query: track='{track}' | artist='{artist}'")

    sp = spotify_client()

    # ---- 1) Search metadata ----
    meta = search_track(sp, track, artist)
    if not meta:
        print("[FAIL] No results returned from Spotify search.")
        return

    # Pretty print the most relevant fields
    proof = {
        "spotify_track_id": meta.get("spotify_track_id"),
        "spotify_album_name": meta.get("spotify_album_name"),
        "spotify_release_date": meta.get("spotify_release_date"),
        "spotify_duration_ms": meta.get("spotify_duration_ms"),
        "spotify_popularity": meta.get("spotify_popularity"),
        "spotify_cover_url": meta.get("spotify_cover_url"),
        "spotify_album_id": meta.get("spotify_album_id"),
        "spotify_artist_ids": meta.get("spotify_artist_ids"),
    }

    print("\n[OK] Spotify search returned metadata:")
    print(json.dumps(proof, indent=2, ensure_ascii=False))

    # ---- 2) OPTIONAL: Try audio features (may be blocked) ----
    try_audio = os.getenv("SPOTIFY_TEST_TRY_AUDIO", "0") == "1"
    if try_audio:
        tid = meta.get("spotify_track_id")
        print("\n[INFO] Trying audio_features for track_id:", tid)

        feats_by_id, blocked = batch_audio_features(sp, [tid] if tid else [])
        if blocked:
            print("[WARN] Spotify audio_features blocked (403). This is expected for many apps.")
        else:
            feats = feats_by_id.get(tid)
            if feats:
                # print only a few common ones
                subset = {k: feats.get(k) for k in [
                    "danceability", "energy", "valence", "acousticness",
                    "instrumentalness", "loudness", "tempo"
                ]}
                print("[OK] audio_features returned:")
                print(json.dumps(subset, indent=2, ensure_ascii=False))
            else:
                print("[WARN] audio_features returned no features for this track id.")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()