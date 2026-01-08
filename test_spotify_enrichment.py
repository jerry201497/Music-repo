import os
import pandas as pd
from src.music.landing_ingest import spotify_client, get_spotify_enrichment

# Turn on debug logs inside get_spotify_enrichment
os.environ["MUSIC_DEBUG_SPOTIFY"] = "1"

df = pd.read_csv("data/music/raw/tracks.csv")

# Just test a few specific songs
sample = df[df["track_name"].isin([
    "velvet light",
    "andy, you're a star",
])].head()

sp = spotify_client()

for _, row in sample.iterrows():
    print("\n=== TEST ROW ===")
    print(row["track_name"], " - ", row["artist_name"])
    e = get_spotify_enrichment(sp, row)
    # show only spotify_* and sp_* keys with values
    filtered = {k: v for k, v in e.items() if ("spotify_" in k or k.startswith("sp_")) and v is not None}
    print("Enrichment keys:", filtered)
