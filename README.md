Multimodal Musical Catalog and Advisor

Authors:
Edwin Santiago Delgado Rospigliosi

github: https://github.com/jerry201497/Music-repo

1. Project Overview

This project implements a complete multimodal data management and evaluation pipeline for a Musical Catalog and Advisor system. The objective is to design, implement, and analyze a reproducible, DataOps-oriented architecture capable of handling heterogeneous data modalities, including structured textual metadata and audio-derived features.

The system is intentionally designed as an educational yet realistic prototype, mirroring real-world data engineering, retrieval, and experimental AI workflows. Emphasis is placed not only on implementation, but also on evaluation, interpretability, performance constraints, and critical analysis of limitations.

The project is divided into two main parts:

Part 1 – Multimodal Data Management Pipeline

Focused on data ingestion, normalization, validation, and storage using a zonal DataOps architecture.

Part 2 – Embedding Generation and Multimodal Evaluation

Focused on representation learning, similarity search, cross-modal alignment, and experimental evaluation.

2. Data Modalities

The pipeline operates on multiple data modalities associated with music tracks:

Textual Data
-Track name
-Artist
-Album
-Genre
-Release year
-Derived natural-language descriptions

Audio-Derived Data
-Danceability
-Energy
-Valence
-Acousticness
 Instrumentalness
 Loudness
-Tempo and additional Spotify-style features

Raw audio waveforms are not processed.
Instead, numerical audio features are discretized into quantile-based semantic bins and converted into natural-language descriptors. This design enables multimodal alignment while keeping computation feasible and representations interpretable.

3. Architecture: Zonal Data Pipeline

The project follows a DataOps-inspired zonal architecture, where each zone represents a different stage of data maturity and responsibility.

3.1 Landing Zone

Raw data ingestion
No transformations applied
Single source of truth
Implemented using Python scripts and MinIO (S3-compatible object storage)

3.2 Formatted Zone

Data normalization and schema standardization
Audio features joined and cleaned
Deterministic transformations
Output in CSV and Parquet formats

3.3 Trusted Zone

Data quality validation
Deduplication of records
Schema consistency checks
Automatic generation of data quality reports

3.4 Exploitation Zone

Embedding generation
Vector indexing using ChromaDB
Similarity search (same-modality and cross-modality)
Experimental evaluation and reporting
This architecture ensures traceability, reproducibility, and controlled data evolution.

4. Part 1 – Multimodal Data Management

Part 1 establishes the data backbone required for all downstream analytical and AI tasks. Key outcomes include:
Automated ingestion of raw datasets into object storage
Standardized formatted datasets (tracks_refined.parquet, .csv)
Trusted datasets validated for quality and consistency
Explicit separation of responsibilities between pipeline stages
Full automation via a single pipeline execution command
This phase demonstrates how disciplined data engineering practices enable scalable and auditable systems.

5. Part 2 – Multimodal Embeddings and Evaluation

Part 2 extends the pipeline into experimental embedding-based analysis, treated as a mini-thesis rather than a pure engineering task.

5.1 Embedding Strategy

Text embeddings generated from track metadata
Audio features converted into semantic text via quantile binning
A shared sentence-level embedding model enables cross-modal comparison
All embeddings stored and queried using ChromaDB
This approach prioritizes interpretability and experimental clarity over optimal performance.

5.2 Experiments
Experiment 1 (Baseline)
Text metadata + audio semantic bins
Weakly supervised multimodal alignment

Experiment 2 (Refined)
Improved natural-language descriptors for audio features
Slightly richer semantic representation
Each experiment indexes paired text/audio embeddings and evaluates how well the system retrieves the correct audio representation given a text query.

5.3 Same-Modality Audio Similarity Analysis

In addition to cross-modal evaluation, the project includes a same-modality similarity experiment based purely on audio features.
Tracks are represented as vectors of standardized audio features
Cosine similarity is computed between track pairs
Similarity distributions are compared for:
Same-genre pairs
Different-genre pairs
This analysis reveals that low-level audio features alone do not strongly separate genres, explaining the difficulty observed in cross-modal alignment.
Implemented in:
src/music/audio_similarity.py

Reports generated:
audio_similarity_report.json
audio_similarity_hist.png
audio_similarity_missingness.json

5.4 Evaluation Metrics

Recall@K (K = 1, 5, 10, 20)
Mean Reciprocal Rank (MRR)
Mean and median rank
Rank distribution histograms
Runtime and scalability observations
Evaluation outputs are stored in the reports/ directory.

6. Technologies Used
Component	Technology
Programming	Python 3.11+
Object Storage	MinIO (S3-compatible)
Data Processing	Pandas, NumPy
Vector Database	ChromaDB
Embeddings	Sentence Transformers
Visualization	Matplotlib
Orchestration	Custom pipeline scripts
Configuration	.env environment variables
7. Project Structure
Music-repo/
│
├── src/
│   └── music/
│       ├── landing_ingest.py
│       ├── formatted_prepare.py
│       ├── trusted_validate.py
│       ├── exploitation_index.py
│       ├── eval_exp_d.py
│       ├── audio_similarity.py
│
├── cache/
│   ├── landing/
│   ├── formatted/
│   └── trusted/
│
├── reports/
│   ├── exp_d_eval_metrics.json
│   ├── exp_d_recall_at_k.png
│   ├── exp_d_rank_hist.png
│   ├── audio_similarity_report.json
│   ├── audio_similarity_hist.png
│   ├── audio_similarity_missingness.json
│
├── .env
├── run_music_pipeline.py
└── README.md

8. How to Run the Project
8.1 Set Environment Variables (Windows PowerShell example)
$env:CHROMA_PATH="C:\temp\chroma_db"
$env:MUSIC_INDEX_MAX="200"
$env:MUSIC_EVAL_MAX="200"
$env:MUSIC_EVAL_KS="1,5,10,20"

8.2 Run Full Pipeline (Part 1)
python run_music_pipeline.py

8.3 Run Embedding Indexing and Evaluation (Part 2)
python src/music/exploitation_index.py
python src/music/eval_exp_d.py
python src/music/audio_similarity.py

9. Results Summary

The pipeline successfully processes data through all zones
Multimodal embeddings are generated and indexed
Same-modality audio similarity shows strong overlap across genres
Cross-modal retrieval works but exhibits limited alignment at low ranks
Results confirm the difficulty of aligning audio and text via indirect representations
Final runtime is fast and suitable for iterative experimentation

10. Limitations

Audio is represented indirectly, resulting in information loss
Raw audio access is restricted and impractical at scale
Embedding models are not music-specialized
Dataset size is limited for controlled experimentation
No user-facing interface is provided
These limitations are explicitly acknowledged and analyzed as part of the project.

11. Future Work

Potential extensions include:
Integration of audio-specific embedding models
Direct audio embedding where licensing permits
Larger-scale datasets
Interactive user interfaces for exploration and recommendation
Retrieval-Augmented Generation (RAG) for music discovery

