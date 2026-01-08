Multimodal Musical Catalog and Advisor

Algorithms, Data Structures and Databases

Authors:
Edwin Santiago Delgado Rospigliosi
Toni

Date: October 2025

1. Project Overview

This project implements a complete multimodal data management pipeline for a Musical Catalog and Advisor system. The objective is to design, implement, and evaluate a reproducible DataOps-oriented architecture capable of handling heterogeneous data modalities, including textual metadata and audio-derived features.

The project is divided into two main parts:

Part 1 – Multimodal Data Management Pipeline
Focused on data ingestion, normalization, quality control, and storage using a zonal architecture.

Part 2 – Embedding Generation and Multimodal Evaluation
Focused on representation learning, cross-modal alignment, similarity search, and experimental evaluation.

The system is designed as an educational but realistic prototype, mirroring real-world data engineering and AI workflows.

2. Data Modalities

The pipeline operates on multiple data modalities related to music tracks:

Textual data

Track name

Artist

Album

Genre

Year

Derived natural-language descriptions

Audio-derived data

Danceability

Energy

Valence

Acousticness

Instrumentalness

Loudness

Additional Spotify-style audio features

Audio is not processed as raw waveforms; instead, numerical features are quantized into semantic bins and converted into natural-language descriptors for multimodal alignment.

3. Architecture: Zonal Data Pipeline

The project follows a DataOps-inspired zonal architecture, where each zone represents a different stage of data maturity.

3.1 Landing Zone

Raw data ingestion

No transformations applied

Acts as the single source of truth

Implemented using Python scripts and MinIO (S3-compatible storage)

3.2 Formatted Zone

Data normalization and standardization

Audio features joined and cleaned

Consistent schema enforced

Deterministic transformations for reproducibility

3.3 Trusted Zone

Data quality validation

Deduplication

Schema consistency checks

Generation of data quality reports

3.4 Exploitation Zone

Embedding generation

Vector indexing

Multimodal similarity search

Experimental evaluation

4. Part 1 – Multimodal Data Management

Part 1 focuses on building a robust and reproducible data pipeline. The main outcomes include:

Automated ingestion of raw data into object storage

Standardized formatted datasets (tracks_refined.parquet, .csv)

Trusted datasets validated for quality and consistency

Clear separation of responsibilities between pipeline stages

Full automation via a single pipeline execution command

This part establishes the data backbone required for all downstream analytical and AI tasks.

5. Part 2 – Multimodal Embeddings and Evaluation

Part 2 extends the pipeline into experimental embedding-based analysis, treated as a mini-thesis rather than a pure engineering task.

5.1 Embedding Strategy

Text embeddings generated from track metadata

Audio features converted into semantic text using quantile binning

A shared embedding model is used to allow cross-modal comparison

All embeddings stored in ChromaDB

5.2 Experiments

Experiment D
Text + audio semantic bins (baseline)

Experiment D2
Refined natural-language audio descriptors

Each experiment indexes paired text/audio representations and evaluates how well the system retrieves the correct audio representation given a text query.

5.3 Evaluation Metrics

Recall@K (K = 1, 5, 10, 20)

Mean Reciprocal Rank (MRR)

Rank distribution statistics

Performance and runtime considerations

Evaluation scripts generate both numerical summaries and plots, stored in the reports/ directory.

6. Technologies Used
Component	Technology
Programming	Python 3.11+
Object Storage	MinIO (S3-compatible)
Data Processing	Pandas, NumPy
Vector Database	ChromaDB
Embeddings	Sentence Transformers
Visualization	Matplotlib
Orchestration	Custom pipeline scripts
Environment	.env configuration
7. Project Structure
ADSDB_2025/
│
├── src/
│   └── music/
│       ├── landing_ingest.py
│       ├── formatted_prepare.py
│       ├── trusted_validate.py
│       ├── exploitation_index.py
│       ├── eval_exp_d.py
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
│
├── .env
├── run_music_pipeline.py
└── README.md

8. How to Run the Project
8.1 Set Environment Variables (example – Windows PowerShell)
$env:CHROMA_PATH="C:\temp\chroma_db"
$env:MUSIC_INDEX_MAX="200"
$env:MUSIC_EVAL_MAX="200"
$env:MUSIC_EVAL_KS="1,5,10,20"

8.2 Run Full Pipeline (Part 1)
python run_music_pipeline.py

8.3 Run Exploitation and Evaluation (Part 2)
python src/music/exploitation_index.py
python src/music/eval_exp_d.py

9. Results

The pipeline successfully processes data through all zones.

Multimodal embeddings are generated and indexed.

Cross-modal retrieval works but shows limited precision at low ranks.

Results highlight the difficulty of aligning audio and text using indirect representations.

Performance is fast and suitable for experimentation.

10. Limitations

Audio is represented indirectly, leading to information loss.

Embedding models are not music-specialized.

Dataset size is limited.

Experiments are exploratory rather than optimized.

These limitations are acknowledged and explicitly discussed as part of the evaluation.

11. Future Work

Planned extensions include:

Direct audio embeddings

Modality-specific encoders

Larger-scale datasets

Interactive recommendation interfaces

Retrieval-Augmented Generation (RAG) for music discovery

12. Educational Value

This project demonstrates:

End-to-end DataOps pipelines

Multimodal data integration

Vector databases and semantic search

Experimental evaluation and critical analysis

It serves as a complete academic example of modern data engineering and multimodal AI systems