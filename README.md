# 🎵 Music Genre Classifier

A machine learning project for classifying music genres from audio files using various techniques.

## Overview

This project explores different approaches to automatic music genre classification:

| Model | Status | Test Accuracy |
|-------|--------|---------------|
| CNN on Mel Spectrograms | ✅ Complete | **81.3%** |
| Random Forest on Extracted Features | 🚧 In Progress | — |
| Pretrained Models | 🚧 In Progress | — |

## Dataset

**GTZAN Dataset** — A benchmark dataset for music genre classification.

- 1,000 audio tracks (30 seconds each)
- 10 genres: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock
- Includes pre-computed mel spectrograms and extracted audio features

📎 [Dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)

## Model Architectures

### CNN (Convolutional Neural Network)

Built with 4 convolutional blocks followed by Global Average Pooling (GAP).

**Convolutional Block:**
```
Conv2D (3×3) → BatchNorm → Conv2D (3×3) → BatchNorm → MaxPool (2×2) → Dropout
```

**Results:** 80.0% validation accuracy | 81.3% test accuracy

### Random Forest
*Coming soon*

### Pretrained Models
*Coming soon*

## Project Structure

```
├── Data/                 # GTZAN dataset
├── config.py             # Training hyperparameters
├── dataset.py            # Data preprocessing and loading
├── model.py              # Model architecture definitions
├── main.py               # Training script
└── environment.yml       # Conda environment dependencies
```

## Getting Started

### Prerequisites

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music-genre-classifier.git
   cd music-genre-classifier
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate genre_classifier_env
   ```

### Training

Run the training script:
```bash
python main.py
```

After training completes, check the generated output files:
- `confusion_matrix.png` — Model performance across genres
- `training_stats.png` — Training and validation metrics over epochs

## Roadmap

- [x] CNN model with mel spectrograms
- [ ] Random Forest with handcrafted audio features
- [ ] Transfer learning with pretrained models
- [ ] Web demo / inference script

## References

- [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data) by Andrada Olteanu on Kaggle
