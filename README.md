# MLB Elo Ratings

## Overview

This project provides tools and notebooks for recreating and building on FiveThirtyEight's Elo Rating system for baseball teams. It includes an **EloTracker** class to automatically produce the latest Elos for each team, given the results presented in an inputted box score dataframe. It also includes an MLBSimulator class to simulated a given schedule dataframe, and see which teams are most successful, using Elos to estimate per-game win probabilities.

---

## Project Highlights

- **Improved Elo rating system:** Re-implemented and enhanced FiveThirtyEight’s Elo model for MLB teams by learning custom rating adjustments for per-game variables, including home advantage, rest days, travel distance, and pitcher ability via logistic regression. This improved 2025 season prediction accuracy from 55.6% to 56.2%.
- **Season simulations:** Leveraged custom Elo ratings to simulate full seasons, generating expected wins and playoff probabilities, improving $R^2$ for 2025 regular season wins from 0.46 to 0.49.
- **Exploring rule changes:** Currently experimenting with simulating seasons under potential rule changes to understand team impacts.

---

## Project Structure

```text
analysis/
├── **data/**  
│   ├── **clean/** – Preprocessed and cleaned datasets  
│   └── **raw/** – Original raw CSV files  
├── src/
│   ├── preprocess/
│   │   └── notebooks/            # Notebooks for cleaning and preprocessing data
│   ├── simulation/
│   │   ├── simulation.py         # Includes MLBSimulator class
│   │   └── __init__.py
│   ├── utils/
│   │   ├── math_utils.py         # Math utility functions
│   │   ├── misc_utils.py         # Miscellaneous utility functions
│   │   └── __init__.py
│   └── elos/
│       ├── elo_tracker.py        # Includes EloTracker class
│       └── __init__.py
└── notebooks/
    ├── feature_analysis/         # Analysis of team performance and features
    ├── simulation/               # Running simulation notebooks
    ├── model_selection/          # Comparing models and evaluation vs 538 baseline
    └── presentation/             # Presentation-ready notebooks and visualizations
```
---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/lanceHEN/Elo-Ratings.git
cd analysis
```
2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```
3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Usage

### Preprocessing Data

Notebooks under `src/preprocess/notebooks/` are for cleaning raw CSVs and generating cleaned datasets in `data/clean/`.

### Simulations

Use scripts and notebooks under `src/simulation/` and `src/notebooks/simulation/` to run Elo simulations and other analyses.

### Feature Analysis and Model Selection

`notebooks/feature_analysis/` and `notebooks/model_selection/` contain notebooks for exploring features, comparing models, and evaluating predictions.

## Utilities

`src/utils/` contains helper modules:
- `math_utils.py` — mathematical and statistical functions
- `misc_utils.py` — miscellaneous helper functions
- `elos_utils.py` — Elo-specific helper functions

`src/elos/elo_tracker.py` provides the EloTracker class for tracking and updating Elo ratings.

## Data
`data/raw/` — Original datasets

`data/clean/` — Cleaned datasets generated from preprocessing notebooks

Recommended workflow: preprocess raw CSVs -> generate cleaned datasets -> run simulations and analyses