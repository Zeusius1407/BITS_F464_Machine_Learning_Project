#!/usr/bin/env bash
# Full pipeline: raw CSVs in data/raw -> processed tables -> models -> results/
set -e
cd "$(dirname "$0")"
python src/clean.py
python src/features_match.py
python src/features_player.py
python src/models_innings.py
python src/models_winner.py
python src/toss_bayes.py
python src/player_roles.py
