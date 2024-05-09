#!/bin/bash

TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

# SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# git add "$SCRIPT_DIR"

git add ./Autosave.bash ./Notes.md ./requirements.txt ./sampleModel.py

git commit -m "$TIMESTAMP"

git push
