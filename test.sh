#!/bin/bash

set -e

DATASET_URL="https://huggingface.co/datasets/abbasfar/backdoor_attack_evaluation_dataset/resolve/main/eval_dataset.zip"
DEST_DIR="eval_dataset"
ZIP_FILE="$DEST_DIR/eval_dataset.zip"

mkdir -p "$DEST_DIR"

wget -P "$DEST_DIR" "$DATASET_URL"

unzip "$ZIP_FILE" -d "$DEST_DIR"

rm "$ZIP_FILE"

echo "Dataset downloaded and extracted successfully to '$DEST_DIR/'."
