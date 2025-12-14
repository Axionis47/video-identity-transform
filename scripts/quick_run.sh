#!/bin/bash
# Quick run script for fast experimentation
# Syncs code and runs on GPU VM in one command

set -e

PROJECT="plotpointe"
ZONE="us-central1-a"
VM_NAME="video-transform-gpu"

# Sync code
echo "📤 Syncing code to VM..."
gcloud compute scp --recurse --zone=$ZONE --project=$PROJECT \
    --compress \
    modules/ main.py configs/ scripts/ utils/ requirements.txt \
    $VM_NAME:~/video-transform/

# Run command on VM
echo "🚀 Running on GPU..."
gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command="
    cd ~/video-transform
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo '---'
    $@
"

