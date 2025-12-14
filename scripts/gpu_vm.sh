#!/bin/bash
# Fast GPU VM management for experimentation
# Usage: ./scripts/gpu_vm.sh [create|start|stop|ssh|delete|status]

set -e

# Configuration
PROJECT="plotpointe"
ZONE="us-central1-a"
VM_NAME="video-transform-gpu"
MACHINE_TYPE="n1-standard-8"  # 8 vCPUs, 30GB RAM
GPU_TYPE="nvidia-l4"          # L4 is fast & cost-effective, use nvidia-tesla-t4 for cheaper
GPU_COUNT=1
DISK_SIZE="200GB"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

create_vm() {
    echo -e "${GREEN}Creating GPU VM: $VM_NAME${NC}"
    gcloud compute instances create $VM_NAME \
        --project=$PROJECT \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --image-family=$IMAGE_FAMILY \
        --image-project=$IMAGE_PROJECT \
        --boot-disk-size=$DISK_SIZE \
        --boot-disk-type=pd-ssd \
        --maintenance-policy=TERMINATE \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --scopes=cloud-platform \
        --metadata="install-nvidia-driver=True"
    
    echo -e "${GREEN}VM created! Waiting for startup...${NC}"
    sleep 30
    
    echo -e "${YELLOW}Setting up environment...${NC}"
    setup_vm
}

setup_vm() {
    gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command="
        # Clone repo and setup
        cd ~ && rm -rf video-transform
        git clone https://github.com/Axionis47/video-identity-transform.git video-transform || true
        cd video-transform
        
        # Install dependencies
        pip install -r requirements.txt
        
        # Download models
        python scripts/download_models.py
        
        echo 'Setup complete!'
    "
}

start_vm() {
    echo -e "${GREEN}Starting VM: $VM_NAME${NC}"
    gcloud compute instances start $VM_NAME --zone=$ZONE --project=$PROJECT
    echo -e "${GREEN}VM started. Wait ~30s for GPU driver, then SSH in.${NC}"
}

stop_vm() {
    echo -e "${YELLOW}Stopping VM: $VM_NAME${NC}"
    gcloud compute instances stop $VM_NAME --zone=$ZONE --project=$PROJECT
    echo -e "${GREEN}VM stopped. No GPU charges while stopped.${NC}"
}

delete_vm() {
    echo -e "${RED}Deleting VM: $VM_NAME${NC}"
    gcloud compute instances delete $VM_NAME --zone=$ZONE --project=$PROJECT --quiet
}

ssh_vm() {
    echo -e "${GREEN}Connecting to VM...${NC}"
    gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT -- -L 8888:localhost:8888
}

status_vm() {
    echo -e "${GREEN}VM Status:${NC}"
    gcloud compute instances describe $VM_NAME --zone=$ZONE --project=$PROJECT \
        --format="table(name,status,machineType.basename(),scheduling.provisioningModel)" 2>/dev/null || echo "VM not found"
}

sync_code() {
    echo -e "${GREEN}Syncing code to VM...${NC}"
    gcloud compute scp --recurse --zone=$ZONE --project=$PROJECT \
        ./* $VM_NAME:~/video-transform/
    echo -e "${GREEN}Code synced!${NC}"
}

case "$1" in
    create) create_vm ;;
    start) start_vm ;;
    stop) stop_vm ;;
    delete) delete_vm ;;
    ssh) ssh_vm ;;
    status) status_vm ;;
    sync) sync_code ;;
    setup) setup_vm ;;
    *)
        echo "Usage: $0 {create|start|stop|delete|ssh|status|sync|setup}"
        echo ""
        echo "  create - Create new GPU VM (SPOT instance, ~\$0.50/hr with L4)"
        echo "  start  - Start stopped VM"
        echo "  stop   - Stop VM (no charges when stopped)"
        echo "  delete - Delete VM completely"
        echo "  ssh    - SSH into VM with port forwarding"
        echo "  status - Show VM status"
        echo "  sync   - Sync local code to VM"
        echo "  setup  - Re-run setup on existing VM"
        ;;
esac

