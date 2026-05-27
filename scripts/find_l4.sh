#!/usr/bin/env bash
# Script to dynamically query zones supporting nvidia-l4 and loop until instance is created.

PROJECT_ID="cse151b-495417"
INSTANCE_NAME="l4-muon"
MACHINE_TYPE="g2-standard-4"
IMAGE_FAMILY="common-gpu-ubuntu-2204"
IMAGE_PROJECT="deeplearning-platform-release"
DISK_SIZE="150GB"

echo "Querying zones with nvidia-l4 accelerator..."
ZONES=$(gcloud compute accelerator-types list --project="$PROJECT_ID" 2>/dev/null | grep nvidia-l4 | awk '{print $2}' | sort -u | tr '\n' ' ')

if [ -z "$ZONES" ]; then
  echo "Error: No zones found with nvidia-l4. Fallback to hardcoded list."
  ZONES="us-west1-a us-west1-b us-west4-a us-east1-b us-east1-c us-east4-a us-central1-a us-central1-b us-central1-c us-central1-f europe-west1-b europe-west4-a asia-east1-a asia-southeast1-a"
fi

echo "Zones to try: $ZONES"

while true; do
  for zone in $ZONES; do
    echo "----------------------------------------"
    echo "Trying zone: $zone"
    echo "----------------------------------------"
    
    if gcloud compute instances create "$INSTANCE_NAME" \
      --zone="$zone" \
      --machine-type="$MACHINE_TYPE" \
      --accelerator=type=nvidia-l4,count=1 \
      --image-family="$IMAGE_FAMILY" \
      --image-project="$IMAGE_PROJECT" \
      --maintenance-policy=TERMINATE \
      --boot-disk-size="$DISK_SIZE" \
      --project="$PROJECT_ID" \
      --quiet; then
      echo ""
      echo "========================================"
      echo "SUCCESS: Instance created in $zone"
      echo "========================================"
      exit 0
    fi
    echo "Failed in $zone. Moving to next..."
  done
  echo "All zones exhausted. Retrying in 60 seconds..."
  sleep 60
done
