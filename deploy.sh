#!/bin/bash
set -e

# Configuration
IMAGE_NAME="pynormalizer"
VERSION=$(date +"%Y%m%d%H%M%S")
REGISTRY="your-registry-url"  # Replace with your actual registry URL

# Step 1: Tag the image with a unique version
echo "Tagging image as ${REGISTRY}/${IMAGE_NAME}:${VERSION}..."
docker tag ${IMAGE_NAME}:latest ${REGISTRY}/${IMAGE_NAME}:${VERSION}
docker tag ${IMAGE_NAME}:latest ${REGISTRY}/${IMAGE_NAME}:latest

# Step 2: Push the images to the registry
echo "Pushing images to registry..."
docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
docker push ${REGISTRY}/${IMAGE_NAME}:latest

echo "==========================================="
echo "Deployment Complete!"
echo "Image: ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
echo "==========================================="
echo ""
echo "To deploy to production, update your infrastructure with this new image."
echo "If using Docker Compose, update the image reference in your docker-compose.yml file."
echo "If using Kubernetes, update your deployment manifest and apply the changes."
echo ""
echo "The new image includes fixes for:"
echo "- Country validation in all tender sources"
echo "- Title extraction for ADB and IADB tenders"
echo "- Date parsing for AFD and AFDB tenders"
echo "- Robust fallback mechanisms for all required fields" 