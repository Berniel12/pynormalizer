# Deployment Guide for Python Normalizer

This guide provides instructions for deploying the latest version of the Python Normalizer with validation fixes.

## Prerequisites

- Docker installed and running
- Access to your container registry
- Permissions to update production services

## Deployment Steps

### 1. Build the Docker Image

The image has already been built locally with the tag `pynormalizer:latest`. This image includes all the validation fixes for:

- Country validation in all tender sources (SAM.gov, TED EU, UNGM, ADB, etc.)
- Title extraction for ADB and IADB tenders
- Date parsing for AFD and AFDB tenders ("Unknown" dates and format issues)
- Robust fallback mechanisms for all required fields

### 2. Customize the Deployment Script

Edit the `deploy.sh` script and update the `REGISTRY` variable with your actual registry URL.

```bash
# Example:
REGISTRY="registry.yourdomain.com" 
```

### 3. Run the Deployment Script

```bash
./deploy.sh
```

This will:
- Tag the image with a timestamp-based version
- Push both the versioned and latest images to your registry

### 4. Update Production Services

#### If Using Docker Compose

Update your `docker-compose.yml` file with the new image:

```yaml
services:
  normalizer:
    image: your-registry-url/pynormalizer:latest
    # ... other configuration ...
```

Then apply the changes:

```bash
docker-compose up -d
```

#### If Using Kubernetes

Update your deployment manifest with the new image:

```yaml
spec:
  containers:
  - name: normalizer
    image: your-registry-url/pynormalizer:latest
    # ... other configuration ...
```

Then apply the changes:

```bash
kubectl apply -f deployment.yaml
```

## Verification

After deployment, monitor the logs to ensure:

1. No validation errors for empty country values
2. No validation errors for empty titles 
3. No date parsing errors for AFD tenders
4. No issues with "Unknown" publication dates for AFDB

If you encounter any issues, please check the application logs for details. 