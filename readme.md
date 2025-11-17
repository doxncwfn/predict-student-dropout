# Predict Student Dropout

Predict Student Dropout is a machine learning project for predicting whether a student will drop out based on academic and demographic features. The repository contains code for data preprocessing, model training, evaluation, and example notebooks to explore results.

## Features

- Data cleaning and feature engineering scripts
- Multiple model implementations (baseline models + tuned classifiers)
- Training and evaluation pipelines
- Jupyter notebooks with exploratory data analysis and model interpretation
- Instructions to reproduce experiments locally or via Docker

## Getting Started

You can run this project locally (Python) or deploy using Docker.

### Prerequisites

- Python 3.8+ (if running locally)
- pip
- (Optional) A virtual environment tool such as venv or conda
- Docker (for containerized deployment)
- Docker Compose (optional, for compose-based deployment)

### Install (local Python)

Clone the repository and install dependencies:

```bash
git clone https://github.com/doxncwfn/predict-student-dropout.git
cd predict-student-dropout
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run (local)

- Preprocess data:

```bash
python src/preprocess.py --input data/raw/dataset.csv --output data/processed/
```

- Train:

```bash
python src/train.py --config configs/train.yaml
```

- Evaluate:

```bash
python src/evaluate.py --model models/best_model.pkl --test data/processed/test.csv
```

(Adjust commands/arguments to match scripts in the repository.)

## Docker deployment

This project can be built and run with Docker. Use the instructions below to build the image and run a container.

Note: If your repo already contains a Dockerfile, these commands assume the Dockerfile is at the repository root. If it's elsewhere or named differently, adjust the paths/commands.

### Build the Docker image

From the project root:

```bash
docker build -t predict-student-dropout:latest .
```

### Run the container

Basic run (replace any environment values as needed):

```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e SOME_ENV_VAR=value \
  predict-student-dropout:latest
```

- -p 8000:8000 maps container port 8000 to host. Adjust based on your app's port.
- -v mounts the host data directory into the container (useful for datasets or model artifacts).
- -e passes environment variables (model paths, config flags, credentials — do not commit secrets).

### Run with Docker Compose

Create a docker-compose.yml (example below) or use the provided one if present.

Example docker-compose.yml:

```yaml
version: "3.8"
services:
  app:
    image: predict-student-dropout:latest
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - SOME_ENV_VAR=value
```

Start:

```bash
docker-compose up --build
```

### Healthchecks and restart policy

For production-like setups, add a HEALTHCHECK and restart policy in your Dockerfile / docker-compose.yml:

```yaml
restart: unless-stopped
```

### Rebuild after changes

If you change code, rebuild the image:

```bash
docker build -t predict-student-dropout:latest .
# Then restart container / compose
docker-compose up --build
```

### Stopping and removing containers

```bash
# Stop by container id
docker stop <container_id>

# Remove dangling images
docker image prune
```

### Notes on volumes and persistence

- Use volumes to persist trained models, logs, and processed data outside the container.
- Do not store secrets in files committed to the repo. Use environment variables, Docker secrets, or your orchestration platform's secret manager.

## Repository Structure

- data/ - (Optional) raw and processed datasets
- notebooks/ - Jupyter notebooks for EDA and experiments
- src/ - Source code (preprocessing, models, training)
- scripts/ - Utility scripts for running experiments
- models/ - Saved model artifacts and checkpoints
- Dockerfile - (optional) Dockerfile used to build the image
- docker-compose.yml - (optional) Compose file for local container orchestration
- README.md - This file

Adjust paths above if your repo structure differs.

## Notebooks

Open the notebooks in the notebooks/ folder to reproduce exploratory analysis and visualizations:

```bash
jupyter lab
# then open notebooks/*.ipynb
```

## Results & Metrics

Typical evaluation metrics included:

- Accuracy
- Precision / Recall
- F1-score
- ROC AUC

Include the actual numbers and plots in this section after running experiments.

## Contact

Maintainer: doxncwfnthehcmuter
Email: doxncwfn@gmail.com

```

```
