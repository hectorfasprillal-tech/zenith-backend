# Zenith Backend

Backend API for the Zenith project.

This repository contains the backend services for Zenith, including:
- REST API implemented in Python
- Machine Learning and NLP processing
- Vector search integration with Pinecone
- Data ingestion and synchronization scripts
- Containerized execution with Docker
- Deployment-ready for Kubernetes / AKS

## Tech Stack

- Python 3.11
- Flask + Gunicorn
- Sentence Transformers / Torch
- LangChain
- Pinecone
- Docker
- Kubernetes (AKS)

## Container Build

The backend is designed to be built as a Docker image.

Example:

```bash
docker build -t zenith-backend .
docker run -p 3000:3000 zenith-backend

