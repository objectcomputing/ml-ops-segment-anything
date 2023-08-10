# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest

COPY ./ml-ops-segment-anything/vertexAI_pipeline/requirements.txt /
RUN python3 -m pip install --no-cache-dir -r requirements.txt