FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
# FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest

COPY ./build/requirements.txt /requirements.txt
RUN apt-get update && apt-get install -y python3-opencv
RUN python3 -m pip install --no-cache-dir -r /requirements.txt
# RUN apt-get update && apt-get install -y libgl1-mesa-dev