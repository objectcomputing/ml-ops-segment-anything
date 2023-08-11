docker build --no-cache -t sam -f ./ml-ops-segment-anything/vertexAI_pipeline/Dockerfile .
docker tag sam gcr.io/ml-ops-segment-anything/sam
docker push gcr.io/ml-ops-segment-anything/sam