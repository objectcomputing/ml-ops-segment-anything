docker build --no-cache -t sam -f ./Dockerfile .
docker tag sam gcr.io/ml-ops-segment-anything/sam
docker push gcr.io/ml-ops-segment-anything/sam
