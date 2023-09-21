docker build --no-cache -t sam -f ./build/Dockerfile .
docker tag sam us-west1-docker.pkg.dev/ml-ops-segment-anything/sam-container/online-container
docker push us-west1-docker.pkg.dev/ml-ops-segment-anything/sam-container/online-container

