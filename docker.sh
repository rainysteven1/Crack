#!/bin/bash
TAG=$1
IMAGE_NAME="rainy/unet:$TAG"
CONTAINER_NAME="RETINA_UNET"

if [ "$(docker image inspect "$IMAGE_NAME" 2>/dev/null)" = "[]" ]; then
    docker buildx build -t "$IMAGE_NAME" "$PWD"
else
    if docker ps -a --format "{{.Names}}" | grep -qw "$CONTAINER_NAME"; then
        docker stop "$CONTAINER_NAME" && docker rm "$CONTAINER_NAME"
    fi
    docker run -itd --name "$CONTAINER_NAME" --rm \
        --gpus all \
        --net host \
        -v "$PWD":/workspace/"$CONTAINER_NAME" \
        "$IMAGE_NAME"
    docker exec -it "$CONTAINER_NAME" /bin/bash -c "/workspace/"$CONTAINER_NAME"/run.sh $CONTAINER_NAME"
fi
