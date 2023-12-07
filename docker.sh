#!/bin/bash

tag=$1
container_name=$2
image_name="rainy/crack:$tag"
workspace_dir="/workspace/$container_name"

if [ "$(docker image inspect "$image_name" 2>/dev/null)" = "[]" ]; then
    docker buildx build -t "$image_name" "$PWD"
else
    if docker ps -a --format "{{.Names}}" | grep -qw "$container_name"; then
        docker stop "$container_name"
    fi
fi

docker run -itd --name "$container_name" --rm \
    --gpus all \
    --net host \
    -v "$PWD":/workspace/"$container_name" \
    "$image_name"
docker exec -it "$container_name" /bin/bash -c "$workspace_dir/run.sh $workspace_dir"
