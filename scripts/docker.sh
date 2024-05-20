#!/bin/bash

tag=$1
container_name="Crack_$2"
gpu_id=$3
image_name="rainy/crack:$tag"
workspace_dir="/workspace/$container_name"

if [ "$(docker image inspect "$image_name" 2>/dev/null)" = "[]" ]; then
    docker buildx build -f ./DockerFile -t "$image_name" "$PWD"
else
    if docker ps -a --format "{{.Names}}" | grep -qw "$container_name"; then
        docker stop "$container_name"
    fi
fi

docker run -itd --name "$container_name" --rm \
    --gpus all \
    --net host \
    --shm-size 16G \
    -e CUDA_DEVICE_ORDER="PCI_BUS_ID" \
    -e CUDA_VISIBLE_DEVICES="$gpu_id" \
    -v "$PWD":/workspace/"$container_name" \
    "$image_name"
docker exec -d "$container_name" /bin/bash -c "$workspace_dir/scripts/schedule.sh $workspace_dir $2"
