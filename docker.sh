#!/bin/bash
CONFIG_PATH="resources/config/configuration.json"
CRACK500_PATH="resources/data/crack500"

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

jq --arg new_path "$workspace_dir/$CRACK500_PATH" '.["data path"]["data_path"] = $new_path' "$CONFIG_PATH" >tmp.json
mv tmp.json "$CONFIG_PATH"

docker run -itd --name "$container_name" --rm \
    --gpus all \
    --net host \
    -v "$PWD":/workspace/"$container_name" \
    "$image_name"
docker exec -d "$container_name" /bin/bash -c "$workspace_dir/run.sh $workspace_dir"
