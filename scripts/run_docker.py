import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List

import docker
from docker import DockerClient
from docker.models.containers import Container
from docker.types import DeviceRequest

_MAX_CONCURRENT = 2
_GPU_IDS = ["0", "1"]


def _is_image_exists(client: DockerClient, image_name: str) -> bool:
    try:
        client.images.get(image_name)
        return True
    except docker.errors.ImageNotFound:
        return False
    except docker.errors.APIError as e:
        print(f"An error occurred: {e}")
        return False


def _build_image(client: DockerClient, image_name: str, path: str = ".") -> None:
    print(f"Building image {image_name}")
    client.images.build(path=path, tag=image_name)


def _stop_container_if_exists(client: DockerClient, container_name: str) -> None:
    try:
        container = client.containers.get(container_name)
        print(f"Stopping and removing existing container {container_name} ...")
        container.stop()
        container.wait()
    except docker.errors.NotFound:
        pass
    except docker.errors.APIError as e:
        print(f"Error stopping/removing container {container_name}: {e}")
        if "removal of container" in str(e) and "is already in progress" in str(e):
            print(
                f"Waiting for the existing removal process of container {container_name} to complete..."
            )
            time.sleep(5)
            _stop_container_if_exists(client, container_name)


def _run_container(
    client: DockerClient,
    command: str,
    container_name: str,
    image_name: str,
    gpu_id: str,
    workspace_dir: str,
) -> None:
    _stop_container_if_exists(client, container_name)
    print(f"Running container {container_name} with image {image_name} on GPU {gpu_id}")
    device_requests = [DeviceRequest(count=-1, capabilities=[["gpu"]])]
    container: Container = client.containers.run(
        image_name,
        command,
        detach=True,
        name=container_name,
        remove=True,
        device_requests=device_requests,
        runtime="nvidia",
        network_mode="host",
        shm_size="16G",
        environment={
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu_id,
        },
        volumes={os.getcwd(): {"bind": workspace_dir, "mode": "rw"}},
    )
    container.start()
    container.wait()
    print(f"Container {container_name} on GPU {gpu_id} has finished.")


def main(tag: str, model_names: List[str]) -> None:
    client = docker.from_env()
    image_name = f"rainy/crack:{tag}"
    container_names = [f"Crack_{model_name}" for model_name in model_names]
    start_time = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")

    if not _is_image_exists(client, image_name):
        _build_image(client, image_name)

    with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT) as executor:
        future_to_container = {}
        for i, (container_name, model_name) in enumerate(
            zip(container_names, model_names)
        ):
            workspace_dir = f"/workspace/{container_name}"
            command = f"/bin/bash -c '{workspace_dir}/scripts/schedule.sh {workspace_dir} {model_name} {start_time}'"
            gpu_id = _GPU_IDS[i % len(_GPU_IDS)]
            args = [client, command, container_name, image_name, gpu_id, workspace_dir]

            future = executor.submit(_run_container, *args)
            future_to_container[future] = (container_name, gpu_id)

        for future in as_completed(future_to_container):
            container_name, gpu_id = future_to_container[future]
            try:
                future.result()
            except Exception as exc:
                print(
                    f"Container {container_name} on GPU {gpu_id} generated an exception: {exc}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="1.0.4", help="Docker tag to use")
    parser.add_argument(
        "--model_names", type=str, default="", help="Experiment names to run"
    )
    args = parser.parse_args()
    if args.model_names:
        args.model_names = args.model_names.split(",")

    main(args.tag, args.model_names)
