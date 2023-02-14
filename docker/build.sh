#!/bin/bash

# This script build Docker image one can run IsoNet inside.
# See interactive.sh to see how to run the built Docker image

set -Eeo pipefail

##############################

SELF_DIR="$(dirname "$(realpath "${0}")")"
REPO_ROOT="$(realpath "${SELF_DIR}/..")"

##############################

# Lower version might be required depending on your Cuda driver version
TF_ver="latest"

##############################

docker_build_context_dir="${SELF_DIR}/build"
docker_image_name="isonet"

##############################

mkdir -p "${docker_build_context_dir}"
cp "${REPO_ROOT}/requirements.txt" "${docker_build_context_dir}"

DOCKER_BUILDKIT=1 \
docker build \
    --quiet \
    --tag "${docker_image_name}" \
    --build-arg "TF_ver=${TF_ver}" \
    --file "${SELF_DIR}/Dockerfile" \
    "${docker_build_context_dir}"
