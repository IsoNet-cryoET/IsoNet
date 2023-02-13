#!/bin/bash

# This script can be used to run interactive Docker session of IsoNet.

set -Eeo pipefail

##############################

SELF_DIR="$(dirname "$(realpath "${0}")")"
REPO_ROOT="$(realpath "${SELF_DIR}/..")"

##############################

"${SELF_DIR}/build.sh"

docker run \
    -it \
    --rm \
    --gpus all \
    --user isonet \
    --volume "${REPO_ROOT}:${REPO_ROOT}" \
    --workdir "${REPO_ROOT}" \
    isonet
