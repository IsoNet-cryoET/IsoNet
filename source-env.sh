#!/bin/bash

###############
# THIS FILE CAN BE SOURCED INTO YOUR SHELL
# TO SET UP ENVIRONMENT WHERE IsoNet CAN BE EXECUTED
###############

_ISONET_REPO_ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

isonet_PYTHONPATH="$(dirname "${_ISONET_REPO_ROOT}")"
if ! echo "${PYTHONPATH}" | grep -- "${isonet_PYTHONPATH}" >/dev/null; then
    export PYTHONPATH="${isonet_PYTHONPATH}:${PYTHONPATH}"
    echo "| ENV: PYTHONPATH='${PYTHONPATH}'"
fi

isonet_PATH="${_ISONET_REPO_ROOT}/bin"
if ! echo "${PATH}" | grep -- "${isonet_PATH}" >/dev/null; then
    export PATH="${isonet_PATH}:${PATH}"
    echo "| ENV: PATH='${PATH}'"
fi
