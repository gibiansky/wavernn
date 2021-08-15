#!/bin/sh
set -eu

echo 'Checking Python formatting...'
black --check src
echo 'Done.'
echo

echo 'Checking C++ formatting...'
clang-format --dry-run --Werror src/kernel/*.cpp
echo 'Done.'
echo

echo 'Typechecking...'
mypy src
echo 'Done.'
echo

echo 'Linting...'
pylint src
echo 'Done.'
echo

echo 'Checking C++ documentation...'
(cd src/kernel && doxygen | grep warning) || true
echo 'Done.'
echo

echo 'Linting C++...'
(cd src/kernel && clang-tidy *.cpp -- \
    -I${VIRTUAL_ENV}/lib/python3.9/site-packages/torch/include \
    -I${VIRTUAL_ENV}/lib/python3.9/site-packages/torch/include/torch/csrc/api/include \
    -I${VIRTUAL_ENV}/lib/python3.9/site-packages/torch/include/TH \
    -I${VIRTUAL_ENV}/lib/python3.9/site-packages/torch/include/THC \
    -I${VIRTUAL_ENV}/include)
echo 'Done.'
echo
