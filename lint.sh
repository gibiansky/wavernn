#!/bin/sh
set -eu

echo 'Checking Python formatting...'
black --check src

echo 'Checking C++ formatting...'
clang-format --dry-run --Werror src/kernel/*.cpp

echo 'Typechecking...'
mypy src

echo 'Linting...'
pylint src

echo 'Generating C++ documentation...'
(cd src/kernel && doxygen | grep warning)
