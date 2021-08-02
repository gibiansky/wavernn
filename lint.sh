#!/bin/sh
echo 'Typechecking...'
mypy src

echo 'Linting...'
pylint src
