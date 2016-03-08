#!/bin/bash

set -e

git status -s | grep '??' | grep '\.dat' | awk '{ print $2 }' | xargs rm -f
