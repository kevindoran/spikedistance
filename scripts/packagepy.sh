#!/bin/bash

# To be run from the docker container.

rm -f retinapy/dist/* && python -m build retinapy
