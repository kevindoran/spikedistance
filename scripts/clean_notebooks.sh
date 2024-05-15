#!/bin/bash

python3 -m nbconvert --clear-output --inplace *.ipynb ./notebooks/**/*.ipynb
