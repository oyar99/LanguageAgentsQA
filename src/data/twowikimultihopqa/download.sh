#!/bin/bash

# download the data
curl -L -o data.zip https://www.dropbox.com/s/npidmtadreo6df2/data.zip?dl=1

# unzip the data
unzip data -d data

# pretty save the data
cat "data/data/dev.json" | python -m json.tool > "dev.json"

# pretty save the data (train)
cat "data/data/train.json" | python -m json.tool > "train.json"

# remove the data
rm data.zip
rm -r data/

# generate corpus
python utils/generate_corpus.py