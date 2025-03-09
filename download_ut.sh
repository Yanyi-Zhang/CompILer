#!/bin/bash

# Get UT-Zappos
wget --show-progress -O local_datasets/splits.tar.gz https://www.senthilpurushwalkam.com/publication/compositional/compositional_split_natural.tar.gz
tar -zxvf local_datasets/splits.tar.gz
mv ut-zap50k/metadata_compositional-split-natural.t7 local_datasets/ut-zappos/
wget --show-progress -O local_datasets/ut-zappos/utzap.zip http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip
unzip local_datasets/ut-zappos/utzap.zip -d local_datasets/ut-zappos/
mv local_datasets/ut-zappos/ut-zap50k-images local_datasets/ut-zappos/_images/
python reorganize_utzap.py
rm local_datasets/splits.tar.gz local_datasets/ut-zappos/utzap.zip local_datasets/ut-zappos/metadata_compositional-split-natural.t7 mit-states ut-zap50k local_datasets/ut-zappos/_images