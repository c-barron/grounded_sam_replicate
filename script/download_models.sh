#!/bin/bash

# Create a directory for the models
mkdir -p models
cd models

# Install Git Large File Storage (LFS)
git-lfs install

# Clone the specified models from Hugging Face
git clone https://huggingface.co/google/owlv2-base-patch16-ensemble
git clone https://huggingface.co/facebook/sam-vit-base

echo "Models have been successfully downloaded."