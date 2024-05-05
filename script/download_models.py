#!/usr/bin/env python3

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers import SamModel, SamProcessor

model = SamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def download_dino(model_path, processor_path):
    # Define the model repository ID
    model_id = "IDEA-Research/grounding-dino-base"
    # Define the path to save the model
    dino_path = "grounding-dino-base"

    # Download the model from Hugging Face and save it locally
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    dino_model.save_pretrained(model_path + dino_path)

    dino_processor = AutoProcessor.from_pretrained(model_id)
    dino_processor.save_pretrained(processor_path + dino_path)

def download_sam(model_path, processor_path):
    # Download SAM Model
    model_id = "facebook/sam-vit-base"
    sam_path = "sam-vit-base"

    model = SamModel.from_pretrained(model_id)
    model.save_pretrained(model_path + sam_path)

    processor = SamProcessor.from_pretrained(model_id)
    processor.save_pretrained(processor_path + sam_path)
    


def main():
    
    model_path = "./model/"
    processor_path = "./processor/"

    download_dino(model_path, processor_path)
    download_sam(model_path, processor_path)
    
    
    


if __name__ == "__main__":
    main()
