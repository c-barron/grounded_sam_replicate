import os
import sys
import subprocess
import torch

from pydantic import ValidationError, parse_raw_as
import numpy as np
from typing import Union

from transformers import AutoModelForMaskGeneration, AutoProcessor
from PIL import Image

#install GroundingDINO and segment_anything
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.7'
os.environ['AM_I_DOCKER'] = 'true'
os.environ['BUILD_WITH_CUDA'] = 'true'

# env_vars = os.environ.copy()
# HOME = os.getcwd()
# sys.path.insert(0, "weights")
# sys.path.insert(0, "weights/GroundingDINO")
# sys.path.insert(0, "weights/segment-anything")
# os.chdir("/src/weights/GroundingDINO")
# subprocess.call([sys.executable, '-m', 'pip', 'install', '-e', '.'], env=env_vars)
# os.chdir("/src/weights/segment-anything")
# subprocess.call([sys.executable, '-m', 'pip', 'install', '-e', '.'], env=env_vars)
# os.chdir(HOME)

from cog import BasePredictor, Input, Path, BaseModel
from typing import Iterator, List, Optional
# from groundingdino.util.slconfig import SLConfig
# from groundingdino.models import build_model
# from groundingdino.util.utils import clean_state_dict
# from segment_anything import build_sam, SamPredictor
from grounded_sam import run_sam_only, PromptList
import uuid
# from hf_path_exports import cache_config_file, cache_file

class Mask(BaseModel):
    mask: Path
    label: str

class Output(BaseModel):
    masks: List[Mask]

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipelines...x")
        
        def load_segmentator(segmenter_id: str):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
            processor = AutoProcessor.from_pretrained(segmenter_id)
            return segmentator, processor

        segmenter_id = "./models/sam-vit-base"
        
        segmentator, processor = load_segmentator(segmenter_id)

        self.segmentator = segmentator
        self.processor = processor

        
        print("Pipelines loaded...x")
    
# label: str
#     is_positive: bool
#     xmin: float
#     xmax: float
#     ymin: float
#     ymax: float

    
    @torch.inference_mode()
    def predict(
            self,
            image: Path = Input(
            description="Image",
            default="https://st.mngbcn.com/rcs/pics/static/T5/fotos/outfit/S20/57034757_56-99999999_01.jpg",
            ),
            prompts_json: str = Input(
                description="JSON string of prompt object type (see documentation)",
                default='{"prompts": [{"label": "face", "xmin": 0.1, "xmax": 0.9, "ymin": 0.1, "ymax": 0.9, "is_positive": true}]}',
            ),
    
    ) -> Output:
        """Run a single prediction on the model"""
        predict_id = str(uuid.uuid4())

        print(f"Running prediction: {predict_id}...")
        
        # Try to parse prompts as a PromptList
        try:
            prompts_list = parse_raw_as(PromptList, prompts_json)        
        except ValidationError as e:
            print(f"Error parsing prompts: {e}")
            raise e

        # Access prompts from the parsed object
        prompts = prompts_list.prompts
        outputs = run_sam_only(image, prompts, self.segmentator, self.processor)
    
        print("Done!")

        output_dir = "/tmp/" + predict_id
        os.makedirs(output_dir, exist_ok=True)
    
        # Create a black image for fallback
        # fallback_image = Image.new('RGB', (10, 10), color='black')

        mask_dicts = []

        # Iterate over the prompts and yield the corresponding mask or fallback image
        for prompt in prompts:
            label = prompt.label
            image = outputs.get(label, None)
            if image is None:
                print(f"Could not find mask for label: {label}")
                continue

            random_filename = os.path.join(output_dir, f"{label.replace(' ', '_')}.jpg")
            if image.mode != 'RGB':
                print("Converting image to RGB")
                image = image.convert('RGB')
            image.save(random_filename)


            mask_dicts.append(Mask(mask=Path(random_filename), label=label))

        print(mask_dicts)

        return Output(masks=mask_dicts)