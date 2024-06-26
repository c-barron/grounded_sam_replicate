import os
import sys
import subprocess
import torch

from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
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
from grounded_sam import run_grounding_sam
import uuid
# from hf_path_exports import cache_config_file, cache_file


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipelines...x")

        def load_detector(detector_id: str):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
            return object_detector
        
        
        def load_segmentator(segmenter_id: str):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
            processor = AutoProcessor.from_pretrained(segmenter_id)
            return segmentator, processor

        # detector_id = "./models/grounding-dino-tiny"
        # detector_id = "./models/grounding-dino-base"
        detector_id = "./models/owlv2-base-patch16-ensemble"
        segmenter_id = "./models/sam-vit-base"
        
        object_detector = load_detector(detector_id)
        segmentator, processor = load_segmentator(segmenter_id)

        self.object_detector = object_detector
        self.segmentator = segmentator
        self.processor = processor

        
        print("Pipelines loaded...x")
    
    
    @torch.inference_mode()
    def predict(
            self,
            image: Path = Input(
                description="Image",
                default="https://st.mngbcn.com/rcs/pics/static/T5/fotos/outfit/S20/57034757_56-99999999_01.jpg",
            ),
            prompts: List[str] = Input(
                description="List of mask prompts. Each should end with a period?",
                default=["face."],
            ),
            threshold: float = Input(
                description="Cutof for object detection",
                default=0.30, #S et to 0.30 for dino, 0.10 for owl
            )
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""
        predict_id = str(uuid.uuid4())

        print(f"Running prediction: {predict_id}...")

        outputs = run_grounding_sam(image, prompts, self.object_detector, self.segmentator, self.processor, threshold)

        print("Done!")

        output_dir = "/tmp/" + predict_id
        os.makedirs(output_dir, exist_ok=True)
    
        # Create a black image for fallback
        fallback_image = Image.new('RGB', (10, 10), color='black')

        # Iterate over the prompts and yield the corresponding mask or fallback image
        for prompt in prompts:
            image = outputs.get(prompt, fallback_image)
            random_filename = os.path.join(output_dir, f"{prompt.replace(' ', '_')}.jpg")
            if image.mode != 'RGB':
                print("Converting image to RGB")
                image = image.convert('RGB')
            image.save(random_filename)
            yield Path(random_filename)  # Yield the path to the saved image

        # Extract the annotated image if it exists and is a PIL Image
        annotated_image = outputs.get('annotated_image', None)
        if isinstance(annotated_image, Image.Image):
            annotated_image_path = os.path.join(output_dir, "annotated_image.jpg")
            annotated_image.save(annotated_image_path)
            yield Path(annotated_image_path)  # Yield the path to the saved annotated image