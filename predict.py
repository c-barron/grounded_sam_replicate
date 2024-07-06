import os
import sys
import subprocess
import torch

from transformers import pipeline
from PIL import Image

os.environ['CUDA_HOME'] = '/usr/local/cuda-11.7'
os.environ['AM_I_DOCKER'] = 'true'
os.environ['BUILD_WITH_CUDA'] = 'true'

from cog import BasePredictor, Input, Path, BaseModel
from typing import List

from grounded_sam import DetectionResult, detection_only
import uuid


class Output(BaseModel):
    label: str
    score: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float



class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipelines...x")

        def load_detector(detector_id: str):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
            return object_detector
        
        
        # def load_segmentator(segmenter_id: str):
        #     device = "cuda" if torch.cuda.is_available() else "cpu"
        #     segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
        #     processor = AutoProcessor.from_pretrained(segmenter_id)
        #     return segmentator, processor

        # detector_id = "./models/grounding-dino-tiny"
        # detector_id = "./models/grounding-dino-base"
        detector_id = "./models/owlv2-base-patch16-ensemble"
        # segmenter_id = "./models/sam-vit-base"
        
        object_detector = load_detector(detector_id)
        # segmentator, processor = load_segmentator(segmenter_id)

        self.object_detector = object_detector
        # self.segmentator = segmentator
        # self.processor = processor

        
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
                default=["face.", "shirt."],
            ),
            threshold: float = Input(
                description="Cutof for object detection",
                default=0.15, #S et to 0.30 for dino, 0.10 for owl
            )
    ) -> List[Output]:
        """Run a single prediction on the model"""
        predict_id = str(uuid.uuid4())

        print(f"Running prediction: {predict_id}...")

        detections, image_resolution = detection_only(self.object_detector, image, prompts, threshold)
        print("DETECTIONS: ", detections)
        

        outputs = []
        #Iterate 
        for detection in detections:
            output = Output(
                label=detection.label,
                score=detection.score,
                # Normalize the coordinates
                xmin=detection.box.xmin / image_resolution[0],
                xmax=detection.box.xmax / image_resolution[0],
                ymin=detection.box.ymin / image_resolution[1], 
                ymax=detection.box.ymax / image_resolution[1],
            )
            outputs.append(output)
        print("OUTPUTS: ", outputs)
        return outputs

