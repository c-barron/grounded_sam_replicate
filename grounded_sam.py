
import sys
# sys.path.insert(0, "weights/Grounded-Segment-Anything/GroundingDINO")
# sys.path.insert(0, "weights/Grounded-Segment-Anything/segment_anything")

# from groundingdino.util import box_ops
# from groundingdino.util.inference import annotate, load_image, predict

import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
from pathlib import Path
from collections import defaultdict
import cv2
import torch
import requests
import numpy as np
from PIL import Image
# import plotly.express as px
import matplotlib.pyplot as plt
# import plotly.graph_objects as go

## Result Utils
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))




## Plot Utils
def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

# def plot_detections(
#     image: Union[Image.Image, np.ndarray],
#     detections: List[DetectionResult],
#     save_name: Optional[str] = None
# ) -> None:
#     annotated_image = annotate(image, detections)
#     plt.imshow(annotated_image)
#     plt.axis('off')
#     if save_name:
#         plt.savefig(save_name, bbox_inches='tight')
#     plt.show()

# def random_named_css_colors(num_colors: int) -> List[str]:
#     """
#     Returns a list of randomly selected named CSS colors.

#     Args:
#     - num_colors (int): Number of random colors to generate.

#     Returns:
#     - list: List of randomly selected named CSS colors.
#     """
#     # List of named CSS colors
#     named_css_colors = [
#         'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
#         'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
#         'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
#         'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
#         'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
#         'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
#         'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
#         'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
#         'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
#         'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
#         'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
#         'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
#         'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
#         'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
#         'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
#         'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
#         'whitesmoke', 'yellow', 'yellowgreen'
#     ]

#     # Sample random named CSS colors
#     return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

# def plot_detections_plotly(
#     image: np.ndarray,
#     detections: List[DetectionResult],
#     class_colors: Optional[Dict[str, str]] = None
# ) -> None:
#     # If class_colors is not provided, generate random colors for each class
#     if class_colors is None:
#         num_detections = len(detections)
#         colors = random_named_css_colors(num_detections)
#         class_colors = {}
#         for i in range(num_detections):
#             class_colors[i] = colors[i]


#     fig = px.imshow(image)

#     # Add bounding boxes
#     shapes = []
#     annotations = []
#     for idx, detection in enumerate(detections):
#         label = detection.label
#         box = detection.box
#         score = detection.score
#         mask = detection.mask

#         polygon = mask_to_polygon(mask)

#         fig.add_trace(go.Scatter(
#             x=[point[0] for point in polygon] + [polygon[0][0]],
#             y=[point[1] for point in polygon] + [polygon[0][1]],
#             mode='lines',
#             line=dict(color=class_colors[idx], width=2),
#             fill='toself',
#             name=f"{label}: {score:.2f}"
#         ))

#         xmin, ymin, xmax, ymax = box.xyxy
#         shape = [
#             dict(
#                 type="rect",
#                 xref="x", yref="y",
#                 x0=xmin, y0=ymin,
#                 x1=xmax, y1=ymax,
#                 line=dict(color=class_colors[idx])
#             )
#         ]
#         annotation = [
#             dict(
#                 x=(xmin+xmax) // 2, y=(ymin+ymax) // 2,
#                 xref="x", yref="y",
#                 text=f"{label}: {score:.2f}",
#             )
#         ]

#         shapes.append(shape)
#         annotations.append(annotation)

#     # Update layout
#     button_shapes = [dict(label="None",method="relayout",args=["shapes", []])]
#     button_shapes = button_shapes + [
#         dict(label=f"Detection {idx+1}",method="relayout",args=["shapes", shape]) for idx, shape in enumerate(shapes)
#     ]
#     button_shapes = button_shapes + [dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])]

#     fig.update_layout(
#         xaxis=dict(visible=False),
#         yaxis=dict(visible=False),
#         # margin=dict(l=0, r=0, t=0, b=0),
#         showlegend=True,
#         updatemenus=[
#             dict(
#                 type="buttons",
#                 direction="up",
#                 buttons=button_shapes
#             )
#         ],
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1
#         )
#     )

#     # Show plot
#     fig.show()



## Utils
def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: Union[str, Path]) -> Image.Image:
    print("Loading Image")
    if isinstance(image_str, Path):
        image_str = str(image_str)
    try:
        if image_str.startswith("http"):
            image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_str).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


## Grounded SAM

def detect(
    object_detector,
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Use Zero Shot Detector to detect a set of labels in an image in a zero-shot fashion.
    """
    # Periods must be used at the end of a label for grounding dino i think?
    # labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results


def segment(
    segmentator,
    processor,
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"


    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    detector,
    segmentator,
    processor,
    image: Union[Image.Image, str, Path],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult], np.ndarray]:
    print("Loading Image")
    
    if isinstance(image, (str, Path)):
        image = load_image(image)

    if image is None:
        raise ValueError("Failed to load the image.")
        
    print("Running Detection")
    print("Prompts: ", labels)
    print("Threshold: ", threshold)
    print(image.size)
    
    detections = detect(detector, image, labels, threshold)
    print(detections)

    if len(detections) == 0: 
        print("No detections found, returning")
        return np.array(image), detections, None

     # Annotate the original image with the detections
    annotated_image_array = annotate(image, detections)


    print("Running Segmentation")
    detections = segment(segmentator, processor, image, detections, polygon_refinement)

    return np.array(image), detections, annotated_image_array

## Mask Merging Utils
def show_masked_image(image: np.ndarray, mask: np.ndarray) -> None:
    masked_image = image.copy()
    masked_image[mask == 0] = 0

    plt.imshow(masked_image)
    plt.axis('off')
    plt.show()

def merge_masks(mask1, mask2):
    # Ensure both masks are of type uint8
    if mask1.dtype != np.uint8:
        mask1 = mask1.astype(np.uint8)
    if mask2.dtype != np.uint8:
        mask2 = mask2.astype(np.uint8)

    # Use np.maximum to merge the masks
    merged_mask = np.maximum(mask1, mask2)
    return merged_mask

def merge_masks_by_class(detection_results: List[DetectionResult]) -> List[DetectionResult]:
    merged_masks = defaultdict(lambda: None)
    total_scores = defaultdict(float)
    counts = defaultdict(int)
    merged_boxes = {}

    for result in detection_results:
        label = result.label
        if merged_masks[label] is None:
            merged_masks[label] = result.mask
            merged_boxes[label] = result.box
        else:
            if result.mask is not None:
                merged_masks[label] = merge_masks(merged_masks[label], result.mask)

            box = merged_boxes[label]
            merged_boxes[label] = BoundingBox(
                xmin=min(box.xmin, result.box.xmin),
                ymin=min(box.ymin, result.box.ymin),
                xmax=max(box.xmax, result.box.xmax),
                ymax=max(box.ymax, result.box.ymax)
            )

        total_scores[label] += result.score
        counts[label] += 1

    merged_detections = [
        DetectionResult(
            score=total_scores[label] / counts[label],
            label=label,
            box=merged_boxes[label],
            mask=mask
        )
        for label, mask in merged_masks.items()
    ]

    return merged_detections


def numpy_to_image(np_array: np.ndarray) -> Image:
    print(f"Array min: {np_array.min()}, Array max: {np_array.max()}")
    # Ensure the array is of type uint8
    if np_array.dtype != np.uint8:
        print(f"Converting array type from {np_array.dtype} to uint8")
        np_array = np_array.astype(np.uint8)
        
    return Image.fromarray(np_array)


## Inference

def run_grounding_sam(image, prompts, detector, segmentator, processor, threshold) -> Dict[str, Image.Image]:
    print(image)
    image_array, detections, annotation_image_array = grounded_segmentation(
        detector,
        segmentator,
        processor,
        image=image,
        labels=prompts,
        threshold=threshold,
        polygon_refinement=True,
    )

    merged_detections = merge_masks_by_class(detections)
    output_dict = {result.label: numpy_to_image(result.mask) for result in merged_detections if result.mask is not None}
    if annotation_image_array is not None:
        annotated_image = numpy_to_image(annotation_image_array)
        output_dict['annotated_image'] = annotated_image

    return output_dict

