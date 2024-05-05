
from PIL import Image, ImageDraw, ImageFont
# import sys
# sys.path.insert(0, "weights/Grounded-Segment-Anything/GroundingDINO")
# sys.path.insert(0, "weights/Grounded-Segment-Anything/segment_anything")

# from groundingdino.util import box_ops
# from groundingdino.util.inference import annotate, load_image, predict

import numpy as np
import torch
import cv2

def adjust_mask(mask, adjustment_factor):
    mask = mask.astype(np.uint8)

    if adjustment_factor == 0:  # Just return the mask as is if adjustment factor is 0
        return mask

    if adjustment_factor < 0:
        mask = cv2.erode(
            mask,
            np.ones((abs(adjustment_factor), abs(adjustment_factor)), np.uint8),
            iterations=1
        )

    if adjustment_factor > 0:
        mask = cv2.dilate(
            mask,
            np.ones((adjustment_factor, adjustment_factor), np.uint8),
            iterations=1
        )

    return mask

def detect(image, text_prompt, model, processor, box_threshold = 0.3, text_threshold = 0.25):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("PROCESSOR: ", processor.__class__.__name__)
    print("MODEL: ", model.__class__.__name__)

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    print("INPUTS READY")
    with torch.no_grad():
        outputs = model(**inputs)
        outputs.logits = outputs.logits.cpu()
        outputs.pred_boxes = outputs.pred_boxes.cpu()


    print("MODEL DONE")

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )
    print("DETECTION DONE")


    print(results)
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]


    # boxes, logits, phrases = predict(
    #     model=model,
    #     image=image,
    #     caption=text_prompt,
    #     box_threshold=box_threshold,
    #     text_threshold=text_threshold
    # )

    # annotated_frame = annotate(image_source=image_src, boxes=boxes, logits=logits, phrases=phrases)
    # annotated_frame = annotated_frame[...,::-1] # BGR to RGB
    return None, boxes

def segment(image, model, processor, boxes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_rgb = image.convert("RGB")
    inputs = processor(image_rgb, input_boxes=boxes, return_tensors="pt").to(device)
    outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    scores = outputs.iou_scores

    print("SEGMENTATION OUTPUT: ", masks)

    return masks
    # sam_model.set_image(image)
    # H, W, _ = image.shape
    # boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
    # masks, _, _ = sam_model.predict_torch(
    #     point_coords = None,
    #     point_labels = None,
    #     boxes = transformed_boxes,
    #     multimask_output = False,
    #     )
    # return masks.cpu()

def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def run_grounding_sam(image_path, positive_prompt, negative_prompt, groundingdino_model, groundingdino_processor, sam_model, sam_processor, adjustment_factor):
    # image_source, image = load_image(local_image_path)
    print("IMAGE PATH: ", image_path)
    image = Image.open(image_path)
    print(image)

    annotated_frame, detected_boxes = detect(image=image, text_prompt=positive_prompt, model=groundingdino_model, processor=groundingdino_processor)
    box_tensor = detected_boxes.unsqueeze(0) #batch size of 1


    segmented_frame_masks = segment(image=image, model=sam_model, processor=sam_processor, boxes=box_tensor)

    # # Merging all positive masks
    # merged_mask = np.logical_or.reduce(segmented_frame_masks[:, 0])

    # # Annotation using merged positive mask
    # final_annotated_frame_with_mask = draw_mask(merged_mask, annotated_frame)

    # # Converting positive mask into PIL image
    # mask = (merged_mask.cpu().numpy() * 255).astype(np.uint8)  # Update mask definition

    # neg_annotated_frame_with_mask = final_annotated_frame_with_mask

    # # If negative_prompt is defined and not empty, process negative mask
    # if negative_prompt:
    #     neg_annotated_frame, neg_detected_boxes = detect(image, image_source, negative_prompt, groundingdino_model)

    #     neg_segmented_frame_masks = segment(image_source, sam_predictor, boxes=neg_detected_boxes)

    #     # Merging all negative masks
    #     merged_neg_mask = np.logical_or.reduce(neg_segmented_frame_masks[:, 0])

    #     # Annotation using merged negative mask
    #     neg_annotated_frame_with_mask = draw_mask(merged_neg_mask, neg_annotated_frame)

    #     neg_mask = (merged_neg_mask.cpu().numpy() * 255).astype(np.uint8)  # Update mask definition

    #     # Use logical operations to subtract the negative mask from the original mask
    #     mask = mask & ~neg_mask

    # erode or dilate based on adjustment_factor
    final_mask = adjust_mask(mask, adjustment_factor)

    # Update inverted mask definition
    final_inverted_mask = 255 - final_mask

    return Image.fromarray(final_annotated_frame_with_mask), Image.fromarray(neg_annotated_frame_with_mask), Image.fromarray(final_mask), Image.fromarray(final_inverted_mask)