# tools.py

import os
from typing import Type # Added for type hinting args_schema

# Updated import for BaseTool
from langchain_core.tools import BaseTool
# Import BaseModel and Field for defining input schema
from pydantic import BaseModel, Field

from PIL import Image
import numpy as np
import nibabel as nib
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    DetrImageProcessor, DetrForObjectDetection
)


def load_any_image(path: str) -> Image.Image:
    """
    Load either a 2D image (PNG/JPG) via PIL,
    or a 3D NIfTI (.nii/.nii.gz) via nibabel → extract middle slice.
    Returns a PIL Image in RGB.
    """
    # Check if path exists before proceeding
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path does not exist: {path}")

    ext = path.lower().split('.')[-1]
    # Handle potential double extensions like .nii.gz
    if path.lower().endswith('.nii.gz'):
        ext = 'gz'
    elif path.lower().endswith('.nii'):
        ext = 'nii'

    if ext in ('nii', 'gz'):
        try:
            # load volume
            img_obj = nib.load(path)
            arr = img_obj.get_fdata()
            if arr.ndim < 3:
                 raise ValueError(f"Expected NIfTI to have at least 3 dimensions, but got {arr.ndim}")
            # pick middle slice on last axis (usually Z)
            z = arr.shape[-1] // 2
            slice_2d = arr[..., z] # Use ellipsis for flexibility with >3D arrays
            # Handle potential NaN/inf values before normalization
            slice_2d = np.nan_to_num(slice_2d)
            # normalize to 0-255
            slice_min = slice_2d.min()
            slice_ptp = slice_2d.ptp() # peak-to-peak (max-min)
            if slice_ptp == 0: # Avoid division by zero for uniform slices
                 norm = np.zeros_like(slice_2d, dtype=np.uint8)
            else:
                 norm = ((slice_2d - slice_min) / slice_ptp * 255).astype(np.uint8)
            pil_img = Image.fromarray(norm)
            pil_img = pil_img.convert("RGB")
            return pil_img
        except Exception as e:
            raise RuntimeError(f"Failed to load or process NIfTI file '{path}': {e}") from e
    elif ext in ('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'):
         try:
             # delegate to PIL
             return Image.open(path).convert("RGB")
         except Exception as e:
             raise RuntimeError(f"Failed to load image file '{path}' using PIL: {e}") from e
    else:
        raise ValueError(f"Unsupported file extension for path: {path}. Supports NIfTI (.nii, .nii.gz) and standard images (PNG, JPG, etc.).")


# --- Define Input Schema for tools that take an image path ---
class ImagePathInput(BaseModel):
    """Input schema for tools operating on a single image path."""
    img_path: str = Field(description="The file path to the local image (PNG/JPG) or NIfTI (.nii, .nii.gz) file.")


# --- Image Caption Tool ---
class ImageCaptionTool(BaseTool):
    # Added type hints for Pydantic v2 compatibility
    name: str = "Image Captioner"
    description: str = (
        "Generates a descriptive caption for the content of an image specified by its file path. "
        "Input MUST be the local file path to the image (e.g., /path/to/image.png or /path/to/brain.nii.gz). "
        "Handles standard image formats (PNG/JPG) and NIfTI (.nii, .nii.gz), using the middle slice for NIfTI."
    )
    # Added explicit input schema
    args_schema: Type[ImagePathInput] = ImagePathInput

    def _run(self, img_path: str) -> str:
        """Use the tool synchronously."""
        try:
            image = load_any_image(img_path)

            # Consider initializing models outside _run if possible for efficiency,
            # though this keeps the tool self-contained.
            model_name = "Salesforce/blip-image-captioning-large"
            # Let transformers handle device placement or specify explicitly (e.g., "cuda" if available)
            device = "cpu"

            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

            # Use text=None for unconditional captioning if desired, or provide prompt
            inputs = processor(images=image, return_tensors='pt').to(device)
            # Increased max_new_tokens for potentially longer captions
            output = model.generate(**inputs, max_new_tokens=50)

            caption = processor.decode(output[0], skip_special_tokens=True)
            # Simple post-processing
            caption = caption.strip()
            # Add prefix for clarity
            return f"Caption: {caption}"

        except FileNotFoundError as e:
             return f"Error: {e}"
        except Exception as e:
            # Catch other potential errors during loading or model inference
            return f"Error processing image for captioning ('{img_path}'): {e}"

    # Added return type hint -> str, even though it raises error, for signature consistency
    async def _arun(self, img_path: str) -> str:
        """Async execution not supported."""
        raise NotImplementedError("Image Captioner tool does not support async execution.")


# --- Object Detection Tool ---
class ObjectDetectionTool(BaseTool):
    # Added type hints for Pydantic v2 compatibility
    name: str = "Object Detector"
    description: str = (
        "Detects objects within an image specified by its file path. "
        "Input MUST be the local file path to the image (e.g., /path/to/image.jpg or /path/to/volume.nii.gz). "
        "Returns a list of detected objects, their bounding boxes, and confidence scores. "
        "Handles standard image formats (PNG/JPG) and NIfTI (.nii, .nii.gz), using the middle slice for NIfTI."
    )
    # Added explicit input schema
    args_schema: Type[ImagePathInput] = ImagePathInput

    def _run(self, img_path: str) -> str:
        """Use the tool synchronously."""
        try:
            image = load_any_image(img_path)

            # Consider initializing models outside _run if possible
            model_name = "facebook/detr-resnet-50"
            processor = DetrImageProcessor.from_pretrained(model_name)
            model = DetrForObjectDetection.from_pretrained(model_name)
            # Move model to GPU if available
            device = "cpu"
            model.to(device)

            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad(): # Inference doesn't need gradient calculation
                 outputs = model(**inputs)

            # Convert outputs to CPU for post-processing if they were on GPU
            outputs.logits = outputs.logits.cpu()
            outputs.pred_boxes = outputs.pred_boxes.cpu()

            target_sizes = torch.tensor([image.size[::-1]]) # (height, width)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.9 # High threshold to reduce noise
            )[0]

            if len(results["scores"]) == 0:
                return "No objects detected with high confidence (threshold > 0.9)."

            detections = "Detected objects:\n"
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = map(int, box.tolist())
                cls = model.config.id2label[int(label)]
                # Add detection line, ensuring newline character is correctly placed
                detections += f"- {cls} (Confidence: {float(score):.2f}) at box [{x1},{y1},{x2},{y2}]\n"

            return detections.strip() # Remove trailing newline

        except FileNotFoundError as e:
             return f"Error: {e}"
        except Exception as e:
            # Catch other potential errors during loading or model inference
            return f"Error processing image for object detection ('{img_path}'): {e}"

    # Added return type hint -> str for signature consistency
    async def _arun(self, img_path: str) -> str:
        """Async execution not supported."""
        raise NotImplementedError("Object Detector tool does not support async execution.")