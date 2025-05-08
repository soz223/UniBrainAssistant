import os
from PIL import Image, ImageDraw, ImageFont

def _fake_image(out_path: str, text: str) -> str:
    """
    Generate a placeholder PNG with the given text.
    """
    img = Image.new("RGB", (400, 200), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Use a default font
    draw.text((20, 80), text, fill=(0, 0, 0))
    img.save(out_path)
    return out_path

def run_pipeline(input_path: str, work_dir: str):
    """
    Dummy UniBrain pipeline.
    For each step, create a PNG and return a list of cards.
    """
    steps = ["Extraction", "Registration", "Segmentation", "Parcellation", "Classification"]
    cards = []

    for step in steps:
        filename = f"{step.lower()}.png"
        out_path = os.path.join(work_dir, filename)
        _fake_image(out_path, f"{step} (demo)")
        cards.append({
            "step": step,
            "image_path": out_path,
            "explanation": f"{step} completed (demo)."
        })

    return cards
