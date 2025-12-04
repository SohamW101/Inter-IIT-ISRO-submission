import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAM_CHECKPOINT = "sam2.1_hiera_large.pt"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

MD_MODEL_ID = "vikhyatk/moondream2"

def run_moondream(image_path, text_prompt):
    pil_img = Image.open(image_path).convert("RGB")
    W, H = pil_img.size

    md_dtype = torch.float32 if DEVICE == "cpu" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        MD_MODEL_ID,
        trust_remote_code=True,
        dtype=md_dtype,
        device_map={"": DEVICE},
    ).to(DEVICE)
    model.eval()

    result = model.detect(pil_img, text_prompt)
    if not isinstance(result, dict) or "objects" not in result:
        raise RuntimeError("Moondream output format incorrect")

    boxes = []
    for obj in result["objects"]:
        try:
            x_min = float(obj.get("x_min", obj.get("xmin", obj.get("left"))))
            y_min = float(obj.get("y_min", obj.get("ymin", obj.get("top"))))
            x_max = float(obj.get("x_max", obj.get("xmax", obj.get("right"))))
            y_max = float(obj.get("y_max", obj.get("ymax", obj.get("bottom"))))
        except:
            continue

        x1 = max(0, min(x_min * W, W - 1))
        y1 = max(0, min(y_min * H, H - 1))
        x2 = max(0, min(x_max * W, W - 1))
        y2 = max(0, min(y_max * H, H - 1))

        if x2 > x1 + 5 and y2 > y1 + 5:
            boxes.append([x1, y1, x2, y2])

    if len(boxes) == 0:
        raise RuntimeError("Moondream produced zero valid boxes")

    return pil_img, boxes

def run_sam2(pil_img, boxes):
    if not os.path.exists(SAM_CHECKPOINT):
        raise FileNotFoundError("Missing SAM checkpoint")

    sam_model = build_sam2(SAM_CONFIG, SAM_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam_model)

    predictor.set_image(np.array(pil_img))

    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(boxes, dtype=np.float32),
        multimask_output=False
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    return masks

def get_rotated_rectangles(image_path, masks, save_path=None):
    img = cv2.imread(image_path)
    rotated_rects = []

    for mask in masks:
        m = (mask * 255).astype("uint8")

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 10:
            continue

        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        rotated_rects.append(box.flatten().tolist())

        if save_path:
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    if save_path:
        cv2.imwrite(save_path, img)

    return rotated_rects

def refine_query(user_query: str) -> str:
    q = user_query.strip().lower()

    if " for " in q:
        q = q.split(" for ", 1)[1]
    else:
        pass

    if q.endswith(" in the image"):
        q = q[: -len(" in the image")]

    q = q.strip(" .,")

    return q

def normalize_obbs(obbs, img):
    W, H = img.size
    norm = []
    for box in obbs:
        norm_box = []
        for i in range(0, 8, 2):
            x = box[i]   / W
            y = box[i+1] / H
            norm_box.extend([x, y])
        norm.append(norm_box)
    return norm

def detect_objects(image_path, text_prompt):
    final_query = refine_query(text_prompt)                                                  
    pil_img, boxes = run_moondream(image_path, final_query) 
    masks = run_sam2(pil_img, boxes)
    rotated_rects = get_rotated_rectangles(image_path, masks, save_path="sam3_result1.jpg")
    final_rotated_rects = normalize_obbs(rotated_rects, pil_img)
    return final_rotated_rects

if __name__ == "__main__":
    query = "locate and return oriented bounding boxes for all the airplanes in the image"
    img_path = "/home/b23cs1037/rust1/Images_val/05896_0000.png"
    result = detect_objects(
        img_path,
        query
    )
    print("Final result:", result)
    