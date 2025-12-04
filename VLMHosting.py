"""
grounding
input:
{
    "image_url": "https://example.com/image.jpg",
    "query": "What objects are present in the image?"
    image_id: "optional_image_id_123"
}
output:
{
    "response": {
        int1: {
            object_id: "id",
            obbox_coords: [x1, y1, x2, y2, x3, y3, x4, y4],
            },
        int2: {
            object_id: "id",
            obbox_coords: [x1, y1, x2, y2, x3, y3, x4, y4],
            },...
        },
    "image_id": "optional_image_id_123"
}
"""
from re import I
import torch
import main_class
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    TextStreamer)
from flask import Flask
from flask import request, jsonify
import requests
from io import BytesIO
from logging import (
    basicConfig,
    getLogger,
    INFO,
    DEBUG,
    ERROR
)
basicConfig(
    level=INFO,
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="VLMHosting.log",
    force=True
)
logger = getLogger(__name__)
app = Flask(__name__)
caller = main_class.FinalRun()

def classify_image(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to load image from URL {url}: {e}")
        return "Unknown"

    image_type = caller.classify_type(url)
    logger.info(f"Classified image type for URL {url}: {image_type}")
    return image_type

def grounding(url, image_type, query=None):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to load image from URL {url}: {e}")
        return query, []

    referring_sentence = query or "What objects are present in the image?"

    det = caller.grounding(url, image_type, referring_sentence)
    logger.info(f"Detections: {det}")

    # Always return the referring sentence and a list of detected objects (may be empty)
    return referring_sentence, det

def captioning(url, query, image_type):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to load image from URL {url}: {e}")
        return query or "Describe the image.", ""

    generated = caller.captioning(url, image_type, query)
    logger.info("Captioning result: %s", generated)

    return query, generated

def vqa(url, query_type, query, image_type):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to load image from URL {url}: {e}")
        return query or "Answer the question about the image.", ""

    if query_type == "binary":
        generated = caller.binary_ans(url, image_type, query)
    elif query_type == "numeric":
        generated = caller.counting_ans(url, image_type, query)
    elif query_type == "semantic":
        generated = caller.semantic_ans(url, image_type, query)
    else:
        generated = "Invalid query type."

    logger.info("VQA result: %s", generated)

    return query, generated


@app.route("/")
def start():
    logger.warning("request came.....")
    return jsonify("VLM Hosting Service is running.")


@app.route("/VLM/classify", methods=["POST"])
def classify_service():
    data = request.json
    image_url = data.get("image_url")
    image_id = data.get("image_id", "")
    image_type = classify_image(image_url)
    response = {
        "image_type": image_type,
        "image_id": image_id
    }
    return jsonify(response)


@app.route("/VLM/captioning", methods=["POST"])
def captioning_service():
    data = request.json
    image_url = data.get("image_url")
    image_id = data.get("image_id", "")
    image_type = data.get("image_type", "RGB")
    query = data.get("instructions", "Describe the image.")
    try:
        _, caption = captioning(image_url, query, image_type)
        response = {
            "response": caption,
            "image_id": image_id
        }
    except Exception as e:
        logger.error(f"Failed to caption image from URL {image_url}: {e}")
        return jsonify("Failed to caption image."), 400

    return jsonify(response)


@app.route("/VLM/grounding", methods=["POST"])
def grounding_service():
    data = request.json
    image_url = data.get("image_url")
    query = data.get("query")
    image_id = data.get("image_id", "")
    image_type = data.get("image_type", "RGB")
    query, detections = grounding(image_url, image_type, query)
    if not detections:
        logger.warning(f"No detections found for image URL {image_url} with query '{query}'")
        detections = []
    logger.info(f"Grounding detections for image URL {image_url}: {detections}")
    response_items = []
    for i, obj in enumerate(detections):
        response_items.append(
            {
                "object-id": str(i),
                "obbox": obj
            }
        )

    response = {
        "response": response_items,
        "image_id": image_id
    }
    return jsonify(response)


@app.route("/VLM/vqa", methods=["POST"])
def vqa_service():
    data = request.json or {}
    image_url = data.get("image_url")
    attribute_queries = data.get("attribute_query", {})
    image_id = data.get("image_id", "")
    image_type = data.get("image_type", "RGB")

    if not image_url:
        return jsonify({"error": "image_url is required"}), 400

    response_items = {}
    for query_type, query_content in attribute_queries.items():
        instruction = (query_content or {}).get("instruction", "")
        if not instruction:
            response_items[query_type] = "No instruction provided."
            continue
        try:
            _, answer = vqa(image_url, query_type, instruction, image_type)
            response_items[query_type] = answer
        except Exception as e:
            logger.error(f"Failed to process {query_type} query for image URL {image_url}: {e}")
            response_items[query_type] = "Failed to get answer."

    response = {
        "response": response_items,
        "image_id": image_id
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
