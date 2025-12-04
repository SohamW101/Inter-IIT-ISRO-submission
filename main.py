import deliverable_cap_vqa.final_single_img_cap as rgb_cap
import deliverable_cap_vqa.final_single_img_vqa_binary_counting as rgb_vqa
import deliverable_cap_vqa.final_single_img_vqa_semantic as rgb_sem
import deliverable_cap_vqa.final_grey_cap as grey_cap
import deliverable_cap_vqa.final_grey_vqa_binary_counting as grey_vqa
import deliverable_cap_vqa.final_grey_vqa_semantic as grey_sem
import grounding.ground as rgb_ground
import classifier.classifier_inference as classify

import re

## load all model instances
cnn_classifier = classify.load_model()
rgb_cap_sem_processor, rgb_cap_sem_model = rgb_cap.load_model()
rgb_vqa_processor, rgb_vqa_model = rgb_vqa.load_model()
grey_cap_sem_processor, grey_cap_sem_model = grey_cap.load_model()
grey_vqa_processor, grey_vqa_model = grey_vqa.load_model()


# classifier model
def classify_type(image_path):
    cnn_output = classify.classify_image(cnn_classifier, image_path)
    return cnn_output
    

# captioning
def captioning(image_path, image_type, query_captioning = "Describe this image in 50 words."):
    if image_type == "RGB":
        output_captioning = rgb_cap.generate_caption(image_path, query_captioning, rgb_cap_sem_processor, rgb_cap_sem_model)
        return output_captioning
    elif image_type == "Greyscale":
        output_captioning = grey_cap.generate_caption(image_path, query_captioning, grey_cap_sem_processor, grey_cap_sem_model)
        return output_captioning


# grounding
def grounding(image_path, image_type, query):
    if image_type == "RGB":
        output_gorunding = rgb_ground.detect_objects(image_path, query)
        return output_gorunding


# VQA
def binary_ans(image_path, image_type, query_binary = "Answer in yes or no only."):
    if image_type == "RGB":
        output_binary = rgb_vqa.answer_vqa(image_path, query_binary, rgb_vqa_processor, rgb_vqa_model)
        return output_binary
    elif image_type == "Greyscale":
        output_binary = grey_vqa.answer_vqa(image_path, query_binary, grey_vqa_processor, grey_vqa_model)
        return output_binary
        

def counting_ans(image_path, image_type, query_counting):
    if image_type == "RGB":
        output_counting = rgb_vqa.answer_vqa(image_path, query_counting, rgb_vqa_processor, rgb_vqa_model)
    elif image_type == "Greyscale":
        output_counting = grey_vqa.answer_vqa(image_path, query_counting, grey_vqa_processor, grey_vqa_model)
    
    pattern = r"\d+"
    matches = re.findall(pattern, output_counting)
    count_int = int(matches[0]) if matches else 0
    return count_int


def semantic_ans(image_path, image_type, query_semantic):
    if image_type == "RGB":
        output_semantic = rgb_sem.answer_vqa(image_path, query_semantic, rgb_cap_sem_processor, rgb_cap_sem_model)
        return output_semantic
    elif image_type == "Greyscale":
        output_semantic = grey_sem.answer_vqa(image_path, query_semantic, grey_cap_sem_processor, grey_cap_sem_model)
        return output_semantic
    
    
def main():
    image_path = "/home/b23cs1037/rust1/Images_val/06406_0000.png"
    image_type = classify_type(image_path)
    print(image_type)
    
    caption = captioning(image_path, image_type, "Describe the image in 50 words.")
    print(caption)
    
    binary = binary_ans(image_path, image_type, "Answer in yes or no. Is there a flyover present in the image?")
    print(binary)
    
    count = counting_ans(image_path, image_type, "Count the number of vehicles")
    print(count)
    
    semantic = semantic_ans(image_path, image_type, "Where is the truck located?")
    print(semantic)
    
    ground = grounding(image_path, image_type, "locate and return the bounding boxes of the truck below the flyover in the image")                  
    print(ground)

if __name__ == "__main__":
    main()
