class FinalRun:
    def __init__(self):
        import deliverable_cap_vqa.final_single_img_cap as rgb_cap
        import deliverable_cap_vqa.final_single_img_vqa_binary_counting as rgb_vqa
        import deliverable_cap_vqa.final_single_img_vqa_semantic as rgb_sem
        import deliverable_cap_vqa.final_grey_cap as grey_cap
        import deliverable_cap_vqa.final_grey_vqa_binary_counting as grey_vqa
        import deliverable_cap_vqa.final_grey_vqa_semantic as grey_sem
        import grounding.ground as rgb_ground
        import classifier.classifier_inference as classify
        import re
        
        self.rgb_cap = rgb_cap
        self.rgb_vqa = rgb_vqa
        self.rgb_sem = rgb_sem
        self.grey_cap = grey_cap
        self.grey_vqa = grey_vqa
        self.grey_sem = grey_sem
        self.rgb_ground = rgb_ground
        self.classify = classify
        
        self.cnn_classifier = classify.load_model()
        self.rgb_cap_sem_processor, self.rgb_cap_sem_model = rgb_cap.load_model()
        self.rgb_vqa_processor, self.rgb_vqa_model = rgb_vqa.load_model()
        self.grey_cap_sem_processor, self.grey_cap_sem_model = grey_cap.load_model()
        self.grey_vqa_processor, self.grey_vqa_model = grey_vqa.load_model()
    
    def classify_type(self, image_path):
        cnn_output = self.classify.classify_image(self.cnn_classifier, image_path)
        return cnn_output
    
    def captioning(self, image_path, image_type, query_captioning):
        if image_type == "RGB":
            output_captioning = self.rgb_cap.generate_caption(image_path, query_captioning, self.rgb_cap_sem_processor, self.rgb_cap_sem_model)
            return output_captioning
        elif image_type == "Greyscale":
            output_captioning = self.grey_cap.generate_caption(image_path, query_captioning, self.grey_cap_sem_processor, self.grey_cap_sem_model)
            return output_captioning
    
    def binary_ans(self, image_path, image_type, query_binary):
        if image_type == "RGB":
            output_binary = self.rgb_vqa.answer_vqa(image_path, query_binary, self.rgb_vqa_processor, self.rgb_vqa_model)
            return output_binary
        elif image_type == "Greyscale":
            output_binary = self.grey_vqa.answer_vqa(image_path, query_binary, self.grey_vqa_processor, self.grey_vqa_model)
            return output_binary
    
    def counting_ans(self, image_path, image_type, query_counting):
        import re
        
        if image_type == "RGB":
            output_counting = self.rgb_vqa.answer_vqa(image_path, query_counting, self.rgb_vqa_processor, self.rgb_vqa_model)
        elif image_type == "Greyscale":
            output_counting = self.grey_vqa.answer_vqa(image_path, query_counting, self.grey_vqa_processor, self.grey_vqa_model)
        
        pattern = r"\d+"
        matches = re.findall(pattern, output_counting)
        count_int = int(matches[0]) if matches else 0
        return count_int
    
    def semantic_ans(self, image_path, image_type, query_semantic):
        if image_type == "RGB":
            output_semantic = self.rgb_sem.answer_vqa(image_path, query_semantic, self.rgb_cap_sem_processor, self.rgb_cap_sem_model)
            return output_semantic
        elif image_type == "Greyscale":
            output_semantic = self.grey_sem.answer_vqa(image_path, query_semantic, self.grey_cap_sem_processor, self.grey_cap_sem_model)
            return output_semantic
        
    def grounding(self, image_path, image_type, query_grounding):
        if image_type == "RGB":
            output_grounding = self.rgb_ground.detect_objects(image_path, query_grounding)
            return output_grounding