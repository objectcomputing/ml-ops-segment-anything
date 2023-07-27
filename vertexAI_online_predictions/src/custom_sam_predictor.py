import torch
from typing import Dict, List
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import base64
import numpy as np
import cv2
import logging

# custom prediction routine
class CustomSamPredictor(Predictor):

    def __init__(self):
        super().__init__()
        #self.predictor = None #uncomment to mask with prompts
        self.mask_generator = None #uncomment to mask without prompts
        self.sam = None
        self.model_type = "vit_b"
        self.device = "cuda"
        self.mask_with_prompts = True
        logging.basicConfig(level=logging.INFO)
    
    # Load the model
    def load(self, artifacts_uri: str):
        """Loads the model artifacts."""
        prediction_utils.download_model_artifacts(artifacts_uri)
        self.sam = sam_model_registry[self.model_type](checkpoint="sam_vit_b_01ec64.pth")
        self.sam.to(device=self.device)
        
    
    # preprocess the data, the image received as base64 is preprocessed
    # The prompt inputs are sent to predict method
    def preprocess(self, prediction_input: Dict) -> Dict:
        print("************** IN PRE PROCESS **********************")
        prediction_input = prediction_input["instances"][0]
        image = prediction_input["image"]
        jpg_original = base64.b64decode(image)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=1)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if len(prediction_input) > 2: # Masking with prompts requires 3 inputs (file_path, image, input_points, input_label)
            """ Masking with prompts """
            print("PREDICTING WITH PROMPTS")
            self.mask_with_prompts = True
            self.predictor = SamPredictor(self.sam)
            self.predictor.set_image(image) 
            del prediction_input["image"] 
        else: # Masking without prompts requires only image input
            """ Masking without prompts / automatic masking """
            self.mask_with_prompts = False
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
            prediction_input["image_cvtColor"] = image
        
        return prediction_input

    # Get the predictions from the loaded model
    @torch.inference_mode()
    def predict(self, prediction_input: Dict) -> List:
        """Performs prediction."""
        print(self.mask_with_prompts)
        if self.mask_with_prompts:
            print("PREDICTING WITH PROMPTS")
            """ Masking with prompts """
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array(prediction_input["input_point"]).reshape(1,2),
                point_labels=np.array(prediction_input["input_label"]),
                multimask_output=True,
            )
            
            return list((prediction_input["file_path"], masks, scores, logits))
        else:
            """ Masking without prompts / automatic masking """
            masks = self.mask_generator.generate(prediction_input["image_cvtColor"])
            
            return list((prediction_input["file_path"], prediction_input["image"], masks))

        
        
    
    # Returns the predictions as a dictionary
    def postprocess(self, prediction_results: List) -> Dict:
        """Postprocessing / construct response structure."""
        prediction={}
        if self.mask_with_prompts:
            """ Prediction response / Masking with prompts """
            prediction["file_path"] = prediction_results[0]
            prediction["masks"] = prediction_results[1].tolist()
            prediction["scores"] = prediction_results[2].tolist()
            prediction["logits"] = prediction_results[3].tolist()
        else:
            """ Prediction response / Masking without prompts / automatic masking """
            prediction["file_path"] = prediction_results[0]
            prediction["image"] = prediction_results[1]
            prediction["masks"] = {}
            for idx, mask in enumerate(prediction_results[2]):
                prediction["masks"][f'mask_{idx}'] = m.tolist() 
            
        return prediction
