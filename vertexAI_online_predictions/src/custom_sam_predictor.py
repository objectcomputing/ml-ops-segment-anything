import torch
from typing import Dict, List
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import base64
import numpy as np
import cv2
import logging


class CustomSamPredictor(Predictor):
    """
    CUSTOM PREDICTION ROUTINE CLASS
    ...

    Attributes
    ----------
    predictor : SamPredictor class instance
        used for prediction with prompts
    mask_generator : SamAutomaticMaskGenerator class instance
        used for automatic prediction
    sam : segment_anything.modeling.sam.Sam
        define the model variant for loading
    model_type: str
        type of model [required to be hardcoded]
    device: str, Optional
        define the device [default value is "cuda"]
    mask_with_prompts: Boolean
        define whether the input requires masking with prompts or not
    
    Methods
    -------
    load(self, artifacts_uri: str):
        Loading the model.
        
    preprocess(self, prediction_input: Dict):
        Preprocess the input data.
        
    predict(self, prediction_input: Dict):
        Predict using the model.
        
    postprocess(self, prediction_results: List):
        Post process the predictions.
    """
    
    def __init__(self, device = "cuda"):
        super().__init__()
        self.predictor = None
        self.mask_generator = None
        self.sam = None
        self.model_type = "vit_b" # change to the type of SAM checkpoint you are using
        self.device = device
        self.mask_with_prompts = True
    
    def load(self, artifacts_uri: str):
        """
            Loads the model artifacts.

            Parameters
            ----------
            artifacts_uri : str
                cloud storage location where the checkpoint is stored
    
            Returns
            -------
            None
        """
        print("************** Loading the model **********************")
        
        prediction_utils.download_model_artifacts(artifacts_uri) # comment this line for the code to work with local deployment testing
        
        """ Change the checkpoint name to one of the three variants of SAM, simultaneously change the model_type in constructor """
        self.sam = sam_model_registry[self.model_type](checkpoint="sam_vit_b_01ec64.pth")
        self.sam.to(device=self.device)
        
    
    def preprocess(self, prediction_input: Dict) -> Dict:
        """
            Data Preprocessing.

            Parameters
            ----------
            prediction_input : Dict
                Model input, requires processing
    
            Returns
            -------
            prediction_input: Dict
                Preprocessed data
        """
        print("************** PRE PROCESSING **********************")
        prediction_input = prediction_input["instances"][0]
        image = prediction_input["image"] # base64 format
        jpg_original = base64.b64decode(image)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=1)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if len(prediction_input) > 2: # Masking with prompts requires 3 inputs (file_path, image, input_points, input_label)
            """ Masking with prompts """
            print("PREDICTING WITH PROMPTS")
            
            self.predictor = SamPredictor(self.sam)
            self.predictor.set_image(image) 
            
            del prediction_input["image"] # deleting image base64 string since it is not required henceforth
            
        else: # Masking without prompts requires only image input
            """ Masking without prompts / automatic masking """
            print("PREDICTING WITHOUT PROMPTS")
            
            self.mask_with_prompts = False
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
            prediction_input["image_cvtColor"] = image
        
        return prediction_input
    
    # Get the predictions from the loaded model
    @torch.inference_mode()
    def predict(self, prediction_input: Dict) -> List:
        """
            Performs prediction.

            Parameters
            ----------
            prediction_input : Dict
                Processed Model input
    
            Returns
            -------
            List : Prediction
        """
        
        print("************** PREDICTING **********************")
    
        if self.mask_with_prompts:
            """ Masking with prompts """
            print("PREDICTING WITH PROMPTS")
            
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array(prediction_input["input_point"]).reshape(1,2),
                point_labels=np.array(prediction_input["input_label"]),
                multimask_output=False, # only one mask will be produced since multimask is set to FALSE
            )
            
            return list((prediction_input["file_path"], masks, scores, logits))
        else:
            """ Masking without prompts / automatic masking """
            print("PREDICTING WITHOUT PROMPTS")
            
            masks = self.mask_generator.generate(prediction_input["image_cvtColor"])
            
            return list((prediction_input["file_path"], prediction_input["image"], masks))
     
    
    # Returns the predictions as a dictionary
    def postprocess(self, prediction_results: List) -> Dict:
        """
            Postprocessing / construct response structure.

            Parameters
            ----------
            prediction_results : List
                Predictions
    
            Returns
            -------
            prediction: Dict
                Processed model predictions
        """
        

        print("************** POST PROCESSING **********************")
        
        prediction={}
        if self.mask_with_prompts:
            print(" Prediction response / Masking with prompts ")
            
            prediction["file_path"] = prediction_results[0]
            prediction["masks"] = prediction_results[1].tolist()
            prediction["scores"] = prediction_results[2].tolist()
            prediction["logits"] = prediction_results[3].tolist()
        else:
            print(" Prediction response / Masking without prompts / automatic masking ")
            
            prediction["file_path"] = prediction_results[0]
            prediction["image"] = prediction_results[1]
            prediction["masks"] = []
            for mask in prediction_results[2]:
                mask["segmentation"] = mask["segmentation"].tolist()
                prediction["masks"].append(mask) 
        
        return prediction