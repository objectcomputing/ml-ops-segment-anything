# ML-Operations-Segment-Anything
Cloud architecture leveraged to unlock segmentation of thousands of images using Google Cloud Platform
![SAM-model-workflow](demo-notebooks/images/sam_workflow.png?raw=true)

## Installation and GCP(Google Cloud Platform) Services config
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Installing both PyTorch and TorchVision with CUDA support is strongly recommended. Also other optional dependencies include opencv-python and matplotlib for masks post-processing. Follow the instructions carefully in each cell of every notebook for easy understanding of code and the necessary installations.

Follow the steps below to enable GCP services usage:
+ Create a Repository in Artifacts Registry on GCP in **standard docker format**.
+ Create new Workbench instance in VertexAI with following configuration and clone the [ml-ops-segment-anything](https://github.com/objectcomputing/ml-ops-segment-anything/tree/main) repository.
  - **Operating System**: Debian 11
  - **Environment**: Pytorch
  - **Machine Type**: n1-standard-2(1vCPUs, 7.5GB RAM)
  - **GPU**: Nvidia T4
  - **GPU Count**: 1
  - Select the checkbox which says **Install NVIDIA GPU driver automatically for me**
  - Keep the remaining configurations as it is
+ Create a bucket with unique name in Google Cloud Storage with default settings. Create 2 folders inside the bucket with following names:
  - **sam-checkpoint**.
  - **batch-prediction-images**.
  - **pipeline**
+ Upload the SAM checkpoint of your choice to **sam-checkpoint** folder inside the bucket created.
+ Upload the images to **batch-prediction-images** folder to test Batch Predictions.

## SAM Demo
Run the [sam-demo-notebook](https://github.com/objectcomputing/ml-ops-segment-anything/blob/dev/demo-notebooks/sam-demo.ipynb) in demo-notebooks folder for examples on using SAM with prompts and automatically generating masks.

## VertexAI Online Predictions
Deploying SAM to an endpoint in Vertex AI requires Custom Container with Custom Prediction Routine.

The [source code](https://github.com/objectcomputing/ml-ops-segment-anything/tree/dev/vertexAI_online_predictions/src) folder in Online Predictions has [Custom Prediction Routine(CPR)](https://github.com/objectcomputing/ml-ops-segment-anything/blob/dev/vertexAI_online_predictions/src/custom_sam_predictor.py) and [requirements](https://github.com/objectcomputing/ml-ops-segment-anything/blob/dev/vertexAI_online_predictions/src/requirements.txt) file that is required to build the custom container. CPR has the capability of handling predictions with prompts and without prompts. 

Follow the steps below to deploy the model and test the endpoint on VertexAI:

+ Run the [Online Prediction Model Input Preparation](https://github.com/objectcomputing/ml-ops-segment-anything/blob/dev/vertexAI_online_predictions/online_predict_model_input_prep.ipynb) notebook to generate 2 JSON files for Endpoint testing. Also refer to the endpoint input JSON structure in the notebook.
+ Follow the instructions in each cell and run the [Containerization notebook](https://github.com/objectcomputing/ml-ops-segment-anything/blob/dev/vertexAI_online_predictions/containerise-deploy.ipynb) to build the custom container and push the container to the Repository created in Artifacts Registry on GCP. You can also deploy locally and test the endpoint.
+ Go to Model Registry on VertexAI to register the model for deployment:
  - Click IMPORT and under **Model Settings** select **Import an existing custom container**. Click on **BROWSE** and select the container from the Artifacts Registry.
  - Browse and select the path to the Cloud Storage directory where the exported model file is stored.
  - Set the prediction route to **/predict** and health route to **/health**.
  - Keep the remaining configuration as it is and import the model.
+ Go to Online Prediction on VertexAI to deploy the model to an Endpoint:
  - Click CREATE and give the endpoint name
  - Under **Model Settings** select the registered model name and the version.
  - Under **ADVANCED SCALING OPTIONS** choose the following configurations:
    * **Machine Type**: n1-standard-2(1vCPUs, 7.5GB RAM)
    * **Accelerator Type**: NVIDIA_TESLA_T4
    * **Accelerator Count**: 1
  - Keep the remaining configuration as it is and initiate endpoint creation
+ Follow the instructions in the [Containerization notebook](https://github.com/objectcomputing/ml-ops-segment-anything/blob/dev/vertexAI_online_predictions/containerise-deploy.ipynb) to test the endpoint once the deployment is complete.
+ The Response JSON structure of the endpoint is of the format:
  - Predicting with prompts
  ```python
  {
    "file_path"                 : file name,
    "masks"                     : [masks],
    "scores"                    : [scores],
    "logits"                    : [logits]
  }
  ```
  
  - Predicting without prompts
  ```python
  {
    "file_path"                 : file name,
    "masks"                     : [masks]
  }
  ```
## VertexAI Batch Predictions
Register the model in Model Registry and use the **Model ID** in **Version Details** of the registered model in the [batch prediction notebook](https://github.com/objectcomputing/ml-ops-segment-anything/blob/dev/vertexAI_batch_prediction/batch_prediction.ipynb) to test Batch Predictions. Follow the instructions in the notebook to set up a batch prediction job.

## VertexAI Pipeline
Machine Learning Pipeline job for Segment-Anything Model is setup using Kubeflow SDK with component based approach. Here the Pipeline job is capable of handling a batch of images and processing them in sequence and finally outputing individual image segments upon original image and saving it into Cloud Storage. As a result of successful pipeline run, to enable Static Visualization, A markdown file is generated which shows individual images along with maximum 10 segments laid separately on the original image.

#### Base Imag





