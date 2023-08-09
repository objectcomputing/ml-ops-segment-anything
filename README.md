# ML-Operations-Segment-Anything
Cloud architecture leveraged to unlock segmentation of thousands of images using Google Cloud Platform
![SAM-model-workflow](demo-notebooks/images/sam_workflow.png?raw=true)

## Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Installing both PyTorch and TorchVision with CUDA support is strongly recommended. Also other optional dependencies include opencv-python and matplotlib for masks post-processing. Follow the instructions carefully in each cell of every notebook for easy understanding of code and the necessary installations

## SAM Demo
Run the [sam-demo-notebook](https://github.com/objectcomputing/ml-ops-segment-anything/blob/dev/demo-notebooks/sam-demo.ipynb) in demo-notebooks folder for examples on using SAM with prompts and automatically generating masks.

## VertexAI Online Predictions
Deploying SAM to an endpoint in Vertex AI requires Custom Container with Custom Prediction Routine.

The [source code](https://github.com/objectcomputing/ml-ops-segment-anything/tree/dev/vertexAI_online_predictions/src) folder in Online Predictions has [Custom Prediction Routine](https://github.com/objectcomputing/ml-ops-segment-anything/blob/dev/vertexAI_online_predictions/src/custom_sam_predictor.py) and (requirements)(https://github.com/objectcomputing/ml-ops-segment-anything/blob/dev/vertexAI_online_predictions/src/requirements.txt) file that is required to build the custom container.

Follow the steps below to deploy the model and test the endpoint on VertexAI:

+ Create a Repository in Artifacts Registry on GCP in standard docker format.
+ Run the [Online Prediction Model Input Preparation](https://github.com/objectcomputing/ml-ops-segment-anything/blob/dev/vertexAI_online_predictions/online_predict_model_input_prep.ipynb) notebook to generate 2 JSON files for Endpoint testing.
+ Follow the instructions in each cell and run the [Containerization notebook](https://github.com/objectcomputing/ml-ops-segment-anything/blob/dev/vertexAI_online_predictions/containerise-deploy.ipynb) to build the custom container and push the container to the Repository created in Artifacts Registry on GCP.

## VertexAI Batch Predictions

## VertexAI Pipeline
Collection of notebooks to setting up the pipeline, allocating resoucrces for batch predictions.




