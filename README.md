# ML-Operations-Segment-Anything
Cloud architecture leveraged to unlock segmentation of thousands of images using Google Cloud Platform
![SAM-model-workflow](demo-notebooks/images/sam_workflow.png?raw=true)

## Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Installing both PyTorch and TorchVision with CUDA support is strongly recommended. Also other optional dependencies include opencv-python and matplotlib for masks post-processing. Follow the instructions carefully in each cell of every notebook for easy understanding of code and the necessary installations

## SAM Demo
Run the [sam-demo-notebook](https://github.com/objectcomputing/ml-ops-segment-anything/blob/dev/demo-notebooks/sam-demo.ipynb) in demo-notebooks folder for examples on using SAM with prompts and automatically generating masks.

### vertex-ai online predictions

### vertex-ai batch predictions

### vertex-ai pipeline
Collection of notebooks to setting up the pipeline, allocating resoucrces for batch predictions.




