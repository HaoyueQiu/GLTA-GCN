# GLTA-GCN
ICME2022 GLTA-GCN

## System requirements

Our model has been tested with PyTorch 1.7.0 and CUDA 11.0

## Usage

* **Setup environment**

We use DLNest to build our model, so the first step is to download DLNest.

```
pip install git+https://github.com/SymenYang/DLNest.git
```

Then according to the instruction of DLNest, place the GLTA-GCN to corresponding directory.

* **Change configs**

  Here are several configs:

  dataset_config.json： Common configurations for datasets, e.g., where the dataset is.

  model_config.json: Common configurations for models

  GCN_Unsupervised/freq_config_part.json：freq_config is used to change some parameters which modify frequently, such as the number of TAU layers.

  

* **Run**

After finishing config,  we can run the model.

Still following DLNest's instructions 

```
run -c root_config -f freq_config -m memory [-mc] -d "comments"
```



