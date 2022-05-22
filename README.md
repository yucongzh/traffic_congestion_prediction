# Traffic Congestion Prediction

## Installation
1. Create environment
```
conda create -n py39 python=3.9
```

2. Install Python packages
```
pip install -r requirements.txt
```


## Usage
1. Feature Extraction 
Download `train.csv` from Kaggle under the current directory. Then, use the command `python feature_engineering.py` to generate the graph spectral embeddings `spectral_embds.pt`.

2. Train the lstm model
Use the command `python train.py` to generate models. The best lstm model will be saved under `./saved_models/` directory.

3. Generate the congestion results for the testing data
Use the command `python generate_results.py` to generate `results.txt`.
