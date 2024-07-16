# Real Estate Recommendation System using Graph Neural Networks
## Project - 1
## Project Overview

This project implements a recommendation system for real estate listings using Graph Neural Networks (GNNs). It is designed to predict user interactions (clicks and purchases) with property listings based on user behavior, property features, and location data.

## Key Points

- Heterogeneous Graph Neural Network model
- Integration of user behavior, ad features, and location data
- Click and purchase link prediction
- Performance evaluation using AUC-ROC and Average Precision metrics


## Installation

1. Clone the repository:
```python
git clone https://github.com/nazlicanto/gnn-estate.git
cd gnn-estate
```

2. Install the required packages:  
```pip install -r requirements.txt```

## Data Preparation
The system requires two main data sources:

1. `davranis.csv`: User interaction data
2. `ad_features.csv`: Property listing features

ENsure these files are placed in the appropriate directories based on the project-number

## Usage

To train and evaluate the model, locate to project-1 and run `python training_sage.py` 



## Model Architecture
The GNN model consists of:

- Heterogeneous Graph Convolution layers (HeteroConv)
- SAGEConv layers for message passing
- Linear layers for final predictions

The model processes three types of nodes: users, ads, and locations.

## Training Process

The training process includes:

1. Graph creation from input data
2. Model initialization
3. Training loop with early stopping
4. Evaluation on a test set

## Evaluation Metrics

- AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
- Average Precision
- NDCG (Normalized Discounted CUmulative Gain)
- MRR (Mean Reciprocal Rank)
- Hit Rate

## Results

- Validation AUC-ROC:
Model's increasing ability to distinguish between positive and negative
samples.
- Validation Average Precision:
Improved precision in ranking relevant items higher.


## Future Improvements

- Better implementation on evaluatio metrics
- Implement more sophisticated negative sampling techniques 
- Explore different GNN architectures (e.g., GAT)


# Real Estate Query Relevance Analysis
## Project - 2
## Project Overview
This project analyzes the relevance of real estate queries using machine learning techniques. Processes dataset of real estate queries and their associated metrics, engineers relevant features, selects the most important features, trains a hyperparameter tuned Random Forest model, and provides insights using SHAP (SHapley Additive exPlanations) analysis.

## Features
- Data preprocessing and feature engineering
- Automatic feature selection using SelectKBest
- Random Forest Regression model
- Cross-validation for model evaluation
- SHAP analysis for model interpretability


## Installation
1. Clone the repository:
```python
git clone https://github.com/nazlicanto/gnn-estate.git
cd gnn-estate
```

2. Install the required packages:  
```pip install -r requirements.txt```


## Usage

Ensure your data file is named 'alaka.csv' and placed in the 'project-2' directory.

Run the script:

``` python 
python random_fo.py
```
The script will create a new directory with the current timestamp to store all outputs.

## Output
The script generates the following outputs in the created directory:

analysis_log.txt: Detailed log of the analysis process  
rf_pipeline.joblib: Saved Random Forest model  
learning_curve.png: Learning curve plot  
shap_summary_plot.png: SHAP summary plot   
shap_bar_plot.png: SHAP bar plot  
shap_dependence_plot_*.png: SHAP dependence plots for top features   
shap_force_plot.png: SHAP force plot for a single prediction   


## Results
1. attr and its related features (attr_cont_product, attr_percentile) have the strongest and most consistent positive impact on relevance predictions.
2. Satisfaction score and sat_click also have significant impacts, but their relationships with relevance are more complex and non-linear.
3. The model gives more weight to high-performing items (high attr percentiles, high satisfaction scores) in determining relevance.

