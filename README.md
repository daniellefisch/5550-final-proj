# Modeling Climate Stress Impacts on U.S. Corn Yield
### Evaluating Predictive Robustness and Regional Differences

**Danielle Fischer**

DSAN5550 - Data Science for Climate Change

Georgetown University

## Repo Structure

Start with: `notebooks/final_pipeline.ipynb`

This notebook walks through nearly the full pipeline, going through all of the data and including most of the main analysis. The poster (in the repo as a PDF) has the final information. 

The repository is organized into four main components:
### `data/`
- `raw/` - original datasets (not included due to size)
- `clean/` – processed datasets used for modeling  
- `shapefiles/` – geographic boundary data (not included)

### `src/`
Contains all data processing and modeling scripts:
- Data processing: `unzip_prism.py`, `process_usda.py`, `process_prism.py`, `process_drought.py`
- Data merging: `merge_final_data.py`
- Models: `model_linear.py`, `model_random_forest`, `model_gradient_boosting`, `model_gp_additional.py`
- Feature analysis: `feature_importance.py`
- Running scripts: `run_all.py`
- Visualizations: `visualizations.py`

### `notebooks/`
- `project_pipeline.ipynb` – prototype pipeline and results, on less data and without feature importance
- `final_pipeline.ipynb` – preliminary analysis on results 

### `outputs/`
- Model metrics, predictions, and feature importance results 
- `visualizations/` -  visualizations used in poster + some extras

## Project Goal

This project investigates how climate stressors affect county-level corn yields across the United States, with a focus on:

- identifying key environmental drivers of yield variability
- evaluating how predictive models perform under different climate conditions
- assessing whether models trained in one region of the US generalize to another

The project specifically compares the **Corn Belt** and **Great Plains** regions to understand:

> How robust are predictive models of agricutlural yield when applied across regions with different climate stress profiles?

## Data and Methods

### Data sources

This project integrates multiple datasets at the county-year level:

- **USDA NASS Yield Data**
    - County-level corn yield (bushels per acre)
    - Coverage: Corn Belt and Great Plains states
    - Time range: 2000-2022

- **PRISM CLimate Data**
    - Temperature (tmax)
    - Precipitation
    - High-resolution gridded climate data aggregated to counties

- **Drought Data**
    - U.S. Drought Monitor indicators
    - Measures of drought severity and duration

Shapefiles (U.S. county boundaries) are not included due to size

They can be downloaded from:
https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

### Data Processing

Data are processed into a unified county-year dataset:

- Standardized county identifiers (FIPS)
- Aggregated climate variables
- Merged yield, climate, and drought data
- Filtered to Corn Belt and Great Plains regions

### Modeling Approach

Models are trained to predict county-level corn yield using clijmate features.

Approach:
- train models on **Corn Belt data**
- evaluate:
    - in-region performance (cross-validation)
    - out-of-region performance (Great Plains)

Models include:
- linear regression
- random forest
- gradient boosting

### Metrics

**Evaluation Metrics**

| Metric | What it Measures | Interpretation | Units | Notes |
|--------|----------------|----------------|-------|------|
| R² (R-squared) | Proportion of variance in yield explained by the model | 1 = perfect fit, 0 = no better than mean, <0 = worse than mean | Unitless | Main metric used in this project; highlights transfer failure |
| RMSE | Average magnitude of prediction error (squared, then rooted) | Lower is better; sensitive to large errors | Bushels per acre | Penalizes large mistakes more heavily |
| MAE | Average absolute prediction error | Lower is better; easier to interpret than RMSE | Bushels per acre | More robust to outliers than RMSE |

All of the resulting metrics can be found at the end of final_pipeline.ipynb

### Results Summary

The results show that climate variables explain a moderate portion of variation in corn yield within regions, with nonlinear models such as Random Forest and Gradient Boosting achieving R² values of approximately 0.20–0.35 and consistently outperforming linear regression. Including drought variables slightly improves within-region performance, but does not improve, and in some cases slightly worsens, performance under transfer. However, when models are applied across regions, performance declines sharply: most transfer scenarios result in negative R² values, indicating that models perform worse than predicting the mean. Transfer from the Great Plains to the Corn Belt performs somewhat better than the reverse, with small positive R² values for nonlinear models, but overall generalization remains limited. Feature importance results show that temperature is the most influential predictor in both regions, while precipitation plays a larger role in the Great Plains and drought intensity is more important in the Corn Belt. These patterns demonstrate that both predictive performance and the relative importance of climate variables differ across regions.


### Prototype Pipeline

A simplified prototype was developed to validate the pipeline:

- Uses a reduced dataset (subset of states and features)
- Includes only temperature as an initial test feature
- Confirms data cleaning, merging, and modeling workflow

This was done to make sure that the pipeline made sense prior to loading in the more complicated prism and drought data. This prototype pipeline can be seen in the folder notebooks/ under the file project_pipeline.ipynb.