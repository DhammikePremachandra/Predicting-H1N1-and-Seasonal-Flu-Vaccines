# Predicting H1N1 and Seasonal Flu Vaccines

## Project Overview
This project focuses on predicting the likelihood of individuals receiving the H1N1 and seasonal flu vaccines. The dataset includes various features such as demographic information, behavioral factors, and health-related conditions. The objective is to build and evaluate machine learning models to accurately predict vaccine uptake.

## Datasets
The project utilizes the following datasets:
- **`training_set_features.csv`**: Contains features for the training data.
- **`training_set_labels.csv`**: Contains labels for the training data, indicating whether an individual received the H1N1 or seasonal flu vaccine.
- **`test_set_features.csv`**: Contains features for the test data to be used for predictions.

## Project Structure
The project is structured into the following steps:

1. **Data Loading**: Import the datasets and perform initial inspection.
2. **Data Preprocessing**:
   - Handling missing values.
   - Encoding categorical variables using OneHotEncoder and OrdinalEncoder.
3. **Feature Selection**:
   - Utilizing `SelectKBest` with the `chi2` statistic to select the most relevant features.
4. **Modeling**:
   - Several models are trained, including Logistic Regression and Random Forest.
   - Addressing class imbalance with SMOTE (Synthetic Minority Over-sampling Technique).
5. **Model Evaluation**:
   - Models are evaluated using metrics such as ROC-AUC scores.
6. **Prediction on Test Data**:
   - Final predictions are made on the test dataset using the best-performing model.

## Requirements
To run the notebook, you need to have the following Python libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## How to Run the Project
1. Clone this repository to your local machine.
2. Ensure that the datasets (`training_set_features.csv`, `training_set_labels.csv`, `test_set_features.csv`) are placed in the same directory as the notebook.
3. Open and run the `Predict_H1N1_Flu.ipynb` notebook using Jupyter or any compatible environment.

## Results
The models were evaluated on the training dataset, with the following key results:
- **ROC-AUC Score**: A measure of the model's ability to distinguish between the classes.

The Random Forest model, when combined with SMOTE for handling class imbalance, performed the best among the tested models.

## Future Work
Potential improvements for this project include:
- **Hyperparameter Tuning**: Experimenting with different hyperparameters to improve model performance.
- **Feature Engineering**: Exploring additional features, feature interactions, and transformations.
- **Model Ensemble**: Using ensemble methods such as stacking or boosting to combine multiple models for better predictions.

## License
This project is licensed under the MIT License. Feel free to use and modify the code as needed.
