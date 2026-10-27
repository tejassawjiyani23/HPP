# House Price Prediction Using Machine Learning: A Comparative Analysis

## 1. Project Overview

This project predicts house prices using machine learning regression
algorithms. It compares multiple models and selects a model for
estimating prices based on property characteristics.

The project also provides an interactive web application built using
Streamlit.

## 2. Objectives

- Analyze the house sales dataset.
- Perform exploratory data analysis (EDA).
- Train multiple machine learning regression models.
- Compare model performance using MAE, RMSE, and R².
- Tune selected models to improve performance.
- Save the selected model for reuse.
- Predict house prices through a web application.

## 3. Dataset

Dataset: House Sales in King County, USA

Source:
https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

The dataset contains house sale records and property information.

Place the dataset in the `data` folder as `train.csv` after preparing
it through the notebook.

## 4. Machine Learning Models

The project compares the following algorithms:

1. Linear Regression
2. K-Nearest Neighbors Regression
3. Decision Tree Regression
4. Random Forest Regression
5. Tuned K-Nearest Neighbors Regression
6. Tuned Random Forest Regression

## 5. Input Features

The application uses these ten features:

- Bedrooms
- Bathrooms
- Living area in square feet
- Lot area in square feet
- Number of floors
- Waterfront status
- View rating
- House condition
- Construction grade
- Year built

## 6. Evaluation Metrics

- **MAE:** Mean Absolute Error. Lower values are better.
- **RMSE:** Root Mean Squared Error. Lower values are better.
- **R² Score:** Measures how well the model explains variation in prices.
  A value closer to 1 generally indicates a better fit.

## 7. Project Structure

House-Price-Prediction/
- House_Price_Prediction.ipynb
- app/
  - app.py
- data/
  - train.csv
- models/
  - best_model.pkl
- outputs/
  - figures/
  - results/
- requirements.txt
- README.md

## 8. Installation

Install Python and open a terminal in the main project directory.

Install the required libraries:

```bash
pip install -r requirements.txt
```

## 9. Train the Model

Run `House_Price_Prediction.ipynb` from beginning to end.

The notebook prepares the dataset, trains and evaluates the models,
saves comparison results and figures, and exports the selected model
to `models/best_model.pkl`.

## 10. Run the Web Application

From the main project directory, run:

```bash
streamlit run app/app.py
```

Open the local URL displayed in the terminal.

Enter the property details and click **Predict House Price**.

## 11. Results

The notebook generates a model comparison table containing MAE,
RMSE, and R² scores. Refer to:

- `outputs/results/model_comparison.csv`
- `outputs/figures/model_mae_comparison.png`
- `outputs/figures/model_r2_comparison.png`

Actual results depend on the dataset and training process.

## 12. Limitations

- Predictions depend on the quality and coverage of the training data.
- The model may not reflect current market changes.
- Estimated prices are not guaranteed selling prices.
- Performance may vary for properties outside the training data range.

## 13. Future Enhancements

- Add location-based features.
- Improve model tuning and validation.
- Connect the application to a database.
- Deploy the application online.
- Add visualizations and prediction history.

## 14. Conclusion

This project demonstrates how machine learning regression models
can be compared for house price prediction and integrated into an
interactive web application.
