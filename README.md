# Sales Prediction Model

A machine learning model created to predict monthly sales for a retail store, with a particular focus on forecasting December sales.

## Project Overview

This project implements a linear regression model to analyze monthly sales data and predict future sales. The model shows a clear upward trend in sales throughout the year.

## Features

- Data preprocessing and cleaning
- Exploratory data analysis with visualizations
- Linear regression model for sales prediction
- Performance evaluation using appropriate metrics
- December sales forecasting
- Data visualization through histograms and scatter plots

## Project Structure

```
previsao_vendas/
│
├── modelo_previsao_vendas.ipynb  # Jupyter notebook with model development
├── Pipfile                       # Dependency management
├── Pipfile.lock                  # Locked dependencies
├── README.md                     # This documentation file
```

## Requirements

The project requires the following Python libraries:
- pandas: Data manipulation and analysis
- matplotlib: Data visualization
- seaborn: Enhanced data visualization
- numpy: Numerical operations
- scikit-learn: Machine learning algorithms

## Installation

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/username/previsao_vendas.git
cd previsao_vendas

# Install dependencies using pipenv
pipenv install
```

Or install the required packages manually:

```bash
pip install pandas matplotlib seaborn numpy scikit-learn
```

## Usage

1. Open the Jupyter notebook `modelo_previsao_vendas.ipynb`
2. Run all cells to:
   - Load and preprocess sales data
   - Train the linear regression model
   - Evaluate model performance
   - Generate visualizations
   - Predict December sales

## Model Development Process

### 1. Data Loading and Preprocessing

The model loads monthly sales data and performs necessary preprocessing steps:
- Data cleaning
- Feature formatting
- Handling missing values (if any)

### 2. Exploratory Data Analysis

Before model training, we explore the data through:
- Histogram of sales distribution
- Scatter plot of monthly sales
- Statistical analysis of sales patterns

### 3. Model Training

We use scikit-learn's LinearRegression to train the model:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
```

### 4. Model Evaluation

The model's performance is evaluated using:
- Mean Squared Error (MSE)
- R-squared (R²) score
- Visual comparison of actual vs. predicted values

### 5. December Sales Prediction

Using the trained model, we forecast December sales to help with inventory planning and business strategy.

## Results

The scatter plot visualization shows:
- A strong linear relationship between month number and sales volume
- Sales increasing from approximately 2,000 units in January to 3,300 units in December
- The red line represents the model's predictions, showing a good fit to the actual data points

## Future Improvements

- Incorporate additional features such as:
  - Seasonal factors
  - Marketing campaign data
  - Economic indicators
- Experiment with more complex models (Random Forest, Gradient Boosting)
- Implement time series forecasting techniques
- Add confidence intervals to predictions

## Contributors

- Matheus Maciel - model training

## License

MIT LICENSE
