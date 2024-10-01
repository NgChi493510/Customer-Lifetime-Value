# Customer Lifetime Value (CLV) Analysis
## Overview
This project focuses on the estimation and prediction of Customer Lifetime Value (CLV), a critical metric for businesses to understand the total value a customer brings over their entire relationship with the company. CLV helps businesses optimize their customer acquisition strategies, improve retention efforts, and maximize long-term profits. This project utilizes multiple approaches, including traditional CLV models, probabilistic models like BG/NBD, and machine learning models to estimate and predict CLV.

The project explores two datasets:

1. Online Retail II - A dataset of UK-based online retail transactions.
2. CDNow - A dataset containing purchase history data.
## Project Motivation
Accurately predicting Customer Lifetime Value (CLV) is essential for:

- Optimizing marketing strategies: Allocate resources to high-value customers.
- Improving customer retention: Identify key segments and reduce churn.
- Maximizing return on investment (ROI): Efficiently allocate budgets to marketing campaigns.
- Understanding customer behavior: Use data to predict future purchasing patterns and loyalty.
## Datasets
1. **Online Retail II** 
This dataset contains all transactions for a UK-based, non-store online retailer from 2009-2011. Key features include:

- InvoiceNo: Unique transaction ID
- StockCode: Product code
- Quantity: Number of units per transaction
- UnitPrice: Price per product
- CustomerID: Unique identifier for customers
2. **CDNow**
This dataset includes purchase history data of 23,570 customers who made their first purchase in the first quarter of 1997.

## Data Preprocessing
- Handling Missing Data: Missing values were imputed using statistical techniques.
- Feature Engineering: Key features such as Recency, Frequency, and Monetary value (RFM) were computed to enhance prediction models.
- Scaling: Data was normalized to improve model performance.
## Methodology
### 1. Traditional CLV Calculation
We employed a basic formula for Customer Lifetime Value using the following steps:

- Average Order Value (AOV): Total revenue divided by the number of orders.
- Purchase Frequency (PF): Total orders divided by the number of customers.
- Churn Rate: 1 - Repeat Rate (percentage of returning customers).
- **CLV Formula:**
$$CLV = \left( \frac{AOV \times PF}{Churn\ Rate} \right) \times Profit\ Margin$$
​
### 2. Probabilistic Models (BG/NBD and Gamma-Gamma)
- BG/NBD Model: Predicts customer purchase behavior by estimating the probability of customer churn and future purchase frequency using historical data.
- Gamma-Gamma Model: Estimates Average Order Value (AOV) and models the variation in customer spending patterns.
### 3. Machine Learning Models
We applied machine learning techniques for CLV prediction:

- RFM Segmentation: Customers are classified based on Recency, Frequency, and Monetary value using models like K-Means, Gaussian Mixture Models (GMM), and Hierarchical Clustering.
- Classification Models: Predict customer segments using models such as XGBoost, K-Nearest Neighbors, Logistic Regression, and Random Forest.
- Regression Models: Predict CLV values using Linear Regression, Random Forest, XGBoost, CatBoost, and more.
### 4. Deep Learning Model
We implemented a Deep Neural Network (DNN) for CLV prediction. The DNN is trained using engineered features and evaluated on its ability to generalize future customer behavior:

- Training and Evaluation: Data was split into training, evaluation, and test sets to ensure proper model validation.
- Hyperparameter Tuning: Techniques like grid search were used to optimize the model for accuracy and minimize overfitting.
## Results
### 1. Probabilistic Models
BG/NBD Model: Accurately predicts customer retention and future purchases.
Gamma-Gamma Model: Improves AOV estimation by considering the heterogeneity in customer spending.
### 2. Machine Learning Models
Classification: Best performing models included Random Forest and Gradient Boosting, with an accuracy of 87% in predicting customer segments.
Regression: The Gradient Boosting Regressor and Linear Regression performed best, achieving high R² scores and low Mean Squared Error (MSE).
### 3. Deep Learning
The Deep Neural Network (DNN) model outperformed traditional models in terms of predictive accuracy, with an R² score of 0.6052 on the Online Retail II dataset, demonstrating its potential for more complex and accurate CLV predictions.
## Conclusion
This project demonstrates that probabilistic models, machine learning, and deep learning are all powerful tools for Customer Lifetime Value (CLV) prediction. The choice of model depends on the available data, complexity, and desired accuracy. For businesses, understanding CLV can significantly enhance marketing efforts, improve resource allocation, and maximize long-term profitability.

## Future Work
Expand datasets: Incorporate additional data sources such as social media or CRM data to further improve predictions.
Advanced probabilistic models: Explore techniques such as Hierarchical Bayesian Models or Hidden Markov Models.
Continuous model updates: Implement systems for real-time model updates and monitoring.
## Usage
**Requirements**
- Python 3
- Libraries: pandas, numpy, scikit-learn, xgboost, catboost, lifetime, tensorflow, matplotlib