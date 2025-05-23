# Detecting Fraudulent Credit Card Transactions

This project is a machine learning-driven fraud detection system that analyzes customer transaction patterns to classify whether a credit card transaction is legitimate or fraudulent. Built using Python and core data science libraries, it simulates real-world financial fraud detection scenarios with over 1 million records and no missing values.


## Objective

To develop and evaluate supervised machine learning models to accurately classify fraudulent credit card transactions and explore the impact of various transaction behaviors on the likelihood of fraud.


## Dataset Overview

- **Source**: Kaggle
- **Size**: 1 million transaction records
- **Target Variable**: `fraud` (binary: 1 = fraudulent, 0 = legitimate)
- **Key Features**:
  - `distance_from_home`
  - `distance_from_last_transaction`
  - `ratio_to_median_purchase_price`
  - `used_pin_number`, `used_chip`, `online_order`


## Research Hypotheses

1. Online transactions are more likely to be fraudulent.
2. Greater distance from home increases the chance of fraud.
3. Long gaps between transaction locations might indicate fraud.


## Tech Stack

- **Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Scikit-learn
- **Modeling**: Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, SVC


## Project Structure

```
Fraud-Detection-ML/
│
├── data/
│   └── card_transdata.csv                  # Raw transaction dataset
│
├── notebooks/
│   ├── SmartQuestion-1.py                  # EDA and hypothesis testing
│   ├── SmartQuestion-3.py                  # Classification models
│   ├── SmartQuestion-4.py                  # Model improvement
│   └── Fraudulent_Credit_Detection.py      # Consolidated fraud detection code
│
├── src/
│   ├── get_data.py                         # Load and preprocess data
│   ├── split_data.py                       # Train-test split utility
│   ├── build_model.py                      # Model training module
│   ├── model_metrics.py                    # Evaluation metrics
│   └── master.py                           # Main controller script
│
└── README.md                               # You are here
```


## Model Performance

| Model                | Accuracy | Recall | F1 Score |
|---------------------|----------|--------|----------|
| Logistic Regression | 0.972    | 0.842  | 0.902    |
| KNN (k=10)           | 0.997    | 0.978  | 0.983    |
| Random Forest        | 1.000*   | 1.000  | 1.000    |
| SVC                  | 0.991    | 0.660  | 0.774    |

>  *Perfect training score; may require tuning or further evaluation for real-world scenarios.*


## Key Takeaways

- KNN with K=10 was the most balanced and effective model.
- Random Forest hit perfect metrics but may be overfitting.
- Online orders and distance features played a significant role in fraud prediction.


## Author

**Surya Vamsi Patiballa**
MS in Data Science – George Washington University

- Email  :-  svamsi2002@gmail.com  
- LinkedIn  :-  https://www.linkedin.com/in/surya-patiballa-b724851aa/


## Future Enhancements

- Real-time prediction web app integration
- Auto-retraining pipelines
- Feature engineering from time-based patterns
