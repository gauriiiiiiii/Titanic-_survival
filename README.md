# 🚢 Titanic Survival Prediction

This project predicts whether a passenger survived the Titanic disaster using machine learning models based on historical data. It is inspired by the famous Kaggle competition and implemented using Python and scikit-learn.

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Evaluation](#evaluation)
- [Results](#results)
- [How to Run](#how-to-run)
- [Conclusion](#conclusion)
- [License](#license)

---

## 📖 Overview

The goal of this project is to build a classification model that can accurately predict which passengers survived the Titanic shipwreck. The steps involved include:
- Data loading and cleaning
- Exploratory data analysis (EDA)
- Feature engineering
- Model training and evaluation

---

## 📂 Dataset

The dataset used is a CSV file named:


It includes the following features:
- `Pclass`: Passenger class (1st, 2nd, 3rd)
- `Sex`: Gender of passenger
- `Age`: Age in years
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Ticket fare
- `Embarked`: Port of embarkation

---

## 🛠 Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Jupyter Notebook

---

## 🧹 Data Preprocessing

Key preprocessing steps:
- Handled missing values (`Age`, `Embarked`) using median/mode
- Dropped non-informative columns: `Name`, `Ticket`, `Cabin`
- Encoded categorical variables: `Sex`, `Embarked`
- Scaled numerical features if needed
- Split data into features `X` and target `Y`

---

## 🤖 Model Building

Two machine learning models were implemented:

1. **Logistic Regression**

```python
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = log_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
