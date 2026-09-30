# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn.

## Project Overview

This project uses the Telco Customer Churn dataset and applies machine learning techniques for customer churn prediction.

Two classification models were trained and evaluated:

* Logistic Regression
* Random Forest

The Logistic Regression model is used for the API deployment based on the evaluation results from the test dataset.

## Model Explanation

### Why Logistic Regression was chosen

Logistic Regression is suitable for this project because customer churn is a binary classification problem where the output is either churn or no churn.

The Logistic Regression model achieved the following results on the test data:

* Accuracy: 81.97%
* Precision: 68.31%
* Recall: 59.52%
* F1 Score: 63.61%

It was selected for the API after comparing its performance with the Random Forest model.

### How Features Impact Prediction

The model uses customer information such as:

* Tenure
* Monthly charges
* Total charges
* Contract type
* Internet service
* Payment method
* Online security
* Technical support
* Partner and dependent information

These features provide information about the customer's service usage, billing, contract, and account characteristics. The model uses these patterns to estimate the probability of churn.

Feature analysis was also performed using the Random Forest model to understand the relative importance of the input features.

## API

The trained Logistic Regression model is deployed using FastAPI.

The API accepts customer information and returns a churn prediction in JSON format.

Example response:

```json
{
  "prediction": "No Churn",
  "churn_probability": 0.3271
}
```

## Project Structure

```text
customer_churn_ai/
│
├── data/
├── notebook/
├── models/
├── api/
├── README.md
└── requirements.txt
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project

```bash
cd customer_churn_ai
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

Go to the API folder:

```bash
cd api
```

Start the FastAPI server:

```bash
python -m uvicorn main:app
```

## GitHub

The complete project is available on GitHub, including:

* Dataset
* Jupyter Notebook
* Trained models
* FastAPI application
* Requirements
* Project documentation
