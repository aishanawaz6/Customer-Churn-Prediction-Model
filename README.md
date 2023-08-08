# Customer-Churn-Prediction-Model
Helping CommLink Telecom tackle high churn rates. I analyzed their data to build an model predicting customer churn. Python tools and EDA uncover insights. Deployed model aids real-time retention efforts.
Project Description
CommLink Telecom, a leading telecommunications company, has been experiencing significant customer churn rates. In an effort to combat this challenge proactively, DataSense Solutions has been engaged to develop a customer churn prediction model. The primary goal of this project is to identify customers who are likely to churn in the near future, enabling CommLink Telecom to implement targeted retention strategies and improve customer retention rates.

# Client Information
## Client Name: CommLink Telecom
## Company Name: DataSense Solutions
## Dataset
The project utilizes the dataset named CommLink_Telecom_Customer_Data.csv, containing information on 1000 customers. The dataset includes various attributes such as customer demographics, contract details, and usage patterns, along with a binary indicator of whether a customer has churned ("Yes") or remains active ("No").

CustomerID	Gender	Age	ServiceLength (months)	ContractType	MonthlyCharges (USD)	TotalCharges (USD)	Churn  
1001	Male	42	24	Two-Year	85.00	2040.00	No  
1002	Female	35	12	One-Year	79.50	942.50	Yes


# Steps
## Data Collection
The initial step involved the collection of the CommLink_Telecom_Customer_Data.csv dataset, which forms the foundation of the project.

## Data Cleaning
Data cleaning encompassed handling missing values, eliminating duplicates, and addressing inconsistencies within the dataset to ensure accurate analysis and model training.

## Data Preprocessing
In this step, categorical variables were encoded, feature scaling was performed, and skewed data were managed to prepare the dataset for analysis and modeling.

## Exploratory Data Analysis
Using Python libraries such as Pandas, Matplotlib, and Seaborn, the dataset was explored to gain insights into factors contributing to customer churn. Visualizations and statistical analyses were employed to uncover patterns and trends.

## Churn Prediction
A machine learning model was constructed, utilizing techniques such as Logistic Regression and Random Forest, to predict customer churn based on historical data.

## Deployment
The developed churn prediction model was deployed as an API or integrated into CommLink Telecom's system, enabling real-time churn predictions for individual customers. This facilitated the implementation of targeted strategies to retain customers at risk of churn.

## Tools Used
-> Python  
-> Jupyter Notebook  
-> Pandas  
-> Matplotlib  
-> Seaborn  
-> Scikit-learn  

## Project Outcome
The final project deliverables comprise a comprehensive Jupyter Notebook documenting the data analysis and modeling processes. Additionally, a report detailing insights into the factors influencing churn, along with the deployed churn prediction model, is provided.
