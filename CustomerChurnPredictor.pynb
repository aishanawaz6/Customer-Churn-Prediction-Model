# Project: Customer Churn Prediction for Telecommunications Company

# Client Name: CommLink Telecom
# Company Name: DataSense Solutions

# Description: CommLink Telecom, a telecommunications company, is facing high customer churn rates 
# and wants to address the issue proactively. They have engaged DataSense Solutions 
# to build a customer churn prediction model that can identify customers likely to churn in the near future. 
# This will enable them to take targeted retention measures and improve customer retention rates.

# Dataset: CommLink_Telecom_Customer_Data.csv

# | CustomerID | Gender | Age | ServiceLength (months) | ContractType | MonthlyCharges (USD) | TotalCharges (USD) | Churn |
# |------------|--------|-----|-----------------------|--------------|---------------------|--------------------|-------|
# | 1001       | Male   | 42  | 24                    | Two-Year     | 85.00               | 2040.00            | No    |
# | 1002       | Female | 35  | 12                    | One-Year     | 79.50               | 942.50             | Yes   |
# | 1003       | Male   | 62  | 48                    | Month-to-Month | 94.20              | 4567.75            | Yes   |
# | 1004       | Female | 52  | 36                    | One-Year     | 78.25               | 2853.50            | No    |
# | 1005       | Male   | 28  | 6                     | Month-to-Month | 68.75              | 452.25             | No    |
# | ...        | ...    | ... | ...                   | ...          | ...                 | ...                | ...   |
# (Note: The dataset contains a total of 1000 customers with some churned (Yes) and others active (No).)

# Steps:
#1) Data Collection: create data set
import pandas as pd
import random
#Generating dataset entirely randomly gives very poor results while analysing data (as its totally random)
#Hence I am changing my approach to creating dataset a bit

size=1010
CommLinkTelecomData={
    'CustomerID':[1001,1002,1003,1004,1005]+[1002,1003,None]+[i for i in range (1006,1006+size-3) ],
    'Gender':['Male','Female','Male','Female','Male']+['Male' for _ in range(750)]+['Female' for _ in range (260)],
    'Age':[42,35,62,52,28]+[int(random.randint(1,100)) for i in range(500)]+[int(random.randint(20,40)) for i in range(510)],
    'ServiceLength (months)':[24,12,48,36,6]+[random.randint(1,90) for i in range (size-5)]+[-10]+[None for i in range(4)],
    'ContractType':['Two-Year','One-Year','Month-to-Month','One-Year','Month-to-Month']+[random.choice(['Two-Year','One-Year','Month-to-Month'])for i in range (500)]+['One-Year' for i in range (507)]+[None for _ in range (3) ],
    'MonthlyCharges (USD)':[85.00,79.50,94.20,78.25, 68.75]+[random.randint(30,80) for i  in range (500)]+[random.randint(50,100) for i in range (509)]+[None],
    'TotalCharges (USD)':[2040.00,942.50,4567.75,2853.50,452.25]+[random.randint(400,5000) for i  in range (500)]+[random.randint(2000,4000) for i  in range (508)]+[None,None],
    'Churn':['No','Yes','Yes','No','No']+[random.choice(['No','Yes']) for i in range(499)]+['Cjo','sfsda']+['Yes' for i in range(509)]
}
dataSET=pd.DataFrame(CommLinkTelecomData)
dataSET.to_csv('CommLink_Telecom_Customer_Data.csv',index=False) #Saving to csv file
dataSET.tail()

#2) Data Cleaning: Handle missing values, duplicates, and any inconsistencies in the dataset.
dataClean=pd.read_csv('CommLink_Telecom_Customer_Data.csv')
dataClean.isnull().sum()  #Checking fo null/missing values

dataClean.dropna(inplace=True) #Dropping null values
dataClean.isnull().sum() #Confirming no more null values left

duplicates=dataClean.duplicated(subset='CustomerID',keep=False) #Checking for duplicate values (Only CustomerID should be unique)
dataClean[duplicates] #Printing duplicates

dataClean.drop_duplicates(subset=['CustomerID'],inplace=True) #Dropping duplicates

duplicates=dataClean.duplicated(subset='CustomerID',keep=False) # Confirming all duplicates removed
dataClean[duplicates] #Should print nothing

dataClean.describe() #Finding inconsistencies in data

# As seen above there is a negative value,-10 in ServiceLength col which is inconsistent and should be removed
dataClean=dataClean.loc[dataClean['ServiceLength (months)']>=0]
dataClean.describe() #Confiriming inconsistent values in numerical columns are removed

dataClean['ContractType'].value_counts()  #Checking for inconsistent values in categorical columns

dataClean['Gender'].value_counts()

dataClean['Churn'].value_counts() #Churn column values can be only Yes or No, but there seems to be other values too

dataClean=dataClean.loc[(dataClean['Churn']=='Yes') | (dataClean['Churn']=='No')] #Removing inconsistent values in Churn col
dataClean['Churn'].value_counts() #Confirming removal

dataClean.info() #Confirming size of data after cleaning

dataClean.to_csv('CommLink_Telecom_Customer_Data_Cleaned.csv',index=False) #Saving Cleaned data for later use

#3) Data Preprocessing: Encode categorical variables, perform feature scaling, and handle skewed data.

dataPreprocessed=pd.read_csv('CommLink_Telecom_Customer_Data_Cleaned.csv')

#Applying one-hot encoding on all categorical Columns:   [ENCODE]
CatCols=['Gender','Churn','ContractType']
dataPreprocessed=pd.get_dummies(dataPreprocessed,columns=CatCols)

#Performing min-max scaling on all relevant numerical columns  [FEATURE SCALING]
from sklearn.preprocessing import MinMaxScaler
numCols=['Age','ServiceLength (months)','MonthlyCharges (USD)','TotalCharges (USD)']
scaler=MinMaxScaler()
dataPreprocessed[numCols]=scaler.fit_transform(dataPreprocessed[numCols])

dataPreprocessed.head()

#Handling skewed Data [SKEWED DATA]
from scipy import stats
# Checking the skewness of each numeric column using stats
skewness = dataPreprocessed[numCols].apply(lambda x: stats.skew(x))
print(skewness)

#Applying logarithm transformation on all columns that have high skewness i.e their absolute value of skewness is > 0.5 
import numpy as np
for col in numCols:
    if skewness[col] > 0.5:
        print('Applying logarithm transformation on column: ',col)
        dataPreprocessed[col] = np.log1p(dataPreprocessed[col])

dataPreprocessed.to_csv('CommLink_Telecom_Customer_Data_Preprocessed.csv',index=False) #Saving preprocessed data for later use

#4)  Exploratory Data Analysis: Explore the dataset using Python libraries 
# (e.g., Pandas, Matplotlib, Seaborn) to understand factors contributing to churn.
data=pd.read_csv('CommLink_Telecom_Customer_Data_Preprocessed.csv')
data2=pd.read_csv('CommLink_Telecom_Customer_Data_Cleaned.csv')
import matplotlib.pyplot as plt
data.describe()

#Visualizing age distribution
dataChurned=data2.loc[data2['Churn']=='Yes']
plt.hist(x=dataChurned['Age'],color='Cyan')
plt.title('Age Distribution of Churned Customers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

More of the churned customers belong to age group 20-40.


#Visualizing MonthlyCharges (USD) distribution
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.histplot(data=dataChurned, x='MonthlyCharges (USD)', kde=True, bins=30,color='Green')
plt.title('MonthlyCharges (USD) Distribution of Churned Customers')
plt.xlabel('MonthlyCharges (USD)')
plt.show()

Most of the churned users monthly charges are around 65 dollars. The company can reduce these to make sure they do not lose more customers. 

# Visualizing TotalCharges (USD) distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=dataChurned, x='TotalCharges (USD)', kde=True, bins=30,color='Purple')
plt.title('TotalCharges (USD) Distribution of Churned Customers')
plt.xlabel('TotalCharges (USD)')
plt.show()

Total Charges are mainly around $3000 which can be reduced by the company to retain customers.

#Visualizing ServiceLength (months) distribution
plt.hist(x=dataChurned['ServiceLength (months)'],color='Magenta')
plt.title('ServiceLength (months) Distribution of Churned Customers')
plt.ylabel('Count')
plt.xlabel('ServiceLength (months)')
plt.show()

Most of churned users had service for around 20 months.

#Visualizing Gender Distribution
dataC=data.loc[data['Churn_Yes']==1]
countGender=[dataC['Gender_Male'].sum(),dataC['Gender_Female'].sum()]
plt.pie(x=countGender,labels=['Male','Female'],autopct='%1.1f%%',colors=['Blue','Pink'])
plt.title('Gender Distribution of Churned Customers')
plt.legend()
plt.axis('equal')
plt.show()

There are way more Male churned cutomers than female

#Visualizing Contract Type Distribution
countContractType=[dataC['ContractType_One-Year'].sum(),dataC['ContractType_Two-Year'].sum(),dataC['ContractType_Month-to-Month'].sum()]
plt.pie(x=countContractType,labels=['One-Year','Two-Year','Month-to-Month'],autopct='%1.1f%%',colors=['Purple','Pink','Magenta'])
plt.title('Contract Type Distribution of Churned Customers')
plt.legend()
plt.axis('equal')
plt.show()

The customers who left used mostly the One-Year contract. 

#Visualizing Churn Distribution
countChurn=[data['Churn_Yes'].sum(),data['Churn_No'].sum()]
plt.pie(x=countChurn,labels=['Yes','NO'],autopct='%1.1f%%',colors=['Red','Green'])
plt.title('Churn Distribution')
plt.legend()
plt.axis('equal')
plt.show()

There are around 72 % churned customers. A very high number for any company.

# Counting the number of churned and non-churned customers [ALTERNATIVE OPTION TO ABOVE]
sns.countplot(x='Churn_Yes', data=data,palette='seismic')
plt.title('Churn Count')
plt.show()

# Calculate the churn rate
churn_rate = data['Churn_Yes'].value_counts(normalize=True)
print(churn_rate)

Around 700 customers out of 1000 have churned

# Plotting the heatmap to visualize correlations
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot=True, cmap='RdGy')
plt.title('Correlation Heatmap')
plt.show()

CommLink Telecom's churned customers seem to have been mainly male and had one-year contract.

# Relationship between Gender and Churn
sns.countplot(x='Gender', hue='Churn', data=data2,palette='viridis')
plt.title('Churn by Gender')
plt.show()

Most of CommLink Telecom's female customers have churned, however mainly the churned customers are male as the company had overall more male customers.

# Relationship between ContractType and Churn
sns.countplot(x='ContractType', hue='Churn', data=data2,palette='RdGy')
plt.title('Churn by Contract Type')
plt.show()

CommLink Telecom's customers having month-to-month contract were equally likely to churn. However more one-year contract type customer's churned.

# Relationship between Age and Churn
sns.boxplot(x='Churn', y='Age', data=data2,palette='gnuplot')
plt.title('Churn by Age')
plt.show()

CommLink Telecom's customers with age range 20-40 have mainly churned. It suggests they need to target people in this age range and attract them long enough.

# Relationship between ServiceLength and Churn
sns.boxplot(x='Churn', y='ServiceLength (months)', data=data2,palette='nipy_spectral')
plt.title('Churn by Service Length')
plt.show()

CommLink Telecom's churned cutomers had service length of around 40-50 months.

# Relationship between MonthlyCharges and Churn
sns.boxplot(x='Churn', y='MonthlyCharges (USD)', data=data2,palette='rainbow')
plt.title('Churn by Monthly Charges')
plt.show()


CommLink Telecom's customers that churned had high monthly prices than those that didn't. It suggests that they need to lower the prices to retain customers in the future. Churned customers had monthly charges mainly around $70

# Relationship between TotalCharges and Churn
sns.boxplot(x='Churn', y='TotalCharges (USD)', data=data2,palette='RdBu')
plt.title('Churn by Total Charges')
plt.show()


CommLink Telecom's churned customer had on average, hight total charges of around $3000. Again suggesting that the company's high prices might be the contributing factor to customers losing interest.

#5) Churn Prediction: Build a machine learning model (e.g., Logistic Regression, Random Forest) 
# to predict customer churn based on historical data.
# Tools: Python, Jupyter Notebook, Pandas, Matplotlib, Seaborn, Scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATASET=pd.read_csv('CommLink_Telecom_Customer_Data_Preprocessed.csv')

# Splitting the data into training and testing sets
X = DATASET.drop(['Churn_Yes','Churn_No'], axis=1) #These attributes are not features
y = DATASET['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #80 % training & 20 % testing

# Using Logistic Regression 
logisticModel = LogisticRegression(solver='liblinear',random_state=42)

# Using Random Forest
rfModel = RandomForestClassifier(random_state=42)

# Training the models
logisticModel.fit(X_train, y_train)
rfModel.fit(X_train, y_train)

# Evaluating the models
def evaluateModel(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print("\nAccuracy:", round(accuracy,2),'%')
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n\n", class_report)

print("\n\n-----------------------------------> Logistic Regression Model Results:\n")
evaluateModel(logisticModel, X_test, y_test)

print("\n\n-----------------------------------> Random Forest Model Results:\n")
evaluateModel(rfModel, X_test, y_test)

import pickle
# Saving Logistic Regression model
with open('logistic_model.pkl', 'wb') as file:
    pickle.dump(logisticModel, file)

# Saving Random Forest model
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rfModel, file)


#6) Deployment: Deploy the churn prediction model as an API or integrate it into CommLink Telecom's system 
# to generate real-time churn predictions for individual customers. 
# This will allow them to take targeted actions to retain customers at risk of churn.
# # Client Name: CommLink Telecom
# Company Name: DataSense Solutions

from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd

app = Flask(__name__)

def loadModels():
    try:
        with open('logistic_model.pkl', 'rb') as file:
            logisticModel = pickle.load(file)
        with open('random_forest_model.pkl', 'rb') as file2:
            rfModel = pickle.load(file2)
        return logisticModel, rfModel
    except Exception as e:
        print("Error loading models:", str(e))
        return None, None

logisticModel, rfModel = loadModels()

newData = { #This helps user test data to be predicted with ease as it has all the columns user won't enter
    'CustomerID':[100090,10097,18876],
    'Gender': ['Male','Female','Female'],
    'Age': [35,45,89],
    'ServiceLength (months)': [12,90,78],
    'ContractType': ['One-Year','Month-to-Month','Two-Year'],
    'MonthlyCharges (USD)': [70,89,90],
    'TotalCharges (USD)': [800,900,890]
}    

# Function for data preprocessing
def preprocessData(customerData):
    # Merging customerData with newData [that is user enteres]
    newData['CustomerID'].append(customerData['CustomerID'])
    newData['Gender'].append(customerData['Gender'])
    newData['Age'].append(customerData['Age'])
    newData['ServiceLength (months)'].append(customerData['ServiceLength (months)'])
    newData['ContractType'].extend(customerData['ContractType'])
    newData['MonthlyCharges (USD)'].append(customerData['MonthlyCharges (USD)'])
    newData['TotalCharges (USD)'].append(customerData['TotalCharges (USD)'])

    
    newData2 = pd.DataFrame(newData) #Converting to dataframe 
    # Converting categorical variables into numerical format using one-hot encoding
    newData2 = pd.get_dummies(newData2, columns=['Gender', 'ContractType'])

    # Performing min-max scaling on all relevant numerical columns 
    numCols = ['Age', 'ServiceLength (months)', 'MonthlyCharges (USD)', 'TotalCharges (USD)']
    scaler = MinMaxScaler()
    newData2[numCols] = scaler.fit_transform(newData2[numCols])
    return newData2

# Will get model predictions & clarify them:
def getPredictions(model, data):
    predictions = model.predict(data)
    #Shows model results with more clarity
    ans=[]
    for i in range (len(predictions)): 
        if(predictions[i]==1):
            show='Customer ID '+str(data['CustomerID'][i])+' CHURN YES'
            print(show)
            ans.append(show)
        else:
            show2='Customer ID '+str(data['CustomerID'][i])+' CHURN NO'
            print(show2)
            ans.append(show2)
    #only last data is user entered so only that will be returned the rest is old test data
    ans=[ans[3]]
    return ans

@app.route('/', methods=['GET', 'POST'])
def predictChurn():
    if request.method == 'GET':
        # Rendering the HTML form for user input
        return render_template('churn_prediction_form.html')

    if request.method == 'POST':
        try:
            # Getting the user entered customer data from the request form
            customerData = {
                'CustomerID': request.form.get('customer_id'),
                'Gender': request.form.get('gender'),
                'Age': float(request.form.get('age')),
                'ServiceLength (months)': float(request.form.get('service_length')),
                'ContractType': [
                    value for value in [
                        request.form.get('contract_type'),
                        request.form.get('contract_type_1'),
                        request.form.get('contract_type_2')] if value],
                'MonthlyCharges (USD)': float(request.form.get('monthly_charges')),
                'TotalCharges (USD)': float(request.form.get('total_charges'))
            }

            # Preprocessing the data
            data = preprocessData(customerData)
            
            # Making predictions using the logistic regression model
            logisticPredictions = getPredictions(logisticModel, data)

            # Making predictions using the random forest model
            rfPredictions = getPredictions(rfModel, data)

            # Returning the predictions as a JSON response
            return jsonify({
                'Logistic Regression Predictions': logisticPredictions,
                'Random Forest Predictions': rfPredictions
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8000, use_reloader=False)


#7) Project Outcome: The final deliverables will include a Jupyter Notebook documenting the data analysis 
# and modeling process, a report summarizing insights into churn factors,
# and the deployed churn prediction model. 
# DataSense Solutions will provide recommendations for churn reduction strategies based on the model's insights.
