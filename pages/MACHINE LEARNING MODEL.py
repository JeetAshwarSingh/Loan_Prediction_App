import streamlit as st
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
logic = LogisticRegression(class_weight='balanced')
label=LabelEncoder()
scaler = StandardScaler()
df = pd.read_csv("data.csv")
df.drop(columns=['Loan_ID'],inplace = True)
columns_to_encode = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']

for column in columns_to_encode:
    df[column]=label.fit_transform(df[column])
credit = df['Credit_History']
loan  =df['Loan_Status']
df.drop(columns=['Credit_History'],inplace = True)
temp = scaler.fit_transform(df)
df = pd.DataFrame(temp)
df.columns = df.columns.astype(str)
df['Credit_History']=credit*10
df['Loan_Status']=loan
df.dropna(inplace=True)
X = df.drop(columns=['Loan_Status'])
Y = df["Loan_Status"]
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.80,random_state=42)
logic.fit(x_train,y_train)
y_prob_logic = logic.predict_proba(x_test)[:, 1]
y_predict_logic = (y_prob_logic > 0.6).astype(int)


st.title(":blue[MACHINE LEARNING MODEL]")
st.write('_Logistic Regression is used in this project as it is simple fast and powerful in making yes/no decisions like predicting whether a loan should be approved or not_')
st.write('_It works by analyzing patterns in past data (like income, credit history, and employment status) and estimating the likelihood of a successful loan repayment. What makes it especially useful is its interpretability, you can actually understand which factors influence decisions the most , which is critical when dealing with real people and financial fairness._')
st.subheader(''' :blue[ METRICS ]''')
st.write('_In this section few metrics which are used to measure the reliability of the model along with its result are discussed as_')
st.write("_Accuracy_score of the Model is_")
st.code(accuracy_score(y_test,y_predict_logic))
st.write("_Confusion_Matrix of the Model is_")
st.code(confusion_matrix(y_test,y_predict_logic))
st.write("_Recall_Score of the Model is_")
st.code(recall_score(y_test,y_predict_logic))
st.write("_Precision_Score of the Model is_")
st.code(precision_score(y_test,y_predict_logic))
st.subheader(''' :blue[DEMO OF THE MODEL]''')
st.write("Here is a Demo of the Model for anyone Interested")
gender = st.selectbox("Enter your Gender",['Male','Female'])
married = st.selectbox("Enter your Martial Status",['No','Yes'])
dependent = st.selectbox("How many People are Dependent over You",[0,1,2,'3+'])
education = st.selectbox("What is your Education",['Graduate','Not Graduate'])
employ = st.selectbox("Are you self Employed",['Yes','No'])
income = st.slider("What is your monthly income? (in Rupees)",0,81000,value=5450,step=50)
coincome = st.slider("What is Coapplicant's monthly income? (in Rupees)",0,42000,value=1650,step=50)
loan_amount = st.slider("What is loan amount you applying for",1,700,value=345,step=5)
st.write('1 = 1000 rupee unit')
loan_term = st.slider('What is the term you applying for (time period of loan in months)',0,480,value=1,step=1)

credits = st.selectbox("What Credit Score will you give Yourself",['Not Satisfactory','Satisfactory'])
if credits=='Not Satisfactory':
    credits=0
else:
    credits=1
area = st.selectbox("where are you planning to buy your house",['Urban','Suburban','Rural'])
try:
    user_data = pd.DataFrame({
        'Gender': ['Male'],
        'Married': ['Yes'],
        'Dependents': ['3+'],
        'Education': ['Graduate'],
        'Self_Employed': ['Yes'],
        'ApplicantIncome': [5000],
        'CoapplicantIncome': [0],
        'LoanAmount': [128],
        'Loan_Amount_Term': [360],
        'Credit_History': [0],
        'Property_Area': ['Urban'],
        'Loan_Status': ['Y']
    })
    for column in columns_to_encode:
        user_data[column]=label.fit_transform(user_data[column])
    credit1 = user_data['Credit_History']
    loan1  =user_data['Loan_Status']
    user_data.drop(columns=['Credit_History'],inplace = True)
    temp = scaler.fit_transform(user_data)
    user_data = pd.DataFrame(temp)
    user_data.columns = user_data.columns.astype(str)
    user_data['Credit_History'] = credit1.values*10
    # user_data['Loan_Status']=loan

    y_user_logic = logic.predict_proba(user_data)[:, 1]
    y_user_predict_logic = (y_user_logic > 0.6).astype(int)
    if st.button(":blue[GET RESULT]"):
        st.code(y_user_predict_logic)
except:
    st.write(':red[Fill all the boxes above]')
st.write(''':grey[This model has been trained on historical financial data and is intended for educational or experimental purposes only. The predictions it generates may not accurately reflect your personal financial situation or real-world lending decisions. Always consult with a certified financial advisor or lending institution before making any financial commitments or decisions based on this model’s output.]''')
