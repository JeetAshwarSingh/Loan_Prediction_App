import streamlit as st
import pandas as pd
st.title('HOME PAGE')
st.header(':blue[ABOUT THIS PROJECT]',divider="blue")
st.text("The primary objective of this project is to develop an effective machine learning model capable of predicting whether a person's loan application will be approved or rejected. This initiative aims to support financial institutions in making more data-driven, consistent, and fair lending decisions.")
st.text("In addition to the predictive model, the project also offers valuable insights into the underlying data. Through exploratory data analysis, it explores the relationships between various applicant attributes—such as income, credit history, marital status, and more—and the loan approval status. These insights not only enhance the interpretability of the model but also help identify key factors that influence loan approval outcomes.")
st.header(':blue[DATASET]', divider='blue')
st.text("This dataset contains loan application for home loans recorded and collected by a financial institution. Each row represents the application of a potential borrower, and the objective is to predict whether the loan should be approved or not, based on the provided attributes.")
table = {'Loan_ID':'Unique identifier for each loan application'}
data = {
    "Feature": [
        "Loan_ID",
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
        "Loan_Status"
    ],
    "Description": [
        "Unique identifier for each loan application",
        "Applicant’s gender (Male, Female)",
        "Marital status (Yes, No)",
        "Number of dependents (0, 1, 2, 3+)",
        "Applicant’s education level (Graduate, Not Graduate)",
        "Whether the applicant is self-employed",
        "Monthly income of the applicant",
        "Monthly income of the co-applicant (if any)",
        "Requested loan amount (in thousands)",
        "Term of the loan (in months)",
        "Whether the applicant has a credit history (1 = good, 0 = bad)",
        "Area where the property is located (Urban, Rural, Semiurban)",
        "Target variable indicating loan approval (Y or N)"
    ]
}
df1 = pd.DataFrame(data)
st.dataframe(df1)
st.text('''Although which currency used is not mentioned but it could be speculated that it’s in INR because the data is retrieved from “Analytics Vidhya”.''')
st.header(':blue[MODEL USED FOR PREDICTION]', divider='blue')
st.image("pages/3_png.png",width=500)
st.write("_The model used for classification in this project is logistic Regression, as I am to predict whether one’s loan will be approved or not. Therefore it is a good fit for this dataset_")