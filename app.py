import streamlit as st
#import tensorflow as tf
#from tensorflow import keras
#from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from keras import models
import pickle
import pandas as pd
import numpy as np

##Load model, scaler pickle file and label encoder pickle file
model=models.load_model('model.h5')
#Load encoder and scalar
with open('label_encode_gender.pkl','rb') as file:
    label_encode_gender=pickle.load(file)
with open('Onehot_encode_geography.pkl','rb') as file:
    Onehot_encode_geography=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

##streamlit app
st.title("Customer Churn Prediction")

#Take inputs from user
geography=st.selectbox('Geography',Onehot_encode_geography.categories_[0])  #
gender=st.selectbox('Gender', label_encode_gender.classes_)  
age=st.slider('Age',18,100)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_credit_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

#Prepare the input data
input_data=pd.DataFrame({'CreditScore':[credit_score],'Gender':[label_encode_gender.transform([gender])[0]],
    'Age':[age],'Tenure':[tenure],'Balance':[balance],'NumOfProducts':[num_of_products],
    'HasCrCard':[has_credit_card],'IsActiveMember':[is_active_member],'EstimatedSalary':[estimated_salary]
})

#one hot encode geography
geography_encoded=Onehot_encode_geography.transform([[geography]]).toarray()
geography_encoded_df=pd.DataFrame(geography_encoded,columns=Onehot_encode_geography.get_feature_names_out(['Geography']))

#concatenate geography data and input data
input_data=pd.concat([input_data,geography_encoded_df],axis=1) #reset_index(drop=True)

#Scale the input data
input_data_scaled=scaler.transform(input_data)

#Predictions
prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]
st.write(f"The churn prediction score is: {prediction_prob:.2f}")

if prediction_prob>0.5:
    st.write("The customer is likely to churn") 
else:
    st.write("The customer is not likely to churn")

