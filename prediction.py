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

#Load the data- Example
input_data={'CreditScore':600,
                         'Geography':'France',
                         'Gender':'Male',
                         'Age':40,
                         'Tenure':3,
                         'Balance':60000,
                         'NumOfProducts':2,
                         'HasCrCard':1,
                         'IsActiveMember':1,
                         'EstimatedSalary':50000}
input_data_df=pd.DataFrame(input_data,index=[0])
print(input_data_df.head())

#Onehot encode 'Geography' column
geo_encode=Onehot_encode_geography.transform([[input_data['Geography']]]).toarray()
geo_encode_df=pd.DataFrame(geo_encode,columns=Onehot_encode_geography.get_feature_names_out(['Geography']))

print(geo_encode_df.head())

#Label encode
input_data_df['Gender']=label_encode_gender.transform([input_data['Gender']])

#concatination of geogfraphy data and input data
input_data_df=pd.concat([input_data_df.drop("Geography", axis=1),geo_encode_df],axis=1)
print(input_data_df.head())

#Scaling the input data
input_data_scaled=scaler.transform(input_data_df)

#Predictions
prediction=model.predict(input_data_scaled)
print("The churn prediction score is-",prediction)
prediction_prob=prediction[0][0]

if prediction_prob>0.5: 
    print("The customer is likely to churn")
else:
    print("The customer is not likely to churn")
