import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

#Load the dataset
data=pd.read_csv("Churn_Modelling.csv")
print(data.head())
print(data.columns)
data=data.drop(['RowNumber','CustomerId', 'Surname'], axis=1)

#Label encode 'Gender' column
label_encode_gender=LabelEncoder()
data['Gender']=label_encode_gender.fit_transform(data['Gender'])
print(data.head())
print(data['Gender'].value_counts())

#OneHot Encode 'Geography' column
Onehot_encode_geography=OneHotEncoder()
geo_encode=Onehot_encode_geography.fit_transform(data[['Geography']])
print(Onehot_encode_geography.get_feature_names_out(['Geography']))
geo_encoded_df=pd.DataFrame(geo_encode.toarray(), columns=Onehot_encode_geography.get_feature_names_out(['Geography']))
print(geo_encoded_df.head())

#Combine One Hot Encoded columns with original data
data=pd.concat([data.drop('Geography', axis=1),geo_encoded_df], axis=1)
print(data.head())
print(data.columns)

#Save the encoder and scalar to pickle file for later 
with open('label_encode_gender.pkl','wb') as file:
    pickle.dump(label_encode_gender,file)
with open('Onehot_encode_geography.pkl', 'wb') as file:
    pickle.dump(Onehot_encode_geography,file)

#Divide the data in to independent and dependent features
X=data.drop('Exited',axis=1)
y=data['Exited']

#Split the data in training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Scale the features
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#Save scaler object as a pickle file
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler,file)

##ANN Implementation
#import libraries
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, TensorBoard
import datetime

#Build our ANN model
model=Sequential([
    Dense(64, activation='relu',input_shape=(X_train.shape[1],) ),  ## HL1 connected with input layer
    Dense(32,activation='relu'), ## HL2
    Dense(1,activation='sigmoid') ## Output layer --use softmax for multiclass classifications
    ])
print(model.summary())  ##Gives trainable parameter details (weights and biases)

#Compile the models
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=['accuracy'])

##Train the model
history=model.fit(
    X_train,y_train,validation_data=(X_test,y_test), epochs=100
)

#Save the model
model.save('model.h5')

