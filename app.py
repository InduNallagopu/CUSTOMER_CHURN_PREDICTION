import streamlit as st
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd 
import pickle

# ---------- Page Config ----------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:700;
    color:#4CAF50;
}
.result-box {
    padding:20px;
    border-radius:10px;
    font-size:20px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("<div class='big-title'>📊 Customer Churn Prediction System</div>", unsafe_allow_html=True)
st.write("Predict whether a customer is likely to churn based on input details.")

st.divider()



# Now load model
model = load_model('model.h5', compile=False)


#load encoders and scaler
with open('label_encoder_gender.pkl','rb') as file:
  label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
  onehot_encoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
  scaler=pickle.load(file)


#streamlit app
#st.title('Customer Churn Prediction')


#user inputs
geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

#prepare the input data
input_data=pd.DataFrame({
  'CreditScore':[credit_score],
  'Gender':[label_encoder_gender.transform([gender])[0]],
  'Age':[age],
  'Tenure':[tenure],
  'Balance':[balance],
  'NumOfProducts':[num_of_products],
  'HasCrCard':[has_cr_card],
  'IsActiveMember':[is_active_member],
  'EstimatedSalary':[estimated_salary]
})

if st.button("Predict Churn"):
  #one-hot encode geography
  geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
  geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

  #combine one-hot encoded geography with input data
  input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

  #scale the input data
  input_data_scaled=scaler.transform(input_data)

  #make prediction
  prediction=model.predict(input_data_scaled)
  prediction_proba=prediction[0][0]

  st.divider()
  st.subheader("📈 Prediction Result")

  st.write(f'Churn Probability: {prediction_proba:.2f}')

  # Progress bar
  st.progress(float(prediction_proba))

  if prediction_proba>0.5:
    st.error('⚠️ The customer is likely to churn')
  else:
    st.success('✅ The customer is not likely to churn')











