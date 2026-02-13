# Customer Churn Prediction using Artificial Neural Network (ANN)

## 🌐 Live Demo

[![Live Demo](https://img.shields.io/badge/Streamlit-Live_App-red?style=for-the-badge&logo=streamlit)](https://customerchurnprediction-ere652kv93imuewg89bnff.streamlit.app)



## Overview
Built an end-to-end deep learning solution to predict customer churn in the banking sector.  
The project implements a fully connected Artificial Neural Network (ANN) to classify whether a customer is likely to leave the bank based on demographic and financial attributes.

The solution includes data preprocessing, feature engineering, model training, serialization, and deployment using a Streamlit web application.

---

## Business Problem
Customer retention is significantly more cost-effective than customer acquisition.  
This project aims to help financial institutions proactively identify high-risk customers and enable data-driven retention strategies.

---

## Technical Implementation

### Model Architecture
- Fully Connected Artificial Neural Network
- Dense Hidden Layers with ReLU activation
- Sigmoid Output Layer for binary classification
- Adam Optimizer
- Binary Crossentropy Loss Function

### Data Engineering
- Categorical Encoding  
  - Label Encoding (Gender)  
  - One-Hot Encoding (Geography)  
- Feature Scaling using StandardScaler  
- Train-Test Split for evaluation  
- Model & preprocessing artifacts serialized for deployment  

---

## Deployment
The trained model is deployed through a Streamlit-based web interface that enables real-time churn prediction using user-provided inputs.

The deployment ensures:
- Consistent preprocessing pipeline
- Scalable inference
- Clean user interaction layer

---

## Tech Stack
- Python
- Pandas & NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Pickle

---
```
## Repository Structure

CUSTOMER_CHURN_PREDICTION/
│
├── app.py # Streamlit application
├── model.h5 # Trained ANN model
├── scaler.pkl # StandardScaler object
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── Churn_Modelling.csv
├── training.ipynb
├── prediction.ipynb
├── requirements.txt
└── runtime.txt

```
---

## Key Highlights
- End-to-End Deep Learning Pipeline  
- Structured Data Modeling using ANN  
- Production-ready Model Serialization  
- Real-world Banking Industry Use Case  
- Deployable ML Application  

---

## How to Run Locally

1.Clone the repository:
 git clone https://github.com/InduNallagopu/CUSTOMER_CHURN_PREDICTION.git

2.Install dependencies:
 pip install -r requirements.txt

3.Run the app:
 streamlit run app.py


---

## License
This project is licensed under the GPL-3.0 License.


