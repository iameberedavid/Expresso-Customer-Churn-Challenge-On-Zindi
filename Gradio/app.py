# Load key libraries
import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os
import catboost
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler


def predict_churn(TENURE, MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT, FREQUENCE, DATA_VOLUME, ON_NET,
                  ORANGE, TIGO, REGULARITY, FREQ_TOP_PACK):

    # Load the ML components
    DIRPATH = os.path.dirname(os.path.realpath(__file__))
    ml_core_fp = os.path.join(DIRPATH, 'ml_components.pkl')
    with open(ml_core_fp, "rb") as f:
        loaded_model_components = pickle.load(f)
    
    # Extract the ML components
    imputer = ml_components_dict['imputer']
    scaler = ml_components_dict['scaler']
    encoder = ml_components_dict['encoder']
    model = ml_components_dict['model']

    numerical_features = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME',
                          'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY', 'FREQ_TOP_PACK']
    categorical_features = ['TENURE']

    # Impute missing values
    imputed_df = imputer.transform(df)

    # Separate the categorical and numeric features
    numerical_df = imputed_df[numerical_features]
    categorical_df = imputed_df[categorical_features]

    # Scale the numeric features
    scaled_df = scaler.transform(numerical_df)

    # Encode the categorical features
    encoded_df = encoder.transform(categorical_df)

    # Concatenate the encoded categorical and scaled numeric features
    processed_df = np.concatenate((scaled_df, encoded_df), axis=1)

    # Get the names of columns after preprocessing
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(encoded_feature_names)

    # Convert the NumPy array to a pandas DataFrame
    processed_df = pd.DataFrame(processed_df, columns=all_feature_names)
    
    # Make prediction using the loaded model
    prediction = model.predict_proba(processed_df)
    prob_churn = float(prediction[0][1])
    return {'Churn Probability': prob_churn}

# Define the inputs
TENURE = gr.Radio(choices=['K > 24 month', 'E 6-9 month', 'H 15-18 month', 'G 12-15 month',
                           'I 18-21 month', 'J 21-24 month', 'F 9-12 month', 'D 3-6 month'], label='TENURE')
MONTANT = gr.Number(label='MONTANT')
FREQUENCE_RECH = gr.Number(label='FREQUENCE_RECH')
REVENUE = gr.Number(label='REVENUE')
ARPU_SEGMENT = gr.Number(label='ARPU_SEGMENT')
FREQUENCE = gr.Number(label='FREQUENCE')
DATA_VOLUME = gr.Number(label='DATA_VOLUME')
ON_NET = gr.Number(label='ON_NET')
ORANGE = gr.Number(label='ORANGE')
TIGO = gr.Number(label='TIGO')
REGULARITY = gr.Number(label='REGULARITY')
FREQ_TOP_PACK = gr.Number(label='FREQ_TOP_PACK')

# Design the interface
gr.Interface(inputs=[TENURE, MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT, FREQUENCE, DATA_VOLUME, ON_NET,
                     ORANGE, TIGO, REGULARITY, FREQ_TOP_PACK],
    outputs=gr.Label('Awaiting Submission...'),
    fn=predict_churn,
    title='Customer Churn Prediction App',
    description="This app predicts whether a telecommunication network's customer will churn or not based on available data."
).launch(inbrowser=True, show_error=True, share=True)