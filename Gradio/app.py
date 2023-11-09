# Load the key libraries
import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the ML components
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, 'Assets/ml_components.pkl')
with open(ml_core_fp, "rb") as f:
    ml_components_dict = pickle.load(f)

# Extract the ML components
imputer = ml_components_dict['imputer']
scaler = ml_components_dict['scaler']
encoder = ml_components_dict['encoder']
model = ml_components_dict['model']

# Define a Gradio function to make predictions
def predict_churn(TENURE, MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT, FREQUENCE, DATA_VOLUME, ON_NET,
                  ORANGE, TIGO, REGULARITY, FREQ_TOP_PACK):
    
    # Set 'K > 24 month' as the default selection as the imputer can only work on numerical values
    if TENURE is None:
        TENURE = 'K > 24 month'

    # Create a dictionary from the inputs
    data = {
        'TENURE': [TENURE],
        'MONTANT': [MONTANT],
        'FREQUENCE_RECH': [FREQUENCE_RECH],
        'REVENUE': [REVENUE],
        'ARPU_SEGMENT': [ARPU_SEGMENT],
        'FREQUENCE': [FREQUENCE],
        'DATA_VOLUME': [DATA_VOLUME],
        'ON_NET': [ON_NET],
        'ORANGE': [ORANGE],
        'TIGO': [TIGO],
        'REGULARITY': [REGULARITY],
        'FREQ_TOP_PACK': [FREQ_TOP_PACK]
    }

    # Create a DataFrame from the input data
    df = pd.DataFrame(data)

    # Separate the categorical and numerical features
    numerical_features = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME',
                          'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY', 'FREQ_TOP_PACK']
    categorical_features = ['TENURE']

    # Impute missing values
    imputed_df = imputer.transform(df[numerical_features])

    # Scale the numerical features
    scaled_df = scaler.transform(imputed_df)

    # Convert the NumPy array to a pandas DataFrame
    scaled_df = pd.DataFrame(scaled_df, columns=numerical_features)

    # Encode the categorical feature
    categorical_df = pd.DataFrame(encoder.transform(df[categorical_features]), columns=categorical_features)

    # Concatenate the encoded categorical and scaled numerical features
    processed_df = pd.concat([categorical_df, scaled_df], axis=1)
    
    # Make prediction using the loaded model
    prediction = model.predict(processed_df)
    if prediction[0] == 1:
        result = 'This customer is likely to churn.'
    else:
        result = 'This customer is not likely to churn.'
    return result

# Define Gradio inputs
TENURE = gr.Radio(choices=['K > 24 month', 'E 6-9 month', 'H 15-18 month', 'G 12-15 month', 'I 18-21 month',
             'J 21-24 month', 'F 9-12 month', 'D 3-6 month'], label='TENURE')
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
    title='Customer Churn Prediction',
    description="This app predicts whether a telecommunication network's customer will churn or not. "
                "It requires the following customer data to make predictions.\n"
                "<div style='display: flex; justify-content: space-between;'>"
                "<div style='flex: 1; margin-right: 10px;'>"
                "1. TENURE: The customer's duration on the network.<br>"
                "2. MONTANT: The customer's top-up amount.<br>"
                "3. FREQUENCE_RECH: The number of times the customer recharged.<br>"
                "4. REVENUE: Monthly income of each customer.<br>"
                "5. ARPU_SEGMENT: Customer's average income over 90 days.<br>"
                "6. FREQUENCE: The number of times the customer made an income."
                "</div>"
                "<div style='flex: 1;'>"
                "7. DATA_VOLUME: The customer's number of connections.<br>"
                "8. ON_NET: The customer's calls within Expresso network.<br>"
                "9. ORANGE: The customer's calls to Orange network.<br>"
                "10. TIGO: The customer's calls to Tigo network.<br>"
                "11. REGULARITY: The number of times the customer has been active for 90 days.<br>"
                "12. FREQ_TOP_PACK: The number of times the customer activated top pack packages."
                "</div>"
                "</div>"
).launch(inbrowser=True, show_error=True, share=True)