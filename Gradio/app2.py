import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the ML components
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, 'ml_components.pkl')
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

    # Impute missing values
    imputed_df = imputer.transform(df)

    # Separate the categorical and numeric features
    numerical_features = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME',
                          'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY', 'FREQ_TOP_PACK']
    categorical_features = ['TENURE']

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
    prediction = model.predict(processed_df)
    if prediction[0] == 1:
        result = 'Customer is likely to churn.'
    else:
        result = 'Customer is not likely to churn.'
    return result

# Define Gradio inputs
input_components = [
    gr.inputs.Radio(choices=['K > 24 month', 'E 6-9 month', 'H 15-18 month', 'G 12-15 month',
                            'I 18-21 month', 'J 21-24 month', 'F 9-12 month', 'D 3-6 month'], label='TENURE'),
    gr.inputs.Number(label='MONTANT'),
    gr.inputs.Number(label='FREQUENCE_RECH'),
    gr.inputs.Number(label='REVENUE'),
    gr.inputs.Number(label='ARPU_SEGMENT'),
    gr.inputs.Number(label='FREQUENCE'),
    gr.inputs.Number(label='DATA_VOLUME'),
    gr.inputs.Number(label='ON_NET'),
    gr.inputs.Number(label='ORANGE'),
    gr.inputs.Number(label='TIGO'),
    gr.inputs.Number(label='REGULARITY'),
    gr.inputs.Number(label='FREQ_TOP_PACK')
]

# Define the Gradio Interface
iface = gr.Interface(
    fn=predict_churn,
    inputs=input_components,
    outputs="text",
    title='Customer Churn Prediction App',
    description="This app predicts whether a telecommunication network's customer will churn or not based on available data."
)

# Launch the Gradio interface
iface.launch()