# Load key libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import catboost
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Useful functions
def load_ml_components(fp):
    'Load the ML component to re-use in app'
    with open(fp, 'rb') as f:
        object = pickle.load(f)
        return object

# Variables and Constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, 'Assets', 'ml_components.pkl')

# Load the Machine Learning components
ml_components_dict = load_ml_components(fp=ml_core_fp)

# Extract the ML components
imputer = ml_components_dict['imputer']
scaler = ml_components_dict['scaler']
encoder = ml_components_dict['encoder']
model = ml_components_dict['model']

# Preprocess the data
def preprocess_data(df, imputer, encoder, scaler):
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

    st.write('Columns in the DataFrame:', processed_df.columns.tolist())
    st.write('DataFrame Shape:', processed_df.shape)

    return processed_df

def main():
    st.set_page_config(page_title='Customer Churn Prediction App', page_icon='📊', layout='wide')
    st.title('Customer Churn Prediction App')
    st.markdown(
    """
<style>
    body {
        background-color: #008080;  /* Blue background color */
        color: #000000;  /* Text color */
    }
</style>
    """,
    unsafe_allow_html=True
)


    # User input section
    TENURE = st.selectbox('TENURE', ['K > 24 month', 'E 6-9 month', 'H 15-18 month', 'G 12-15 month',
                                     'I 18-21 month', 'J 21-24 month', 'F 9-12 month', 'D 3-6 month'])
    MONTANT = st.number_input('MONTANT', min_value=0, step=1)
    FREQUENCE_RECH = st.number_input('FREQUENCE_RECH', min_value=0, step=1)
    REVENUE = st.number_input('REVENUE', min_value=0, step=1)
    ARPU_SEGMENT = st.number_input('Transactions', min_value=0, step=1)
    FREQUENCE = st.number_input('FREQUENCE', min_value=0, step=1)
    DATA_VOLUME = st.number_input('DATA_VOLUME', min_value=0, step=1)
    ON_NET = st.number_input('ON_NET', min_value=0, step=1)
    ORANGE = st.number_input('ORANGE', min_value=0, step=1)
    TIGO = st.number_input('TIGO', min_value=0, step=1)
    REGULARITY = st.number_input('REGULARITY', min_value=0, step=1)
    FREQ_TOP_PACK = st.number_input('FREQ_TOP_PACK', min_value=0, step=1)
							
    # Predict button
    if st.button('Predict Customer Churn'):
        # Prepare input data for prediction
        input_data = {
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
            'FREQ_TOP_PACK': [FREQ_TOP_PACK],
        }

        df = pd.DataFrame(input_data)

        # Preprocess the input data
        df = preprocess_data(df, imputer, scaler, encoder)

        # Make predictions using the loaded model
        prediction = model.predict(df)
        
        # Display the prediction
        if prediction[0] == 1:
            st.write("Customer is likely to churn.")
        else:
            st.write("Customer is not likely to churn.")

if __name__ == '__main__':
    main()