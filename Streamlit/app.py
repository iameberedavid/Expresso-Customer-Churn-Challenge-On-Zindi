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
    imputed_df = imputer.transform(df[numerical_features])

    # Scale the numerical features
    scaled_df = scaler.transform(imputed_df)

    # Convert the NumPy array to a pandas DataFrame
    scaled_df = pd.DataFrame(scaled_df, columns=numerical_features)

    # Encode the categorical feature
    categorical_df = pd.DataFrame(encoder.transform(df[categorical_features]), columns=categorical_features)

    # Concatenate the encoded categorical and scaled numerical features
    processed_df = pd.concat([categorical_df, scaled_df], axis=1)

    st.write('Columns in the DataFrame:', processed_df.columns.tolist())
    st.write('DataFrame Shape:', processed_df.shape)

    return processed_df

def main():
    st.set_page_config(page_title='Customer Churn Prediction App', page_icon='⚠', layout='wide')
    st.title('Customer Churn Prediction App')

    st.write("This app predicts whether a telecommunication network's customer will churn or not. "
             "It requires the following customer data to make predictions.")
    
    columns = st.columns(2)

    with columns[0]:
        st.markdown("""1. TENURE: The customer's duration on the network.""")
        st.markdown("""2. MONTANT: The customer's top-up amount.""")
        st.markdown("""3. FREQUENCE_RECH: The number of times the customer recharged.""")
        st.markdown("""4. REVENUE: Monthly income of each customer.""")
        st.markdown("""5. ARPU_SEGMENT: Customer's average income over 90 days.""")
        st.markdown("""6. FREQUENCE: The number of times the customer made an income.""")

    with columns[1]:
        st.markdown("""7. DATA_VOLUME: The customer's number of connections.""")
        st.markdown("""8. ON_NET: The customer's calls within Expresso network.""")
        st.markdown("""9. ORANGE: The customer's calls to Orange network.""")
        st.markdown("""10. TIGO: The customer's calls to Tigo network.""")
        st.markdown("""11. REGULARITY: The number of times the customer has been active for 90 days.""")
        st.markdown("""12. FREQ_TOP_PACK: The number of times the customer activated top pack packages.""")

    st.markdown("""
<style>
    body {
        background-color: #008080;  /* Blue background color */
        color: #000000;  /* Text color */
    }
</style>
    """,
    unsafe_allow_html=True)

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

        # Convert the user input to a DataFrame with a single row
        df = pd.DataFrame([input_data])

        # Preprocess the user input data
        processed_data = preprocess_data(df, imputer, encoder, scaler)

        # Make prediction using the loaded model
        prediction = model.predict(processed_data)

        # Display the prediction
        if prediction[0] == 1:
            st.write('This customer is likely to churn.')
        else:
            st.write('This customer is not likely to churn.')

if __name__ == '__main__':
    main()