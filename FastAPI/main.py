# Load the key libraries
from fastapi import FastAPI
from fastapi import HTTPException
from typing import Union, Literal
import pandas as pd
import pickle, os
from pydantic import BaseModel

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

# Display execution
print(f'\n[Info] ML components loaded: {list(ml_components_dict.keys())}')

# Extract the ML components
imputer = ml_components_dict['imputer']
scaler = ml_components_dict['scaler']
encoder = ml_components_dict['encoder']
model = ml_components_dict['model']

# Input Modelling
class ChurnPrediction(BaseModel):
    execution_msg: str
    execution_code: int
    predictions: Union[list, None]

    ## Input features
    TENURE: str
    MONTANT: float
    FREQUENCE_RECH: float
    REVENUE: float
    ARPU_SEGMENT: float
    FREQUENCE: float
    DATA_VOLUME: float
    ON_NET: float
    ORANGE: float
    TIGO: float
    REGULARITY: float
    FREQ_TOP_PACK: float

# APP
app = FastAPI(title='Customer Churn Prediction App')

@app.get('/')
async def root():
    return {
        "This app predicts whether a telecommunication network's customer will churn or not. "
    }

@app.post('/classify', response_model=ChurnPrediction)
async def predict_churn(data: ChurnPrediction):
    try:
        # Creation of Dataframe
        df = pd.DataFrame(
            {
                'TENURE': [data.TENURE],
                'MONTANT': [data.MONTANT],
                'FREQUENCE_RECH': [data.FREQUENCE_RECH],
                'REVENUE': [data.REVENUE],
                'ARPU_SEGMENT': [data.ARPU_SEGMENT],
                'FREQUENCE': [data.FREQUENCE],
                'DATA_VOLUME': [data.DATA_VOLUME],
                'ON_NET': [data.ON_NET],
                'ORANGE': [data.ORANGE],
                'TIGO': [data.TIGO],
                'REGULARITY': [data.REGULARITY],
                'FREQ_TOP_PACK': [data.FREQ_TOP_PACK]
            }
        )
        print(f'[Info] Input data as DataFrame:\n{df.to_markdown()}')
        
        # Separate the categorical and numerical features
        numerical_features = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE',
                              'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY', 'FREQ_TOP_PACK']
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
        
        # Calculate the confidence score by taking the maximum probability among predicted classes
        confidence_score = prediction.max(axis=-1)

        # Calculate confidence score percentage and round to 2 decimal places
        confidence_score_percentage = round(confidence_score * 100, 2)
        print(f'The confidence score is {confidence_score_percentage}%')

        # Extract the class label with the highest probability as the predicted label
        df['predicted_label'] = prediction.argmax(axis=-1)

        # Define a mapping from class labels to human-readable labels
        mapping = {0: 'the customer is not likely to churn', 1: ' the customer is likely to churn'}

        # Replace the numeric predicted labels with human-readable labels using the mapping
        df['predicted_label'] = [mapping[x] for x in df['predicted_label']]

        # Store the confidence score percentage in the 'confidence_score' column
        df['confidence_score'] = f'{confidence_score_percentage}%'

        # Print results
        print(f"The customer data reveals that : {df['predicted_label'].values[0]}.")
        msg = 'Execution went fine'
        code = 1
        pred = df.to_dict('records')

        result = {'execution_msg': 'Execution went fine', 'execution_code': 1, 'predictions': df.to_dict('records')}
        return result

    except Exception as e:
        # Error handling with HTTPException
        raise HTTPException(status_code=500, detail=f'Something went wrong during the Churn prediction: {str(e)}')

if __name__ == '__main__':
    uvicorn.run('main:app', reload=True)