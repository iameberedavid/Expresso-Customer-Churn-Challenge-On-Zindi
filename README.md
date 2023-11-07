# EXPRESSO CUSTOMER CHURN CHALLENGE ON ZINDI

## Technologies

![Data Analysis](https://img.shields.io/badge/Data-Analysis-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-blue)
![App Development](https://img.shields.io/badge/App-Development-blue)
![Containerization](https://img.shields.io/badge/Containerization-blue)
![DevOps](https://img.shields.io/badge/DevOps-blue)
![MIT](https://img.shields.io/badge/MIT-License-blue?style=flat)

![Python](https://img.shields.io/badge/Python-3.11-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27.2-brightgreen)
![Gradio](https://img.shields.io/badge/Gradio-3.50.2-brightgreen)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-brightgreen)
![HuggingFace](https://img.shields.io/badge/HuggingFace-0.17.3-brightgreen)
![Docker](https://img.shields.io/badge/Docker-24.0.6-brightgreen)


## Project Description

Welcome to the **Expresso Customer Churn Challenge On Zindi**. The aim of this challenge is to analyze the customer churn data of Expresso, a prominent telecommunication network. Afterwards, Machine Learning models will be trained to predict customer churn for the company using the customer churn data. The best model will be selected based on the AUC scores of the trained models. This best model will be used to make predictions on new customer data that it has not seen before. This will help the company monitor the churn tendencies of their customers, enabling them to know in advance the customers that are likely to churn, and make informed business decisions aimed at retaining these customers. The predictions, along with the user_ids of the customers on the new customer data, will be saved as a csv file for submission on Zindi.

The best model, along with other Machine Learning components (such as the imputer, encoder, and scaler) used to prepare the customer churn data for modelling will be exported using pickle. They will then be used to build user-friendly apps using Gradio and Streamlit libraries, as well as a FastAPI web interface. These will simplify the usage of the model to make customer churn predictions for new customers based on their available data. Finally, these apps and API will be containerized using docker and deployed to Huggingface to make them publicly available.

## Preview 🔍

Below is a preview showcasing some features of the notebook.

<div style="display: flex; align-items: center;">
    <div style="flex: 33.33%; text-align: center;">
        <p>Overall Churn Rate Of The Telecommunication Network</p>
        <img src="Images/Readmepics/Overall Churn Rate Of The Telecommunication Network.png" alt="Top" width="90%"/>
    </div>
    <div style="flex: 33.33%; text-align: center;">
        <p>Churn Rate by Tenure</p>
        <img src="Images/Readmepics/Churn Rate by Tenure.png" alt="Middle" width="90%"/>
        </div>
    <div style="flex: 33.33%; text-align: center;">
        <p>Churn Rate by Region</p>
        <img src="Images/Readmepics/Churn Rate by Region.png" alt="Middle" width="90%"/>
        </div>
    <div style="flex: 33.33%; text-align: center;">
        <p>Confusion Matrix of the Best Model</p>
        <img src="Images/Readmepics/Confusion Matrix of the Best Model.png" alt="Middle" width="90%"/>
    </div>
    <div style="flex: 33.33%; text-align: center;">
        <p>Precision-Recall Curve of the Best Model</p>
        <img src="Images/Readmepics/Precision-Recall Curve of the Best Model.png" alt="Bottom" width="90%"/>
    </div>
</div>

## Prerequisites for the Apps and Web API

Ensure that you install the following libraries in your Python environment or virtual environment:

* Gradio
* Streamlit
* FastAPI
* Tabulate
* Uvicorn
* Pandas
* Logistic Regression

The libraries can be installed using the following command:

![Installations](Images/Readmepics/Installations.png)

## Setup

To set up and run the apps in your local environment, follow these instructions:

1. Clone this repository to your local machine using the following command. Replace \<repository-url\> with the actual url to this repository:

![Clone](Images/Readmepics/Clone.png)

2. Create and activate a virtual environment:

![venv](Images/Readmepics/venv.png)

3. Install requirements.txt:

![Requirements](Images/Readmepics/Requirements.png)

4. Run the apps using the following commands:

- For Gradio

![Run](Images/Readmepics/Gradio run.png)

- For Streamlit

![Run](Images/Readmepics/Streamlit run.png)

- For FastAPI

![Run](Images/Readmepics/FastAPI run.png)

Each of the apps will be launched in your default web browser and can then be used to make predictions based on the customer information provided.