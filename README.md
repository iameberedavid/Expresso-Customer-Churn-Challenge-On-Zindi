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

Welcome to the **Expresso Customer Churn Challenge On Zindi**. The primary goal of this challenge is to analyze the customer churn data of Expresso, a prominent telecommunication network. Afterwards, a Machine Learning model will be trained to predict customer churn for the company using the customer churn data. This will help the company monitor the churn tendencies of their customers, enabling them to know in advance the customers that are likely to churn, and take steps to retain these customers. The model will be used to make predictions on a test dataset that it has not seen before. The predictions, along with the user_ids of the customers, will be saved as a csv file and submitted on the Zindi challenge. The trained model, along with other Machine Learning components (such as the imputer, encoder, and scaler) used to prepare the customer churn data for modelling will be exported using pickle. They will then be used to build user-friendly apps using Gradio and Streamlit libraries, as well as a FastAPI web interface. These will simplify the usage of the model to make customer churn predictions for new customers based on their usage of the telecommunication nework. These apps and API will be dockerized and deployed to Huggingface to make them publicly available.

## Setup