Research on Machine Learning-Assisted Prediction of Electrochemical Performance of Magnesium-Lithium Alloys
===========================================================================================
This repository contains the datasets and figures supporting the manuscript submitted to Materials Letters.

Research Overview：
•	Dataset: Comprises approximately 110 data points collected from existing literature.
•	Objective: Modeling the electrochemical properties of Mg-Li alloys, specifically corrosion potential ( EcorrEcorr ) and corrosion current density ( icorricorr ).
•	Models: Four regression algorithms were employed: Gradient Boosting Decision Tree (GBDT), XGBoost, Random Forest (RF), and Support Vector Regression (SVR).
•	Optimization: Bayesian Optimization was utilized to tune hyperparameters and enhance model accuracy.
•	Interpretability: SHAP (SHapley Additive exPlanations) analysis was conducted to interpret model predictions and feature importance.

Reproduction Guide
•	It is recommended to execute the code in the Kaggle environment to ensure compatibility and ease of reproduction.

Data Sources：
•	The electrochemical data for Mg-Li alloys were curated from peer-reviewed publications indexed in ScienceDirect, Springer, and CNKI (China National Knowledge Infrastructure).
• All data provided in this repository are preprocessed and ready for direct use in model training.

Data and Code Availability
•	The full dataset and source code will be made publicly available upon acceptance of the manuscript.
•	This repository is currently provided exclusively for reproduction and verification purposes during the peer-review process.
