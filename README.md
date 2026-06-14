# 🏥 CriticalCare AI — ICU Mortality Risk Predictor

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![AUC](https://img.shields.io/badge/AUC-0.8884-brightgreen)

> A machine learning system that predicts ICU patient mortality risk 
> using real clinical data — with full SHAP explainability.

## 🎯 Problem Statement
ICU mortality prediction helps doctors prioritize critical patients. 
This model predicts in-hospital death probability using 180+ clinical features.

## 🔬 Models Compared
| Model | AUC Score |
|-------|-----------|
| Random Forest | ~0.84 |
| XGBoost | ~0.87 |
| **LightGBM** ✅ | **0.8884** |

## 🧠 Key Features
- ✅ Full EDA with missing value analysis
- ✅ 3-model comparison (RF, XGBoost, LightGBM)
- ✅ SHAP explainability (Beeswarm + Waterfall plots)
- ✅ Interactive Streamlit web app
- ✅ Real ICU clinical dataset (91,000+ patients)

## 🚀 Run Locally
pip install -r requirements.txt
streamlit run app/app.py

## 🌐 Live Demo
[CriticalCare AI →](https://criticalcare-ai-4z822c62f93ndcfoyd3ndn.streamlit.app)