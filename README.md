## 📈 Stock Market Forecasting using ARIMA

This is a Streamlit-based web application that forecasts future stock prices using the ARIMA (AutoRegressive Integrated Moving Average) model. It allows users to input custom ARIMA parameters, select stocks, and define forecast horizons. The app also evaluates forecasting accuracy and provides export options in CSV, PDF, and DOCX formats.

🔧 Features
Interactive UI for stock selection (e.g., AAPL, MSFT, TCS.NS)

Custom date range and forecast duration (7–90 days)

Manual ARIMA (p, d, q) parameter tuning

Real-time data fetching from Yahoo Finance

Forecast visualization with historical trends

Forecast accuracy metrics: RMSE and MAE

Download forecast results as:

📄 CSV

📄 PDF

📄 DOCX

🛠️ Technologies Used
Python

Streamlit

yFinance (for real-time stock data)

Statsmodels (ARIMA model)

Matplotlib & Pandas (data processing & plotting)

Scikit-learn (accuracy metrics)

FPDF, python-docx (document generation)

