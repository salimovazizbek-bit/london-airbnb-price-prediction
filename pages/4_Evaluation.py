import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split

st.title('Model Evaluation')

val_results = pd.read_csv('models/val_results.csv', index_col=0)
test_results = pd.read_csv('models/test_results.csv', index_col=0)
importance = pd.read_csv('models/feature_importance.csv', index_col=0)

st.subheader('Validation set')
st.dataframe(val_results)

st.subheader('Test set')
st.dataframe(test_results)

metric = st.selectbox('Metric', ['MAE', 'RMSE', 'R2', 'MAPE'])
fig = px.bar(test_results, x=test_results.index, y=metric, title=metric + ' on test set')
st.plotly_chart(fig)

st.subheader('Best model on the test set')
df = pd.read_csv('data/listings_clean.csv')
X = df.drop(columns=['price', 'log_price'])
y = df['log_price']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = joblib.load('models/best_model.joblib')
result = pd.DataFrame({'Actual': np.exp(y_test), 'Predicted': np.exp(model.predict(X_test))})
result['Error'] = result['Actual'] - result['Predicted']

fig = px.scatter(result.sample(3000, random_state=42), x='Actual', y='Predicted', opacity=0.4, title='Actual vs Predicted')
st.plotly_chart(fig)

fig = px.histogram(result, x='Error', nbins=100, title='Residuals')
st.plotly_chart(fig)

st.subheader('Feature importance (Random Forest)')
fig = px.bar(importance.sort_values('importance'), x='importance', orientation='h')
st.plotly_chart(fig)
