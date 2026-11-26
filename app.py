import pandas as pd
import streamlit as st

st.set_page_config(page_title='London Airbnb Price Prediction', layout='wide')

st.title('London Airbnb Price Prediction')
st.write('This app shows the machine learning project for predicting the nightly price of Airbnb listings in London.')
st.write('Data source: Inside Airbnb, London detailed listings (https://insideairbnb.com/get-the-data/).')

df = pd.read_csv('data/listings_clean.csv')
test_results = pd.read_csv('models/test_results.csv', index_col=0)
best = test_results['MAE'].idxmin()

col1, col2, col3, col4 = st.columns(4)
col1.metric('Listings after cleaning', len(df))
col2.metric('Median price', '£' + str(round(df['price'].median())))
col3.metric('Best model', best)
col4.metric('Test MAE', '£' + str(round(test_results.loc[best, 'MAE'])))

st.subheader('Pages')
st.write('1. Data Exploration: statistics, distributions and correlations')
st.write('2. Preprocessing: missing values, outliers and scaling')
st.write('3. Model Training: train a model with your own parameters')
st.write('4. Evaluation: comparison of the models on the test set')
st.write('5. Dashboard: overview of the London market')
st.write('6. Price Prediction: predict the price of a new listing')

st.subheader('Sample of the data')
st.dataframe(df.head(20))
