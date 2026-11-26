import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.title('Preprocessing')

df = pd.read_csv('data/listings_clean.csv')
numeric_cols = df.select_dtypes(include='number').columns.tolist()

st.subheader('Missing values')
missing = df.isnull().sum()
missing = missing[missing > 0]
st.dataframe(pd.DataFrame({'missing': missing, 'percent': (missing / len(df) * 100).round(2)}))

col = st.selectbox('Column to fill', missing.index.tolist())
method = st.radio('Method', ['median', 'mean', 'zero'], horizontal=True)

if method == 'median':
    filled = df[col].fillna(df[col].median())
elif method == 'mean':
    filled = df[col].fillna(df[col].mean())
else:
    filled = df[col].fillna(0)

c1, c2 = st.columns(2)
c1.metric('Mean before', round(df[col].mean(), 3))
c2.metric('Mean after', round(filled.mean(), 3))
st.plotly_chart(px.histogram(filled, nbins=40))

st.subheader('Outliers (IQR method)')
k = st.slider('IQR multiplier', 1.0, 3.0, 1.5, 0.5)
log_price = np.log(df['price'])
Q1 = log_price.quantile(0.25)
Q3 = log_price.quantile(0.75)
IQR = Q3 - Q1
lower = np.exp(Q1 - k * IQR)
upper = np.exp(Q3 + k * IQR)
kept = df[(df['price'] >= lower) & (df['price'] <= upper)]

c1, c2, c3 = st.columns(3)
c1.metric('Lower limit', '£' + str(round(lower)))
c2.metric('Upper limit', '£' + str(round(upper)))
c3.metric('Rows removed', len(df) - len(kept))
st.plotly_chart(px.box(kept, x='price'))

st.subheader('Scaling')
scale_col = st.selectbox('Column to scale', numeric_cols, index=numeric_cols.index('number_of_reviews'))
scaler_name = st.radio('Scaler', ['StandardScaler', 'MinMaxScaler'], horizontal=True)

if scaler_name == 'StandardScaler':
    scaler = StandardScaler()
else:
    scaler = MinMaxScaler()

values = df[[scale_col]].fillna(df[scale_col].median())
scaled = pd.DataFrame(scaler.fit_transform(values), columns=[scale_col])

c1, c2 = st.columns(2)
c1.write('Before scaling')
c1.dataframe(values.describe())
c2.write('After scaling')
c2.dataframe(scaled.describe())
