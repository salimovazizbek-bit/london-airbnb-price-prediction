import pandas as pd
import plotly.express as px
import streamlit as st

st.title('Data Exploration')

df = pd.read_csv('data/listings_clean.csv')
numeric_cols = df.select_dtypes(include='number').columns.tolist()

st.write('Shape of the dataset:', df.shape)

st.subheader('Summary statistics')
st.dataframe(df.describe().T)

st.subheader('Distribution')
col = st.selectbox('Select a column', numeric_cols, index=numeric_cols.index('price'))
fig = px.histogram(df, x=col, nbins=50)
st.plotly_chart(fig)

st.subheader('Price by category')
cat = st.selectbox('Select a category', ['room_type', 'property_type', 'neighbourhood_cleansed'])
fig = px.box(df, x=cat, y='price')
st.plotly_chart(fig)

st.subheader('Scatter plot')
x_col = st.selectbox('Select X', numeric_cols, index=numeric_cols.index('accommodates'))
fig = px.scatter(df.sample(3000, random_state=42), x=x_col, y='price', color='room_type', opacity=0.5)
st.plotly_chart(fig)

st.subheader('Correlation matrix')
selected = st.multiselect('Select columns', numeric_cols,
                          default=['price', 'accommodates', 'bedrooms', 'beds', 'bathrooms', 'distance_centre', 'number_of_reviews', 'review_scores_rating'])
if len(selected) > 1:
    fig = px.imshow(df[selected].corr().round(2), text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    st.plotly_chart(fig)
