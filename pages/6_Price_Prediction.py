import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.title('Price Prediction')

df = pd.read_csv('data/listings_clean.csv')
model = joblib.load('models/best_model.joblib')
X = df.drop(columns=['price', 'log_price'])

c1, c2, c3 = st.columns(3)
borough = c1.selectbox('Borough', sorted(df['neighbourhood_cleansed'].unique()))
room_type = c1.selectbox('Room type', df['room_type'].unique())
property_type = c1.selectbox('Property type', df['property_type'].unique())
accommodates = c2.number_input('Guests', 1, 16, 2)
bedrooms = c2.number_input('Bedrooms', 0, 10, 1)
beds = c2.number_input('Beds', 0, 20, 1)
bathrooms = c3.number_input('Bathrooms', 0.0, 10.0, 1.0, 0.5)
shared_bath = c3.checkbox('Shared bathroom')
superhost = c3.checkbox('Superhost')
minimum_nights = c3.number_input('Minimum nights', 1, 365, 2)

amenities = ['air_conditioning', 'dishwasher', 'elevator', 'gym', 'pool', 'free_parking', 'dedicated_workspace', 'bathtub']
selected = st.multiselect('Amenities', amenities)

if st.button('Predict price'):
    area = df[df['neighbourhood_cleansed'] == borough]

    row = X.median(numeric_only=True)
    row['latitude'] = area['latitude'].median()
    row['longitude'] = area['longitude'].median()
    row['distance_centre'] = area['distance_centre'].median()
    row['accommodates'] = accommodates
    row['bedrooms'] = bedrooms
    row['beds'] = beds
    row['bathrooms'] = bathrooms
    row['shared_bath'] = int(shared_bath)
    row['host_is_superhost'] = int(superhost)
    row['minimum_nights'] = minimum_nights
    for a in amenities:
        row['has_' + a] = int(a in selected)

    new_listing = pd.DataFrame([row])
    new_listing['neighbourhood_cleansed'] = borough
    new_listing['room_type'] = room_type
    new_listing['property_type'] = property_type
    new_listing = new_listing[X.columns]

    price = np.exp(model.predict(new_listing)[0])
    st.success('Predicted price: £' + str(round(price)) + ' per night')
    st.write('Values that are not in the form (reviews, availability, host listings) are filled with the median of the dataset.')
