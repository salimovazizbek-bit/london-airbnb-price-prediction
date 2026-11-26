import pandas as pd
import plotly.express as px
import streamlit as st

st.title('London Airbnb Dashboard')

df = pd.read_csv('data/listings_clean.csv')

room_types = st.sidebar.multiselect('Room type', df['room_type'].unique().tolist(), default=df['room_type'].unique().tolist())
boroughs = st.sidebar.multiselect('Borough', sorted(df['neighbourhood_cleansed'].unique()))
price_range = st.sidebar.slider('Price range', 20, 1350, (20, 1350))

data = df[df['room_type'].isin(room_types)]
if len(boroughs) > 0:
    data = data[data['neighbourhood_cleansed'].isin(boroughs)]
data = data[(data['price'] >= price_range[0]) & (data['price'] <= price_range[1])]

if len(data) == 0:
    st.warning('No listings for the selected filters')
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric('Listings', len(data))
c2.metric('Median price', '£' + str(round(data['price'].median())))
c3.metric('Average rating', round(data['review_scores_rating'].mean(), 2))
c4.metric('Superhosts', str(round(data['host_is_superhost'].mean() * 100)) + '%')

st.subheader('Map of listings')
map_data = data.sample(min(5000, len(data)), random_state=42)
fig = px.scatter_map(map_data, lat='latitude', lon='longitude', color='price', zoom=9, height=500,
                     range_color=[30, 500], map_style='open-street-map')
st.plotly_chart(fig)

left, right = st.columns(2)

borough_price = data.groupby('neighbourhood_cleansed')['price'].median().sort_values().reset_index()
fig = px.bar(borough_price, x='price', y='neighbourhood_cleansed', orientation='h', height=700, title='Median price by borough')
left.plotly_chart(fig)

room_count = data['room_type'].value_counts().reset_index()
fig = px.pie(room_count, names='room_type', values='count', title='Listings by room type')
right.plotly_chart(fig)

guest_price = data.groupby('accommodates')['price'].median().reset_index()
fig = px.line(guest_price, x='accommodates', y='price', markers=True, title='Median price by number of guests')
right.plotly_chart(fig)
