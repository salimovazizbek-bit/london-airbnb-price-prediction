import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

st.title('Model Training')

df = pd.read_csv('data/listings_clean.csv')
X = df.drop(columns=['price', 'log_price'])
y = df['log_price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

num_features = X.select_dtypes(include='number').columns.tolist()
cat_features = X.select_dtypes(exclude='number').columns.tolist()

numeric_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

categorical_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocess = ColumnTransformer([
    ('num', numeric_pipe, num_features),
    ('cat', categorical_pipe, cat_features)
])

model_name = st.selectbox('Model', ['Linear Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost'])
n_rows = st.slider('Number of training rows', 2000, len(X_train), 10000, 1000)

if model_name == 'Linear Regression':
    model = LinearRegression()
elif model_name == 'KNN':
    k = st.slider('n_neighbors', 1, 50, 11)
    weights = st.selectbox('weights', ['uniform', 'distance'])
    model = KNeighborsRegressor(n_neighbors=k, weights=weights)
elif model_name == 'Decision Tree':
    depth = st.slider('max_depth', 2, 30, 10)
    leaf = st.slider('min_samples_leaf', 1, 50, 10)
    model = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=42)
elif model_name == 'Random Forest':
    trees = st.slider('n_estimators', 20, 200, 50, 10)
    depth = st.slider('max_depth', 2, 30, 15)
    model = RandomForestRegressor(n_estimators=trees, max_depth=depth, random_state=42, n_jobs=-1)
else:
    trees = st.slider('n_estimators', 100, 800, 300, 100)
    depth = st.slider('max_depth', 2, 10, 6)
    rate = st.selectbox('learning_rate', [0.01, 0.05, 0.1, 0.3], index=1)
    model = XGBRegressor(n_estimators=trees, max_depth=depth, learning_rate=rate, random_state=42)

if st.button('Train model'):
    pipe = Pipeline([
        ('prep', preprocess),
        ('model', model)
    ])
    pipe.fit(X_train.head(n_rows), y_train.head(n_rows))

    actual = np.exp(y_val)
    pred = np.exp(pipe.predict(X_val))

    c1, c2, c3 = st.columns(3)
    c1.metric('R2', round(r2_score(actual, pred), 3))
    c2.metric('MAE', '£' + str(round(mean_absolute_error(actual, pred), 2)))
    c3.metric('RMSE', '£' + str(round(np.sqrt(mean_squared_error(actual, pred)), 2)))

    result = pd.DataFrame({'Actual': actual, 'Predicted': pred})
    fig = px.scatter(result.sample(2000, random_state=42), x='Actual', y='Predicted', opacity=0.4, title='Actual vs Predicted (validation set)')
    st.plotly_chart(fig)
