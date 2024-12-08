import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
from streamlit_option_menu import option_menu
import plotly.express as px
from sklearn.ensemble import ExtraTreesRegressor

df = pd.read_csv(r'Real_Combine.csv')
df = df.dropna()
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

st.title('Air Quality Prediction')
st.write('This is a simple web application that predicts the air quality based on the given parameters')

page = option_menu(
    menu_title= None,
    options =  ['Prediction','Features Overview'],
    orientation='horizontal'
)

st.sidebar.subheader('Select the model for prediction')
mod = st.sidebar.selectbox('Model', ['Random Forest', 'Linear Regression', 'Ridge Regression', 'Lasso Regression'])

model_performance = {}
best_model_name = None
best_model_score = -np.inf

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


with st.spinner('Training the models'):
    # Random forest regression
    rf_model_name = 'rfc_model.pkl'
    if os.path.exists(rf_model_name):
        rf_rfc = joblib.load(rf_model_name)
    else:
        rf = RandomForestRegressor(random_state=42)
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
        min_samples_split = [2, 5, 10, 15, 100]
        min_samples_leaf = [1, 2, 5, 10]

        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}

        rf_rfc = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)
        rf_rfc.fit(X_train, y_train)
        joblib.dump(rf_rfc, rf_model_name)

    rf_pred = rf_rfc.predict(X_test)
    rf_score = r2_score(y_test, rf_pred)
    model_performance['Random Forest'] = rf_score

    # Linear Regression
    lr_model_name = 'lr_model.pkl'
    if os.path.exists(lr_model_name):
        lr = joblib.load(lr_model_name)
    else:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        joblib.dump(lr, lr_model_name)

    lr_pred = lr.predict(X_test)
    lr_score = r2_score(y_test, lr_pred)
    model_performance['Linear Regression'] = lr_score

    # Ridge Regression
    ridge_model_name = 'ridge_model.pkl'
    if os.path.exists(ridge_model_name):
        ridge = joblib.load(ridge_model_name)
    else:
        ridge=Ridge()
        parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
        ridge=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
        ridge.fit(X,y)
        joblib.dump(ridge,ridge_model_name)

    ridge_pred = ridge.predict(X_test)
    ridge_score = r2_score(y_test, ridge_pred)
    model_performance['Ridge Regression'] = ridge_score

    # Lasso Regression
    lasso_model_name = 'lasso_model.pkl'
    if os.path.exists(lasso_model_name):
        lasso = joblib.load(lasso_model_name)
    else:
        lasso = Lasso()
        parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
        lasso=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
        lasso.fit(X,y)
        joblib.dump(lasso,lasso_model_name)

    lasso_pred = lasso.predict(X_test)
    lasso_score = r2_score(y_test, lasso_pred)
    model_performance['Lasso Regression'] = lasso_score

best_model_name, best_model_score = max(model_performance.items(), key=lambda x: x[1])
st.sidebar.write(f"Best Model: {best_model_name} with R² Score: {best_model_score:.2f}")

if page == 'Prediction':
    left_col, right_col = st.columns([2, 2])

    with left_col:
        st.subheader('Enter the parameters')
        t = st.number_input('Temperature (C)', value=float(df['T'].mean()), step=0.1)
        tm = st.number_input('Maximum Temperature (C)', value=float(df['TM'].mean()), step=0.1)
        tmm = st.number_input('Minimum Temperature (C)', value=float(df['Tm'].mean()), step=0.1)
        slp = st.number_input('Sea-level Pressure (hPa)', value=float(df['SLP'].mean()), step=0.1)
        h = st.number_input('Humidity (%)', value=float(df['H'].mean()), step=0.1)
        vv = st.number_input('Visibility (km)', value=float(df['VV'].mean()), step=0.1)
        v = st.number_input('Wind Speed (km/h)', value=float(df['V'].mean()), step=0.1)
        vm = st.number_input('Maximum Wind Speed (km/h)', value=float(df['VM'].mean()), step=0.1)

    with right_col:
        st.header('Prediction Part')
        if st.button('Predict the PM 2.5 value'):
            inputs = np.array([[t, tm, tmm, slp, h, vv, v, vm]])
            if mod == 'Random Forest':
                st.subheader('Random Forest Regressor')
                st.info(f'R² Score: {model_performance['Random Forest']:.2f}')
                prediction = rf_rfc.predict(inputs)[0]
            elif mod == 'Linear Regression':
                st.subheader('Linear Regression')
                st.info(f'R² Score: {model_performance['Linear Regression']:.2f}')
                prediction = lr.predict(inputs)[0]
            elif mod == 'Ridge Regression':
                st.subheader('Ridge Regression')
                st.info(f'R² Score: {model_performance['Ridge Regression']:.2f}')
                prediction = ridge.predict(inputs)[0]
            elif mod == 'Lasso Regression':
                st.subheader('Lasso Regression')
                st.info(f'R² Score: {model_performance['Lasso Regression']:.2f}')
                prediction = lasso.predict(inputs)[0]

            st.success(f"Predicted PM 2.5 value: {prediction:.2f}")
            st.write('The PM 2.5 value is the Particulate matter concentration (µg/m³) which Represents the average concentration of fine particulate matter with a diameter of 2.5 microns or smaller, a key measure of air pollution.')


if page == 'Features Overview':
    st.header('Features Overview')
    st.subheader('The Correlation Matrix')
    corrmat = df.corr().reset_index().melt(id_vars='index', var_name='Variable', value_name='Correlation')
    fig_heatmap = px.imshow(
        df.corr(),
        labels=dict(color="Correlation"),
        x=df.columns,
        y=df.columns,
        color_continuous_scale="RdYlGn",
        title="Feature Correlation Heatmap"
    )

    st.plotly_chart(fig_heatmap)

    st.subheader('Feature Importance using Extra Trees Regression')
    model = ExtraTreesRegressor()
    model.fit(X, y)

    feat_importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(5)

    fig_feat_importance = px.bar(
        feat_importances,
        x=feat_importances.values,
        y=feat_importances.index,
        orientation='h',
        labels={'x': 'Feature Importance', 'index': 'Features'},
        title="Top 5 Feature Importances",
        color=feat_importances.values,
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig_feat_importance)

    st.subheader('The Model Evaluation Histogram for each model')
    dd = st.selectbox(label='Select Model',options=['Random Forest', 'Linear Regression', 'Ridge Regression', 'Lasso Regression'])

    if dd == 'Random Forest':
        residuals = y_test - rf_pred
    if dd == 'Linear Regression':
        residuals = y_test - lr_pred
    if dd == 'Ridge Regression':
        residuals = y_test - ridge_pred 
    if dd == 'Lasso Regression':
        residuals = y_test - lasso_pred

    if dd:
        residuals_df = pd.DataFrame({'Residuals': residuals})
        fig = px.histogram(residuals_df, x="Residuals", marginal="box", nbins=50, title="Residuals Distribution")
        st.plotly_chart(fig)
