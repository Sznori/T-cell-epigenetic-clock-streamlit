import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from scipy.stats import pearsonr
import plotly.express as px # to make it interactive
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from glmnet import ElasticNet
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle
import tempfile


# Elastic Net model
# Azért hogy ugyanazokkal a paraméterekkel ne traineljen többször modellt
def elasticnet_model(x,y,x_test,alpha):
    enet = ElasticNet(alpha=alpha)
    # Train an Elastic Net model
    model = enet.fit(x, y)
    # Make age predictions on test data
    y_pred = model.predict(x_test)
    return y_pred, model


# LightGBMRegressor (with default hyperparameters)
@st.cache_resource
def lgbmRegressor_model(x,y,x_test,maxdepth,nestimators):
    lgbm_regressor = lgb.LGBMRegressor(max_depth=maxdepth, n_estimators=nestimators)
    lgbm_regressor.fit(x,y)
    y_pred = lgbm_regressor.predict(x_test)
    return y_pred, lgbm_regressor


# XGBRegressor model
@st.cache_resource 
def xgbregressor_model(x,y,x_test,maxdepth,nestimators):
    xgboost_reg = xgb.XGBRegressor(max_depth=maxdepth,n_estimators=nestimators) 
    xgboost_reg.fit(x, y)
    y_pred = xgboost_reg.predict(x_test)
    return y_pred, xgboost_reg


# Support Vector Regression model
@st.cache_resource 
def svr_model(x,y,x_test):
    regressor = SVR(kernel = 'linear')
    regressor.fit(x, y)
    y_pred = regressor.predict(x_test)
    return y_pred, regressor


# Linear Regression model
@st.cache_resource 
def linear_regression_model(x,y,x_test):
    # Create linear regression object
    regr = LinearRegression() 
    regr.fit(x, y)
    y_pred = regr.predict(x_test)
    return y_pred, regr


# RandomForestRegressor model
@st.cache_resource 
def random_forest_model(x,y,x_test,nestimators):
    # Create linear regression object
    rf = RandomForestRegressor(n_estimators = nestimators, random_state = 42)
    rf.fit(x, y)
    y_pred = rf.predict(x_test)
    return y_pred, rf


def save_model(type, model):
    if type=='Elasticnet':
        model_bytes = pickle.dumps(model)
        st.download_button('Download the chosen ElasticNet model', data=model_bytes, file_name='elasticnet_model.pkl')
    elif type=='LightGBMRegressor': 
        if st.button('Download the chosen LightGBMRegressor model'):
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=True) as temp_file:
                # Save the model to the temporary file
                model.booster_.save_model(temp_file.name)

                # Read the contents of the temporary file
                with open(temp_file.name, 'r') as file:
                    model = file.read()

                # Serve the model as a download to the user
                st.download_button(
                    label='Download',
                    data=model,
                    file_name='lightgbm_model.txt',
                    mime='text/plain'
                )
    elif type=='XGBRegressor':
        if st.button('Download the chosen XGBRegressor model'):
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=True) as temp_file:
                # Save the model to the temporary file
                model.save_model(temp_file.name)

                # Read the contents of the temporary file
                with open(temp_file.name, 'r') as file:
                    model = file.read()

                # Serve the model as a download to the user
                st.download_button(
                    label='Download',
                    data=model,
                    file_name='xgbregressor_model.txt',
                    mime='text/plain'
                )
    elif type=='Support Vector Regression':
        model_bytes = pickle.dumps(model)
        st.download_button('Download the chosen Support Vector Regression model', data=model_bytes, file_name='svr_model.pkl')
    elif type=='Linear Regression':
        model_bytes = pickle.dumps(model)
        st.download_button('Download the chosen Linear Regression model', data=model_bytes, file_name='linear_regression_model.pkl')
    elif type=='RandomForestRegressor':
        model_bytes = pickle.dumps(model)
        st.download_button('Download the chosen RandomForestRegressor model', data=model_bytes, file_name='random_forest_model.pkl')
    else:
        st.error('Wrong model type')
        return
    return


# -------------------------------------------------------------------------


@st.cache_data
def read_data(path,format):
    if format=='csv':
        return pd.read_csv(path,index_col=0)
    elif format=="pkl":
        return pd.read_pickle(path)
    
# global variables (sokat használt)
meta_table = read_data("./merged_metadata.pkl",'pkl')
metil_table = read_data("./filtered_merged_metilations.pkl",'pkl')
control_meta_table = meta_table[(meta_table['Diagnosis']=='Control') & (meta_table['Age'])]
chosen_models = {}

# -------------------------------------------------------------------------


def visualize_age(project_meta):
    st.subheader('Statistics of the age')
    st.dataframe(project_meta['Age'].describe().to_frame().T)
    # For visualization create boxplot
    fig = px.box(project_meta, y='Age')
    fig.update_layout(yaxis_title=" Ages  (years)")
    st.plotly_chart(fig)


def visualize_model(y_test,y_pred):
    # Plotly visualization
    fig = px.scatter(x=y_test, y=y_pred ,trendline="ols" ) 
    n=len(y_test)
    r,p = pearsonr(y_test,y_pred)
    MedAE = median_absolute_error(y_test,y_pred) 
    MAE = mean_absolute_error(y_test,y_pred)
    fig.update_layout(title={'text': "Predicted vs Real Ages<br><span style='font-size: 14px; color: gray;'>r="+str(round(r,2))+', MedAE='+str(round(MedAE,1))+', MAE='+str(round(MAE,1))+"</span>"}, xaxis_title="Real Age (years)", yaxis_title="Predicted Age  (years)")
    # add the x=y line
    fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color='gray', width=2, dash='dash'))
    st.plotly_chart(fig)
    st.caption('OLS trendline stands for "ordinary least squares", which is a method for estimating the parameters of a linear regression model. When trendline="ols" is specified, it adds a trendline to the scatter plot that represents the best fit line using the ordinary least squares method.')

def visualize_model_with_categories(y_test,y_pred,labels):
    # Plotly visualization
    fig = px.scatter(x=y_test, y=y_pred , color=labels , trendline="ols", trendline_scope="overall") 
    n=len(y_test)
    r,p = pearsonr(y_test,y_pred)
    MedAE = median_absolute_error(y_test,y_pred) 
    MAE = mean_absolute_error(y_test,y_pred)
    fig.update_layout(title={'text': "Predicted vs Real Ages<br><span style='font-size: 14px; color: gray;'>r="+str(round(r,2))+', MedAE='+str(round(MedAE,1))+', MAE='+str(round(MAE,1))+"</span>"}, \
                        xaxis_title="Real Age (years)", \
                        yaxis_title="Predicted Age  (years)")
    fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color='grey', width=2, dash='dash'))
    st.plotly_chart(fig)
    st.caption('OLS trendline stands for "ordinary least squares", which is a method for estimating the parameters of a linear regression model. When trendline="ols" is specified, it adds a trendline to the scatter plot that represents the best fit line using the ordinary least squares method.')


def visualize_metilation(gse_name, table):
    if gse_name not in ['GSE20242', 'GSE117050', 'GSE34639']: # nincs unnamed : 0
        data = table.iloc[: , 1:].mean() # átalgoljuk cgsiteonként
    else:
        data = table.mean() # átalgoljuk cgsiteonként
    bin_num = st.slider("Set the number of bins: ",10,100,50)
    fig = px.histogram(data,nbins=bin_num,histnorm="probability density")
    fig.update_layout( xaxis_title="Methylation value", yaxis_title="Density")
    st.plotly_chart(fig)


def visualize_cpg_methylation_change(cpg1, cpg2):

    fig = make_subplots(rows=1, cols=2, subplot_titles=(cpg1, cpg2), horizontal_spacing=0.1)

    # Add traces to the first plot with trendline
    fig.add_trace(px.scatter(x=meta_table['Age'], y=metil_table[cpg1]).data[0],
                row=1, col=1)

    # Add traces to the second plot with trendline
    fig.add_trace(px.scatter(x=meta_table['Age'], y=metil_table[cpg2]).data[0],
                row=1, col=2)
    
    fig.update_layout( xaxis_title="Age (years)", yaxis_title="Methylation levels")
    fig.update_layout(title='Observe the changes in the methylation level of a relevant cpg site depending on age: ')
    st.plotly_chart(fig)


def visualize_age_acc(age_acc_c,age_acc_not_c):
    # boxplot
    if type(age_acc_c) == type(None):
        data = age_acc_not_c
        labels = np.repeat('Not control', len(age_acc_not_c))
    else:
        data = np.concatenate((age_acc_c, age_acc_not_c))
        labels = np.concatenate((np.repeat('Control', len(age_acc_c)), np.repeat('Not control', len(age_acc_not_c))))
    df = pd.DataFrame({'AgeAcc': data, 'Category': labels})
    fig = px.box(df, x='Category', y='AgeAcc')
    fig.update_layout(title='Age accelaration',yaxis_title=" Age accelaration value")
    st.plotly_chart(fig)


# Az input paraméterek: az életkorok (age) és a prediktált életkorok (pred1) numpy tömbje (természetesen matchelve a sorrend).
def AgeAcceleration(age,pred_age):
    x = age.to_numpy().reshape(len(age), 1)
    y = pred_age
    reg = LinearRegression().fit(x, y)
    linregpred = age * reg.coef_[0] + reg.intercept_
    age_acc = pred_age - linregpred
    return age_acc