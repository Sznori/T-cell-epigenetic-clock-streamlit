import streamlit as st
import pandas as pd
import numpy as np

# ennek kell az első lefutó parancsnak lennie
st.set_page_config(
page_title="T-cell specific epigenetic clock",
page_icon="🧬",
layout="wide",
initial_sidebar_state="expanded",
)

from common.common_functions import *


def explore_model():

    # Add a title and intro text
    st.title('T-cell specific epigenetic clock')
    st.write('With the help of this application we can predict epigenetic age based on t-cell metilation values.')

    st.header('Cross fold validation diagram of the model trained with all of the control samples')
    st.warning('The model is trained in real-time so it might take a few minutes to get the result.')

    if st.session_state.get("predict_button", False):
        st.session_state.disabled = True

    model_option = st.selectbox('Choose model type:',['Elasticnet','LightGBMRegressor','XGBRegressor','Support Vector Regression','Linear Regression','RandomForestRegressor'],disabled=st.session_state.get("disabled", False))
    
    alpha = None
    maxdepth = None
    nestimators = None

    if model_option=='Elasticnet':
        alpha = st.slider( 'Select the alpha value: ', 0.0, 1.0, value=0.5, disabled=st.session_state.get("disabled", False))
    elif model_option=='LightGBMRegressor':
        maxdepth = st.slider( 'Select the maximum depth of each tree : ', -1, 5, value=1, disabled=st.session_state.get("disabled", False))
        nestimators = st.slider( 'Select number of boosted trees to fit : ', 5, 200, value=50, disabled=st.session_state.get("disabled", False))
    elif model_option=='XGBRegressor':
        maxdepth = st.slider( 'Select the maximum depth of each tree : ', -1, 5, value=1 , disabled=st.session_state.get("disabled", False))
        nestimators = st.slider( 'Select number of boosted trees to fit : ', 5, 200, value=50, disabled=st.session_state.get("disabled", False))
    elif model_option=='RandomForestRegressor':
        nestimators = st.slider( 'Select number of boosted trees to fit : ', 5, 200, value=10, disabled=st.session_state.get("disabled", False))

    if st.button('Predict',key='predict_button',disabled=st.session_state.get("disabled", False)):
        st.markdown('##')
        st.divider()
        # selector desibaled legyen hogy ne zavarjon be
        cv_model(model_option,alpha,maxdepth,nestimators)
        st.session_state.disabled = False 
        st.experimental_rerun()  # hogy újra enabled legyenek az elemek
    elif 'full' in chosen_models:
        st.markdown('##')
        st.divider()
        st.subheader('Chosen model is: ')
        code = str(chosen_models['full'])
        st.code(code, language='python')
        visualize_cv_model()



def cv_model(model_option,alpha,maxdepth,nestimators):

    chosen_models.clear()
    chosen_models['type']=model_option
    
    y_test_full = pd.Series(dtype='float64')
    y_pred_full = pd.Series(dtype='float64')
    labels = pd.Series(dtype='float64')

    for gse_name in ('GSE71955','GSE130029','GSE130030','GSE184500','GSE189148','GSE20242','GSE153459','GSE117050', 'GSE34639'):

        train_gsm_list = (control_meta_table[(control_meta_table['Project']!=gse_name)].index)
        y = (control_meta_table.loc[train_gsm_list]['Age'])
        x = metil_table.loc[train_gsm_list]

        test_gsm_list = (control_meta_table[(control_meta_table['Project']==gse_name)].index)
        y_test = (control_meta_table.loc[test_gsm_list]['Age'])
        x_test = metil_table.loc[test_gsm_list]
        
        if len(x_test)!=0:
            y_pred = None
            model = None
            if chosen_models['type']=='Elasticnet':
                y_pred, model = elasticnet_model(x,y,x_test,alpha)
            elif chosen_models['type']=='LightGBMRegressor':
                y_pred, model = lgbmRegressor_model(x,y,x_test,maxdepth,nestimators)
            elif chosen_models['type']=='XGBRegressor':
                y_pred, model = xgbregressor_model(x,y,x_test,maxdepth,nestimators)
            elif chosen_models['type']=='Support Vector Regression':
                y_pred, model = svr_model(x,y,x_test)
            elif chosen_models['type']=='Linear Regression':
                y_pred, model = linear_regression_model(x,y,x_test)
            elif chosen_models['type']=='RandomForestRegressor':
                y_pred, model = random_forest_model(x,y,x_test,nestimators)

            y_test_full = np.concatenate((y_test_full,y_test))
            y_pred_full = np.concatenate((y_pred_full,y_pred))
            labels = np.concatenate((labels, np.where(y_test,gse_name,"") ))
            chosen_models[gse_name]=model

    # hogy lap újratöltődése esetén ne kelljen újra kiszámolni ezeket az ábrázoláshoz
    chosen_models['y_test_full'] = y_test_full
    chosen_models['y_pred_full'] = y_pred_full
    chosen_models['labels'] = labels

    # Előbb le kell generálni az összes projektre betrainelt modelt
    train_gsm_list = (control_meta_table.index)
    y_full = control_meta_table.loc[train_gsm_list]['Age'] # azért hogy sorrend biztosan megmaradjon (loc[train_gsm_list])
    x_full = metil_table.loc[train_gsm_list]

    test_gsm_list = (control_meta_table[(control_meta_table['Project']=='GSE130029')].index)
    x_test_tmp = metil_table.loc[test_gsm_list]     # x_test mindegy micsoda, csak ne legyen üres

    if chosen_models['type']=='Elasticnet':
        _ , chosen_models['full'] =  elasticnet_model(x_full,y_full,x_test_tmp,alpha)
    elif chosen_models['type']=='LightGBMRegressor':
        _ , chosen_models['full'] =  lgbmRegressor_model(x_full,y_full,x_test_tmp,maxdepth,nestimators)
    elif chosen_models['type']=='XGBRegressor':
        _ , chosen_models['full'] =  xgbregressor_model(x_full,y_full,x_test_tmp,maxdepth,nestimators)
    elif chosen_models['type']=='Support Vector Regression':
        _ , chosen_models['full'] =  svr_model(x_full,y_full,x_test_tmp)
    elif chosen_models['type']=='Linear Regression':
        _ , chosen_models['full'] =  linear_regression_model(x_full,y_full,x_test_tmp)
    elif chosen_models['type']=='RandomForestRegressor':
        _ , chosen_models['full'] =  random_forest_model(x_full,y_full,x_test_tmp,nestimators)


def visualize_cv_model():

    y_test_full = chosen_models['y_test_full']
    y_pred_full = chosen_models['y_pred_full']
    labels = chosen_models['labels']
    model_type = chosen_models['type']

    visualize_model_with_categories(y_test_full,y_pred_full,labels);

    # Modell letöltése link
    save_model(model_type,chosen_models['full'])
    # TODO : kirpóbáli hogy visszaolvasva működik e

    # TODO feautre_importancesre rákérdezni csabánál ??? és kipróbálni a letöltéseket
    # fontos cg-siteok és párról diagram
    # Relevant cpg-sites with weights on the fully trained model letöltése
    # Extract and process the relevant coefficients
    if model_type=='Elasticnet'  or model_type=='Linear Regression' :
        coef = pd.DataFrame(chosen_models['full'].coef_,index=metil_table.columns,columns=['Weights'])
        relevant_coef = coef[chosen_models['full'].coef_!=0]
        st.write('There are '+str(relevant_coef.shape[0])+' relevant cpg-sites:')
        # Prepare the downloadable model weights
        donwloadable_model = relevant_coef.T
        donwloadable_model.insert(loc=0, column='Interception', value=chosen_models['full'].intercept_)
        st.write(donwloadable_model)
        st.download_button('Download the weights as csv',donwloadable_model.to_csv(),'model_weights.csv')

        max_cpg = relevant_coef['Weights'].idxmax()
        min_cpg = relevant_coef['Weights'].idxmin()
        visualize_cpg_methylation_change(min_cpg,max_cpg)
    elif model_type=='LightGBMRegressor'  or model_type=='RandomForestRegressor':
        coef = pd.DataFrame(chosen_models['full'].feature_importances_,index=metil_table.columns,columns=['Weights'])
        relevant_coef = coef[chosen_models['full'].feature_importances_!=0]
        st.write('There are '+str(relevant_coef.shape[0])+' relevant cpg-sites:')
        # Prepare the downloadable model weights
        donwloadable_model = relevant_coef.T
        st.write(donwloadable_model)
        st.download_button('Download the weights as csv',donwloadable_model.to_csv(),'model_weights.csv')

        max_cpg = relevant_coef['Weights'].idxmax()
        min_cpg = relevant_coef['Weights'].idxmin()
        visualize_cpg_methylation_change(min_cpg,max_cpg)

        # or model_type=='XGBRegressor'
        # or model_type=='Support Vector Regression'
        #  LightGBMRegressor???? TODO


    st.markdown('##')

    visualize_metilation("",metil_table.loc[control_meta_table.index])
    visualize_age(control_meta_table)


if __name__ == '__main__':
    explore_model()