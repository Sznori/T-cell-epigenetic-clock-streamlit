import streamlit as st
import pandas as pd

# ennek kell az első lefutó parancsnak lennie
st.set_page_config(
page_title="T-cell specific epigenetic clock",
page_icon="🧬",
layout="wide",
initial_sidebar_state="expanded",
)

from common.common_functions import *


def explore_projects():

    # Add a title and intro text
    st.title('T-cell specific epigenetic clock')
    st.write('With the help of this application we can predict epigenetic age based on t-cell methylation values.')

    st.header('Explore the projects')

    gse_option = st.selectbox(
    'The machine learning model was trained with the data of 9 projects. You can explore these projects here.',
    ('GSE71955','GSE130029','GSE130030','GSE184500','GSE189148','GSE20242','GSE153459','GSE117050', 'GSE34639'))

    descriptions = pd.read_excel('descriptions.xlsx', index_col=0)

    description_opt, meta_opt, methyl_opt, predict_opt = st.tabs( ['Description', 'Metadata (information about the participants)', 'Methylation data','Prediction on this dataset when it was the testing in the cross-validation'])

    with description_opt:
        st.header('Description')
        st.dataframe(descriptions.loc[gse_option],use_container_width=True)
        st.write('Click on the cells twice to view the whole text!')

    with meta_opt:
        st.header('Metadata about the samples')
        st.write('F - Female, M - Male')
        project_meta = meta_table.loc[meta_table['Project']==gse_option]
        st.dataframe(project_meta)
        st.download_button('Download the metadata table for this project as a csv',project_meta.to_csv().encode('utf-8'),gse_option+'_metadata.csv')
        st.markdown('#') # seperator
        visualize_age(project_meta)
        
    with methyl_opt:
        st.header('Methylation data')
        st.write('First 100 cpg-sites of the first 20 people')
        gsm_list = (meta_table[(meta_table['Project']==gse_option)].index)
        methylations = metil_table.loc[gsm_list]
        st.dataframe(methylations.iloc[:100,:20])
        st.download_button('Download the methylation table for this project as a csv',methylations.to_csv().encode('utf-8'),gse_option+'_methylation.csv')
        st.markdown('#') # seperator
        visualize_metilation(gse_option,metil_table)

    with predict_opt: 
        st.header('Prediction on this dataset when it was the testing in the cross-validation')
        model_option = st.selectbox('Choose model type:',['Elasticnet','LightGBMRegressor','XGBRegressor','Support Vector Regression','Linear Regression','RandomForestRegressor'])
        use_model(gse_option,model_option)


@st.cache_resource(experimental_allow_widgets=True)
def use_model(gse_option,model_option):
    
    train_gsm_list = (control_meta_table[(control_meta_table['Project']!=gse_option)].index)
    y = (control_meta_table.loc[train_gsm_list]['Age'])
    x = metil_table.loc[train_gsm_list] 

    test_gsm_list = (control_meta_table[(control_meta_table['Project']==gse_option)].index)
    y_test = (control_meta_table.loc[test_gsm_list]['Age'])
    x_test = metil_table.loc[test_gsm_list]

    if len(x_test)!=0:
        y_pred = None
        if model_option=='Elasticnet':
            # slide az alpha állításához
            alpha = st.slider( 'Select the alpha value: ', 0.0, 1.0, value=0.5)
            y_pred, _ = elasticnet_model(x,y,x_test,alpha)
        elif model_option=='LightGBMRegressor':
            maxdepth = st.slider( 'Select the maximum depth of each tree : ', -1, 5, value=1)
            nestimators = st.slider( 'Select number of boosted trees to fit : ', 5, 200, value=50)
            y_pred, _ = lgbmRegressor_model(x,y,x_test,maxdepth,nestimators)
        elif model_option=='XGBRegressor':
            maxdepth = st.slider( 'Select the maximum depth of each tree : ', -1, 5, value=1)
            nestimators = st.slider( 'Select number of boosted trees to fit : ', 5, 200, value=50)
            y_pred, _ = xgbregressor_model(x,y,x_test,maxdepth,nestimators)
        elif model_option=='Support Vector Regression':
            y_pred, _ = svr_model(x,y,x_test)
        elif model_option=='Linear Regression':
            y_pred, _ = linear_regression_model(x,y,x_test)
        elif model_option=='RandomForestRegressor':
            nestimators = st.slider( 'Select number of boosted trees to fit : ', 5, 200, value=10)
            y_pred, _ = random_forest_model(x,y,x_test,nestimators)
        visualize_model(y_test,y_pred)
    else:
        st.write('We couldnt use this model as a test project, because it either doesnt have age attributumes or healthy,control patients.')



if __name__ == '__main__':
    explore_projects()