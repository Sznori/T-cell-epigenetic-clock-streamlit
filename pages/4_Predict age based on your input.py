import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import pandas as pd
from typing import Optional

# ennek kell az első lefutó parancsnak lennie
st.set_page_config(
page_title="T-cell specific epigenetic clock",
page_icon="🧬",
layout="wide",
initial_sidebar_state="expanded",
)

from common.common_functions import *


def predict_age():
    # Add a title and intro text
    st.title('T-cell specific epigenetic clock')
    st.write('With the help of this application we can predict epigenetic age based on t-cell metilation values.')

    st.header("To find out your or a group of people's epigenetic age please upload your file here")

    if 'full' not in chosen_models:
        st.error('You have to choose your model first.')
        switch = st.button("Go to 'Choose a model' here!")
        if switch:
            switch_page("Choose a model")
        return

    with st.expander('Your selected model to test on is: '):
        st.write(str(chosen_models['full']))
    
    left, right = st.columns(2)
    with left:
        upload_metil_file = st.file_uploader('Upload your methylation csv or pkl file, where each row is a cpg-site and each column is a sample/person/gsm.')
    with right:
        upload_age_file = st.file_uploader('Upload a csv or pkl file with the ages of the corresponding gsm-s (people). Each row should be a sample/person/gsm and there is only one column, named "Age".')

    metil_df,age = file_check(upload_metil_file,upload_age_file)
    if metil_df is not None :
        y_pred = use_model_full(metil_df)

        # Common cpg-sites:
        common_cpg_count = len(metil_df.index.intersection(metil_table.index))
        st.write("There are "+str(common_cpg_count)+" common cpg-sites between the training and testing datasets.")
        st.write("If this is too low the prediction can be worse. ")
        if common_cpg_count<30 :
            st.error("The common cpg-site count is too low for testing the dataset.")
            return

        if age is not None :
            if len(metil_df.columns)==1 : # egy emberes
                st.subheader('The predicted age is: '+ str(y_pred[0]))
                st.subheader('The error in the prediction is: '+ str(y_pred[0] - age.values[0, 0]))
                st.write('Your files:')
                st.write(metil_df)
                st.write(age)
            else:
                # create dataframe to show also which sample has which age, not just the ages
                y_pred_df = pd.DataFrame(data = y_pred, index = metil_df.columns, columns=['Age'])
                y_pred_vs_real = pd.DataFrame(data = pd.concat([y_pred_df,age['Age']],axis=1))
                y_pred_vs_real.columns=['Predicted age','Real age']
                st.subheader('The predicted ages vs real ages are: ')
                st.dataframe(y_pred_vs_real.T)
                # Download results as csv
                filename = upload_metil_file.name.split(".")[0]
                st.download_button('Download your predicted age results here as csv',y_pred_vs_real.to_csv().encode('utf-8'),filename+'_predicted_age.csv')
                # Ha van condition akkor aszerint külön visualize
                if 'Condition' in age.columns:
                    visualize_model_with_categories(age['Age'],y_pred,age['Condition'])
                else:
                    visualize_model(age['Age'],y_pred)
                st.write('Your files:')
                st.write(metil_df)
                st.write(age.T)
        else:
            if len(metil_df.columns)==1 : # egy emberes
                st.subheader('The predicted age is: '+ str(y_pred[0]))
            else :
                # create dataframe to show also which sample has which age, not just the ages
                y_pred_df = pd.DataFrame(data = y_pred, index = metil_df.columns, columns=['Age'])
                st.subheader('The predicted ages are: ')
                st.dataframe(y_pred_df.T)
                # Download results as csv
                filename = upload_metil_file.name.split(".")[0]
                st.download_button('Download your predicted age results here as csv',y_pred_df.to_csv().encode('utf-8'),filename+'_predicted_age.csv')
                # Scatterplot
                fig = px.scatter(y_pred_df)
                fig.update_layout(xaxis_title='Samples',yaxis_title='Predicted ages (years)')
                st.plotly_chart(fig)
            st.write('Your file:')
            st.write(metil_df)


def check_dataframe_columns_and_indexes(df):
    all_columns_are_strings = all(isinstance(col, str) for col in df.columns)
    all_indexes_are_strings = all(isinstance(idx, str) for idx in df.index)
    return all_columns_are_strings and all_indexes_are_strings    

def file_check(metil_file : Optional[str] = None, age_file : Optional[str] = None ):
    if metil_file is None:
        st.error("No methylation file yet.")
        return None,None

    metil_filename = metil_file.name
    file_extension = metil_filename.split(".")[-1]
    if file_extension!='pkl' and file_extension!='csv':
        st.warning("You can only upload csv or pkl files!")
        return None,None
    metil_df = read_data(metil_file,file_extension)

    # ELLENŐRZÉSEK:
    if metil_df is None:
        st.warning("Dataframe couldnt be loaded.")
        return None,None
    # Egy emberes pkl esetén series-nek olvassa be amit át kell alakítani DataFrame-é (a column name = series name)
    if isinstance(metil_df,pd.Series):
        metil_df = pd.DataFrame(metil_df, columns=[metil_df.name])

    if not metil_df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()): 
        st.warning('You should upload only numeric values.')
        return None,None
    if not check_dataframe_columns_and_indexes(metil_df): 
        st.warning('The column names and indexes should be string values.')
        return None,None
    if (metil_df < 0).any().any() or (metil_df > 1).any().any():
        st.warning('You should upload beta values.')
        return None,None
    st.success('Methylation file successfully read.')

    if age_file is not None:
        age_filename = age_file.name
        file_extension = age_filename.split(".")[-1]
        if file_extension!='pkl' and file_extension!='csv':
            return st.warning("You can only upload csv or pkl files!")
        age = read_data(age_file,file_extension)

        # ELLENŐRZÉSEK:
        # ez tényleg age file e
        if age is None:
            st.warning("Dataframe couldnt be loaded.")
            return None,None
        # Egy emberes pkl esetén series-nek olvassa be amit át kell alakítani DataFrame-é (a column name = series name)
        if isinstance(age,pd.Series):
            age = pd.DataFrame(age, columns=[age.name])

        if not age.to_numeric(age.iloc[:, 0], errors='coerce').notna().all():
            # errors='coerce' parameter is used to convert non-numeric values to NaN
            st.warning('You should upload only numeric values in the first column.')
            return None,None
        if not check_dataframe_columns_and_indexes(metil_df):
            st.warning('The index values should be strings.')
            return None,None
        if age.columns[0] != "Age" if len(age.columns) > 0 else True: 
            st.warning('There should be at least 1 column, and the first column name should be "Age".')
            return None,None
        if age.columns[1] != "Condition" if len(age.columns) > 1 else False: 
            st.warning('The second column name should be "Condition".')
            return None,None
        # ugyanahhoz az adathalmazhhoz tartozik az methyil és age file
        if len(age.index)!=len(metil_df.columns) :
            st.warning('The age file should contain the same amount of ages and in the same order (gsm order) as for in the uploaded mehtyhaltion file.')
            return metil_df,None 
        elif (age.index!=metil_df.columns).any() :
            st.warning('The age file should contain the ages for the same people (gsm) and same order as for in the uploaded mehtyhaltion file.')
            return metil_df,None 
        return metil_df,age

    return metil_df,None    

 

def use_model_full(test_df):

    train_gsm_list = (control_meta_table.index)
    y = control_meta_table.loc[train_gsm_list]['Age'] # azért hogy sorrend biztosan megmaradjon (loc[train_gsm_list])
    x = metil_table.loc[train_gsm_list]
    y_pred = chosen_models['full'].predict(test_df.T)

    return y_pred


if __name__ == '__main__':
    predict_age()



# TODO
# condition column 
# elég egyező site van??
# ellenőrzésekre ellenpéldát?