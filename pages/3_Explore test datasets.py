import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import numpy as np

# ennek kell az első lefutó parancsnak lennie
st.set_page_config(
page_title="T-cell specific epigenetic clock",
page_icon="🧬",
layout="wide",
initial_sidebar_state="expanded",
)

from common.common_functions import *


def try_out():
     # Add a title and intro text
    st.title('T-cell specific epigenetic clock')
    st.write('With the help of this application we can predict epigenetic age based on t-cell metilation values.')

    if 'full' not in chosen_models:
        st.error('You have to choose your model first.')
        switch = st.button("Go to 'Choose a model' here!")
        if switch:
            switch_page("Choose a model")
        return

    with st.expander('Your selected model to test on is: '):
        st.write(str(chosen_models['full']))

    st.header('Explore test datasets')
    disease_options = ('Graves disease','Multiple sclerosis','Psoriasis disease','Pregnancy', 'Post-transplantation cSCC (cancer)')
    selected_disease = st.selectbox(
    'Select a category and predict age. These are all not control patients so you can see how the different conditions effect the methylation values, therefore the epigenetic age. ',
    disease_options)
    if selected_disease == 'Graves disease' : 
        gse_option='GSE71955'
    elif selected_disease == 'Multiple sclerosis' : 
        gse_option='GSE130029'
    elif selected_disease == 'Psoriasis disease' : 
        gse_option='GSE184500'
    elif selected_disease == 'Pregnancy' : 
        gse_option='GSE153459'
    elif selected_disease == 'Post-transplantation cSCC (cancer)' : 
        gse_option='GSE117050'

    use_model_not_control(gse_option,selected_disease)


# Train model on the other projects controls
def use_model_not_control(gse_option, selected_disease):

    not_control_meta_table = meta_table[(meta_table['Diagnosis']!='Control') & (meta_table['Age'])]

    # not control test
    test_gsm_list_not_c = (not_control_meta_table[not_control_meta_table['Project']==gse_option].index)
    y_test_not_c = (not_control_meta_table.loc[test_gsm_list_not_c]['Age']) 
    y_test_not_c_category = (not_control_meta_table.loc[test_gsm_list_not_c]['Diagnosis']) 
    x_test_not_c = metil_table.loc[test_gsm_list_not_c]

    # control test
    # összehasonlításnak
    test_gsm_list_c = (control_meta_table[control_meta_table['Project']==gse_option].index)
    y_test_c = (control_meta_table.loc[test_gsm_list_c]['Age']) 
    x_test_c = metil_table.loc[test_gsm_list_c]

    # Make age predictions on test datas
    if len(y_test_c)==0: # ha nincs control (mert van egy ilyen projekt) akkor nincs chosen_models["gse"] sem
        y_pred_c = []
        y_pred_not_c = chosen_models['full'].predict(x_test_not_c)
    else:
        y_pred_c = chosen_models[gse_option].predict(x_test_c)
        y_pred_not_c = chosen_models[gse_option].predict(x_test_not_c)

    visualize_test_model(y_test_c, y_test_not_c ,y_test_not_c_category, y_pred_c, y_pred_not_c, selected_disease)



def visualize_test_model(y_test_c, y_test_not_c ,y_test_not_c_category, y_pred_c, y_pred_not_c, selected_disease):

    # diagram nem controlokon belül is kategóriánként szétszedni
    # Create a variable to indicate whether the data is control or not control
    control_samples = np.where(y_test_c, "Control Samples", "")
    disease_samples = y_test_not_c_category # np.where(y_test_not_c, selected_disease , "")   
    labels = np.concatenate((control_samples, disease_samples))

    # Define the default color and allow the user to change it using a color picker
    col1, col2 = st.columns(2)
    selected_color1 = col1.color_picker("Select a color for control samples", '#0000FF')
    selected_color2 = col2.color_picker("Select a color for " + selected_disease + " samples", '#FF00FF')

    # Plotly visualization
    y_pred= np.concatenate((y_pred_not_c,y_pred_c))
    y_test = np.concatenate((y_test_not_c,y_test_c))
    fig = px.scatter(x=y_test, y=y_pred , color=labels, \
                    color_discrete_map={"Control Samples": selected_color1, selected_disease: selected_color2}, \
                    trendline="ols" ) 
    n=len(y_test)
    r,p = pearsonr(y_test,y_pred)
    MedAE = median_absolute_error(y_test,y_pred) 
    MAE = mean_absolute_error(y_test,y_pred)
    fig.update_layout(title={'text': "Predicted vs Real Ages<br><span style='font-size: 14px; color: gray;'>r="+str(round(r,2))+', MedAE='+str(round(MedAE,1))+', MAE='+str(round(MAE,1))+"</span>"}, \
                        xaxis_title="Real Age (years)", \
                        yaxis_title="Predicted Age  (years)")
    # add the x=y line
    fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color='grey', width=2, dash='dash'))
    st.plotly_chart(fig)
    st.write('OLS trendline stands for "ordinary least squares", which is a method for estimating the parameters of a linear regression model. When trendline="ols" is specified, it adds a trendline to the scatter plot that represents the best fit line using the ordinary least squares method.')

    st.markdown('##')

    # Age Accelaration
    if len(y_test_c)!=0:
        age_acc_c = AgeAcceleration(y_test_c,y_pred_c)
        age_acc_not_c = AgeAcceleration(y_test_not_c,y_pred_not_c)
        visualize_age_acc(age_acc_c,age_acc_not_c)
    else:
        age_acc_not_c = AgeAcceleration(y_test_not_c,y_pred_not_c)
        visualize_age_acc(None,age_acc_not_c)
    

if __name__ == '__main__':
    try_out()

