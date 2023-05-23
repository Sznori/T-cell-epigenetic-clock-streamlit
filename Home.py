import streamlit as st

def main():
    
    # Add a title and intro text
    st.title('T-cell specific epigenetic clock')
    st.write('With the help of this application we can predict epigenetic age based on t-cell methylation values.')

    st.header('Home')
    st.write('This webpage was created to present an epigenetic clock and calculate DNA methylation (DNAm) age based on T-cell methylation data measured using the Illumina Infinium platform (450K, 27K, EPIC).')
    st.write('The model was trained using 9 projects downloaded from the public GEO database : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi selecting the common cpg-sites')
    
    st.markdown('##')

    st.markdown("In the ***:red[Explore the projects]*** section we can examine these projects. From a list we have to select a project and what we want to see.")
    st.image('images/ketto.png', caption='Explore the projects example screenshot') # TODO images nem jelenik meg néha
    st.markdown('The ***:orange[Description]*** section provides general information such as when, where, and by whom the research was conducted, as well as what it was specifically about.')
    st.markdown('We can see information about the subjects involved in the research in the ***:orange[Metadata]*** section. Each subject is assigned an unique GSM identifiers. We can download this dataset and see statistics and diagrams about the age distribution below the table, as this is one of the most essential data in the epigenetic clock.')
    st.markdown('Clicking on ***:orange[Methylation data]***, we can examine the distribution of methylation levels and download the full methylation table for the project.')
    st.markdown('Under ***:orange[Prediction]***, it is possible to make predictions about the control samples of the selected project. In this case the model will be trained using the other 8 projects. First we have to select then set up the the model. We can choose between ***ElasticNet*** and ***LightGBM***. In case of ElasticNet it is possible to set the value of the *:blue[alpha hyperparameter]*. The application then makes predictions on the data using the selected model, then visualizes the results with a scatter plot that shows the actual and predicted age values, as well as the ordinary least squares (OLS) trendline.',
    help="ElasticNet uses both lasso and ridge penalty. With setting the alpha parameter we can adjust the ratio. If alpha=0 its equal to Lasso regularization and if alpha=1 it is Ridge regularization. Intermediate values of alpha result in a combination of Lasso and Ridge regularization. Most common value used is alpha=0.5.")
    
    st.markdown('##')

    st.markdown('Navigating to the ***:red[Explore the model]*** tab, we see the results of the model using cross-fold validation. It is possible to select model type, alpha value (in case of ElasticNet) and to download the selected model. Under the scatter plot we see the distribution of the methylation data and age of every training project alltogether. We can also see the number of used cpg sites (based on correlation) and their weights.')

    st.markdown('##')

    st.markdown("In case of ***:red[Explore test dataset]***, we can choose between 5 cases ('Graves disease','Multiple sclerosis','Psoriasis disease','Pregnancy', 'Post-transplantation cSCC (cancer)') and observe what the model predicts on them compared to the control samples. Here the model uses alpha=0.5.") 

    st.markdown('##')

    st.markdown("We can upload our own methylation and age table in the ***:red[Predict age based on your input]*** section. An ElasticNet model with alpha=0.5 predicts the ages, and the error if an age file was also present. Both of the files should be in csv or pkl format. Regarding the methylations the values should be beta values, the name of a cpg-site should be in the rows and the different GSM-s (samples) labels in the columns. In the age file the GSM identifiers need to be in the rows and there should be only one column with the name of 'Age'."\
                +"The model checks if the files are in the right format, if the methylations are beta values and if the age files matches the samples in the methylation file")

    st.markdown('The diagrams can be downloaded by hovering over them and clicking on the camera icon in the upper-right corner.')
    # Methylation Data Examples
    

if __name__ == '__main__':

    st.set_page_config(
    page_title="T-cell specific epigenetic clock",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    )

    main()
    

# TODO befejezni 



