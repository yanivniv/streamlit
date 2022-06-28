import streamlit as st
st.set_page_config(layout="wide")
from streamlit_shap import st_shap
import shap
from scipy import stats
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import requests

#from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
path_var = "https://github.com/yanivniv/streamlit/blob/main/"

@st.experimental_memo
def load_data():
    #df = pd.read_csv('c:\castpone\df_final.csv')
    df = pd.read_csv('https://raw.githubusercontent.com/yanivniv/streamlit/main/df_final.csv')
    X = pd.read_csv('https://raw.githubusercontent.com/yanivniv/streamlit/main/X_new.csv')                
    picklefile = open("c:/castpone/top_10_model.pkl", "rb")
    model = pickle.load(picklefile)       
    return df,model,X

@st.experimental_memo
def load_model():
    #df = pd.read_csv('c:\castpone\df_final.csv')
    df = pd.read_csv('https://raw.githubusercontent.com/yanivniv/streamlit/main/df_final.csv')
    #X = pd.read_csv('c:\castpone\X_new.csv')                
    mLink = 'https://github.com/yanivniv/streamlit/raw/main/top_10_model.pkl'
    mfile = BytesIO(requests.get(mLink).content)        
    model = pickle.load(mfile)       
    return df,model

app_mode = st.sidebar.selectbox('Select Page',['Resume','Model Feature Importance','Dynamic Prediction'])

if app_mode=='Model Feature Importance': 

    st.title('Airbnb Prediction Feature Importance') 
    df,model,X = load_data()

    # compute SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    st.write("### Combined SHAP values")
    st_shap(shap.plots.beeswarm(shap_values), height=500, width=800)

    st.write("### Snapshot of processed data, Prediction and Error")
    df_sample = df.sample(10, random_state=42)
    st.write(df_sample)

    option = st.selectbox('Which record would you like to explain?',tuple(df_sample.index))

    st.write('Please wait while trying to explain record:', option)     

    st.write("Random Record Explainer")
    st_shap(shap.plots.waterfall(shap_values[option]), height=500, width=800)


if app_mode=='Dynamic Prediction':
    df,model = load_model()

    st.write("## Property Questions")
    room_type = st.selectbox('What is the room type',("Entire Home","Private Room","Other"))
    loction = st.selectbox('Is it in Manhattan',("Yes","No")) 
    bathroom = st.selectbox('What Is the bathroom type',("Private, With Bath","Shared", "Other")) 
    accomodates = st.number_input("How many people it can accomodate?",min_value=1)
    bedrooms = st.number_input("How many Bedrooms in the House?",min_value=1)
    location_score = st.number_input("What is the predicted location score by users (1-5)",min_value=1, max_value=5)
    gym_var = st.selectbox('Does the property have Gym',("Yes","No")) 
    


    if room_type == "Private Room":
        room_type_Entire = 0
        room_type_Private = 1
    elif room_type == "Entire Home":
        room_type_Entire = 1
        room_type_Private = 0
    else:
        room_type_Entire = 0
        room_type_Private = 0

    if loction == "Yes":
        neighbourhood_group_cleansed_Manhattan = 1
    else:
        neighbourhood_group_cleansed_Manhattan = 0

    if bathroom == "Private, With Bath":
        bathroom_type_baths = 1
        bathroom_type_shared = 0
    elif bathroom == "Shared":
        bathroom_type_baths = 0
        bathroom_type_shared = 1
    else:
        bathroom_type_baths = 0
        bathroom_type_shared = 0

    bedrooms_per_accommodates = bedrooms/accomodates

    if gym_var == "Yes":
        gym = 1
    else:
        gym = 0

    features = [room_type_Entire
                ,neighbourhood_group_cleansed_Manhattan
                ,accomodates
                ,bathroom_type_baths
                ,bedrooms
                ,bathroom_type_shared
                ,bedrooms_per_accommodates
                ,location_score
                ,gym
                ,room_type_Private
                ]
    
    feature_to_model = np.array(features).reshape(1, -1)

    if st.button("Predict Night Price"):
        result = model.predict(feature_to_model)
        st.write("The suggested nightly price based on your inputs is $" + str(result[0].round(1)))
        st.balloons()
        percentile = stats.percentileofscore(df.price, result[0])
        ax = sns.distplot(df[df.price<1500].price, kde=False,fit=stats.norm)
        plt.xlim(0,1500)
        plt.axvline(result, c='green',linestyle='--')
        plt.text(result,0.005,' $'  + str(result[0].round(1)) + ' - Pricer then ' + str(round(percentile,0)) + '% of the other listings in NY')
        plt.title('Night Price Plot')

        st.pyplot(plt)


if app_mode=='Resume':
    st.title('Yaniv Schwartz TLDR; Resume') 
    st.markdown("""    
    <h1> Education </h1>

    <h2>MSc in Data Science </h2> <h4>  Eastern University (PA, USA) 2021-2022 </h4>
    <p>
     Learned multiple courses on programming (SQL, Python, R), statistical analysis, BI (Tablau) and Ethics. <br>
     Developed Machine learning system for airbnb price prediction in NY (Final Project)
    </p>


    <h2>  BSc in Industrial Engineering & Information Systems </h2> <h4> Ben-Gurion University of the Negev (Israel) 2009-2013</h4> 
    <p>
     Industrial Engineering and Management, Majored in Information Systems. <br>
    - Developed Machine learning system for stock prices price direction, using social media. (Final Project)<br>
    - 5 courses in Statistics<br>
    - 5 courses in Programming
    </p>

    
    <h1> Work Experience </h1> 
   
    <h2> Senior Data Scientist </h2> <h4> Nov 2021 - Present · 8 mos,  Coinbase, NY </h4> 
    <p>
     Leading Coinbase NFT Data Science team. responsibilities includes creating pipelines, dashboards and statistical analysis into the NFT market
    </p>
    
    <h2> Senior Data Scientist </h2> <h4> Feb 2020 - Nov 2021 · 1 yr 10 mos,  Aetna, NY </h4> 
    <p>
     Led a group of Data scientists, Solving analytical problems for the Medicare members in Aetna.

    * Hypothesis testing, Experiments, Effective marketing campaigns. Led a nationwide campaign efforts with digital/offline marketing, reducing costs and increasing retention and member's health.
    * Retention Modelling which contributed reudction of 2.5% YoY churn
    * Clinical Modelling using healthcare claims data
    * Junior mentoring - Developed coleages with technical mentoring
    * Led problem solving sessions, Innovation session etc.
    </p>

    <h2> Senior Data Scientist </h2> <h4> Feb 2018 - Jan 2020 · 2 yrs,  Matrix-IFS (Deutsche Bank), NY </h4> 
    <p>    
    • Led 2-3 DS juniors development and shaping future progress and deliveries. <br>
    • Driving Innovation: Led Production Model Monitoring framework which saved the department 300k/yr by detecting model drifting and accuracy issues. <br>
    • Led Marketing Operations optimization engine - Maximizing digital campaigns <br>
    efficiency (Emails based) contributing 29% performance increase in digital engagement <br>
    • Partnering with business leaders to understand strategic challenges and delivering solutions and insights, which includes dashboard definitions / KPI / Scorecards. <br>
    </p>

    <h2> Data Scientist </h2> <h4> Aug 2015 - Feb 2018 ·  2 yrs 7 mos,  Matrix-IFS (Deutsche Bank), NY </h4> 
    <p>    
    • Solving fraud modeling sensitivity: Production delivered classification, regression and clustering models to predict trading/fraud activities, and marketing operations <br>
    • Optimizing compliance efforts: Developed cross-departments models supporting bank compliance analysts decisions - saving 20 working hours per case on average, reaching combined savings of $700k/yr. <br>
    • Preprocessing expert – Highly Skilled in ETL pipelines utilizing Python, Hive, SQL and Shell. <br>
    </p>


    <h1> Picture </h1>
     <p>I'm the tired dad on the left </p>""",True) 

    st.image('yaniv_Family.jpg', caption='Me and my family')

    #selected_indices = st.multiselect('Select rows:', df.head(15).index)
    #st.write(selected_indices[0])

#st.title("Airbnb Price Prediction")

# train XGBoost model
#X,y = load_data()
#X_display,y_display = shap.datasets.adult(display=True)
