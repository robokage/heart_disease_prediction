import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

st.title("Heart Disease Prediction")

#navigation
st.sidebar.title("Navigation Bar")
nav= st.sidebar.radio("",["Home", "Predict"])

#defining all the non numerical categories
sex_li =["Female","Male"]
cp_li = ["Typical angina","Atypical angina","Non-anginal pain","Asymptomatic"]
fbs_li = ["Less than 120 mg/dl", "More than 120 mg/dl"]
restecg_li = ["Nothing to note","ST-T Wave abnormality", "Possible or definite left ventricular hypertrophy"]
exang_li = ["No", "Yes"]
slope_li =["Upsloping", "Flatsloping", "Downsloping"]

#encoding non-numerical categories
def encode(display):
    options = list(range(len(display)))
    return options

if nav== "Home":
    st.image('heart_image2.jpg',  width=550 )

    st.subheader("Data Required to predict if you have Heart disease or not:")

    st.text_area("",""" 
                1. Age - in years
                2. Sex - (Male/Female)
                3. Chest pain type:
                                # Typical angina
                                # Atypical angina
                                # Non-anginal pain 
                                # Asymptomatic
                4. Resting Blood Pressure
                5. Serum Cholestrol in mg/dl
                6. Fasting Blood Sugar:
                                # Less than 120 mg/dl
                                # More than 120 mg/dl
                7. Resting ECG Results:
                                # Nothing to note
                                # ST-T Wave abnormality
                                # Possible or definite left ventricular hypertrophy
                8. Max Heart Rate Achieved
                9. Do you suffer from exercise induced angina:
                                # No
                                # Yes
                10.Oldpeak
                11.Slope:
                        # Upsloping
                        # Flatsloping
                        # Downsloping
                12.Number of major vessels (0-3) colored by flourosopy
                13.Thalium Stress Results
                """, height=350)
    
else:
    st.markdown("### Fill in the details to make Prediction")
    age, sex, chest_pain = st.beta_columns([1,1,2])
    age= age.number_input("Age", min_value=10, max_value=150, value=50, step=1)
    sex= sex.selectbox("Gender",encode(sex_li), format_func=lambda x: sex_li[x])
    cp = chest_pain.selectbox("Chest Pain type", encode(cp_li),format_func=lambda x: cp_li[x])

    trestbps, chol, fbs =st.beta_columns([1,1,2])
    trestbps= trestbps.number_input("Resting Blood Pressure", min_value=30, max_value=180, value=120, step=1)
    chol = chol.number_input("Serum Cholestrol in mg/dl", min_value=20, max_value=500, value=200, step=1)
    fbs= fbs.selectbox("Fasting Blood Sugar", encode(fbs_li),format_func=lambda x: fbs_li[x])

    restecg, thalach, exang= st.beta_columns([2,1,1])
    restecg= restecg.selectbox("Resting ECG Results",encode(restecg_li),format_func=lambda x: restecg_li[x] )
    thalach= thalach.number_input("Max Heart Rate Achieved", min_value=40, value=175, max_value=300)
    exang =exang.selectbox("Do you suffer from exercise induced angina",encode(exang_li) , index=1,format_func=lambda x: exang_li[x])

    oldpeak, slope, ca = st.beta_columns([1,1,2])
    oldpeak = round(oldpeak.number_input("Oldpeak",max_value=10.0, min_value=0.0, value=2.0),1)
    slope = slope.selectbox("Slope",encode(slope_li), format_func=lambda x: slope_li[x] )
    ca = ca.number_input("Number of major vessels (0-3) colored by flourosopy",max_value=3, min_value=0, value=1, step=1)

    thal, extra = st.beta_columns(2)
    thal=thal.number_input("Thalium Stress Results",min_value=1,max_value=7, step=1, value=2)
    
    submit= st.button("Submit")

    model = pickle.load(open("heart_dis_model.pkl",'rb'))
    dict= {"age":age,"sex":sex, "cp":cp, "trestbps":trestbps, "chol":chol, "fbs":fbs,
           "restecg":restecg, "thalach":thalach, "exang":exang, "oldpeak":oldpeak,
           "slope":slope, "ca":ca, "thal":thal}

    if submit:
        data= pd.DataFrame(dict, index=[0])
        prediction = model.predict(data)
        if prediction==1:
            st.warning("You have Heart Disease. Please consult a doctor ASAP")
            
        else:
            st.success("Congratulations you have no Heart Disease!!")
            st.balloons()

        pred_proba = model.predict_proba(data)
        df= pd.DataFrame(pred_proba,index=[0])
        df_new = df.rename(columns={0: 'Heart Disease Absent', 1:"Heart Disease Present"}, index={0: 'Probability'})
        st.write(df_new)





