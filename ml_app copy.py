import streamlit as st
import numpy as np

#Machine Learning
import joblib
import os

attribute_info = """
                 1. Age: 18 - 90
                 2. Job: Type of job
                 3. Marital: Marital status
                 4. Education: Educational stages
                 5. Default: Has credit in default?
                 6. Balance: Average yearly balance, in euros
                 7. Housing: Has housing loan?
                 8. Loan: Has personal loan?
                 #### Related with last contact of the Term Deposit Campaign ####
                 9. Contact: Contact communication type
                 10. Day: Last contacted day, 1 - 30 (days of month)
                 11. Month: Last contacted month, 1 - 12 (months of year)
                 12. Duration: Last contacted duration, in seconds
                 13. Campaign: Number of contacts performed during this campaign on the client
                 14. Pdays: Number of days has passed after the client was last contacted from the previous campaign
                 15. Previous: Number of contacts performed on the previous campaign for the client
                 16. Poutcome: Outcome of the previous marketing campaign
                 """

# Categorical Option
job_list = ["Admin", "Blue-Collar", "Entrepreneur", "Housemaid", "Management", "Retired", "Self-Employed", "Services", "Student", "Technician", "Unemployed", "Unknown"]
jobs = {"Admin":0, "Blue-Collar":1, "Entrepreneur":2, "Housemaid":3, 
        "Management":4, "Retired":5, "Self-Employed":6, "Services":7, 
        "Student":8, "Technician":9, "Unemployed":10, "Unknown":11}
mar_list = ["Divorced", "Married", "Single"]
mar = {"Divorced":0, "Married":1, "Single":2}
edu_list = ["Primary", "Secondary", "Tertiary", "Unknown"]
edu = {"Primary":0, "Secondary":1, "Tertiary":2, "Unknown":3}
yn_list = ["No", "Yes"]
yn = {"No":0, "No":1}
con_list = ["Unknown", "Telephone", "Cellular"]
con = {"Unknown":0, "Telephone":1, "Cellular":2}
out_list = ["Unknown", "Other", "Failure", "Success"]
out = {"Unknown":0, "Other":1, "Failure":2, "Success":3}


def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

@st.cache
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),'rb'))
    return loaded_model    

def run_ml_app():
    st.subheader("ML Section")

    with st.expander('Attribute Info'):
         st.markdown(attribute_info)

    st.subheader("Input the Data")
    age = st.number_input("1. Age",18,90)
    job = st.selectbox("2. Job",job_list)
    marital = st.selectbox("3. Marital status",mar_list)
    education = st.selectbox("4. Educational stage",edu_list)
    default = st.selectbox("5. Have credit in default?",yn_list)
    balance = st.number_input("6. Average yearly balance")
    housing = st.selectbox("7. Has housing loan?",yn_list)
    loan = st.selectbox("8. Has personal loan?",yn_list)
    contact = st.selectbox("9. Contact communication type?",con_list)
    day = st.number_input("10. Last contacted (day of month = 1 - 30/31)",1,31)
    month = st.number_input("11. Last contacted (month of year = 1 - 12)",1,12)
    duration = st.number_input("12. Duration at last contact (in seconds)")
    campaign = st.number_input("13. Number of contacts performed during this campaign")
    is_pdays = st.selectbox("14. Have you ever been contacted from the previous campaign?", ["Yes", "No"])
    if is_pdays == "Yes":
        pdays = st.number_input("How many days from the last contacted?",0,900)
    else:
        pdays = -1
    previous = st.number_input("15. Number of contacts from the previous campaign")
    poutcome = st.selectbox("16. Outcome of the previous campaign",out_list)
    

    with st.expander("Your Selected Options"):
        result = {
            "Age":age,
            "Job":job,
            "Marital":marital,
            "Education":education,
            "Default":default,
            "Balance":balance,
            "Housing":housing,
            "Loan":loan,
            "Contact":contact,
            "Day":day,
            "Month":month,
            "Duration":duration,
            "Campagin":campaign,
            "Pdays":pdays,
            "Previous":previous,
            "Poutcome":poutcome,
        }
    
    # st.write(result)

    encoded_result = []
    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)
        elif i in job_list:
            res = get_value(i, jobs)
            encoded_result.append(res)
        elif i in mar_list:
            res = get_value(i, mar)
            encoded_result.append(res)
        elif i in edu_list:
            res = get_value(i, edu)
            encoded_result.append(res)
        elif i in yn_list:
            res = get_value(i, yn)
            encoded_result.append(res)
        elif i in con_list:
            res = get_value(i, con)
            encoded_result.append(res)
        elif i in out_list:
            res = get_value(i, out)
            encoded_result.append(res)
    
    # st.write(encoded_result)

    st.subheader("Prediction Result")
    single_sample = np.array(encoded_result).reshape(1,-1)
    # st.write(single_sample)

    model = load_model("model_xgb.pkl")

    prediction = model.predict(single_sample)
    pred_prob = model.predict_proba(single_sample)

    # st.subheader("Prediction Absolute")
    # st.write(prediction)
    # st.subheader("Prediction Probability")
    # st.write(pred_prob)

    pred_probability = {'Subscribe':round(pred_prob[0][1]*100,4),
                        'Not Subscribe':round(pred_prob[0][0]*100,4)}
    if prediction == 1:
        st.success("The Client is Subscribe")
        st.write(pred_probability)
    else:
        st.warning("The Client is Not Subscribe")
        st.write(pred_probability)