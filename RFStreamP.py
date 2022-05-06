
from ast import Str
import pandas as pd
import numpy as np
import streamlit as st
import pickle

df = pd.read_csv('HRI.csv')

def display_answer(result):
    if result==0:
        st.success("Leave")
    elif result==1:
        st.success("Retain")
    elif result==2:
        st.success("Somehow")
    else:
        pass
    

page = st.sidebar.selectbox("Menu",("Home","Analysis"))
if page=="Home":
    
    st.title("JOB RETAINING PREDUCTION")
    st.subheader("Project description")
    st.write("This project is for helping Human Resource department to predict whether a certain employ tends to leave their current jobs or are to stay. It predicts that basing on different features including employee's salary,country,job title, education... . This will also help the finance department to take decisions accordingly.   ")
    # survey_year = st.number_input("Survey_year",min_value=2000, max_value=2022)
    salary = int(st.number_input("Salary"))
    # Country=st.text_input("Country")
    EmploymenStatus = st.selectbox('Employment Status', set(df['EmploymentStatus']))
    #EmploymentStatus=st.selectbox("EmploymentStatus",["Full time employee","Part Time","Full time employee of a consulting/contracting company","Independent or freelancer or company owner"])
    dkt ={'Full time employee':0,'Part time':4,"Full time employee of a consulting/contracting company":1,"Independent or freelancer or company owner":2,'Independent or freelancer or company owner':3}
    EmploymentStatus=dkt[EmploymenStatus]
    # JobTitle=st.text_input("JobTitle")
    # ManageStaff=st.radio("ManageStaff",["Yes","No"])
    # YearsWithThisTypeOfJob=st.number_input("YearsWithThisTypeOfJob")
    YearsWithThisTypeOfJob=st.slider("Enter you experience",min_value=0,max_value=50)
   
    Education=st.selectbox('Education', set(df["Education"]))
    # Education=st.selectbox("Education",["None (no degree completed)","Associates (2 years)","Bachelors (4 years)","Masters","Doctorate/PhD"])
    dkt1 = {'None (no degree completed)':0,'Associates (2 years)':1,'Bachelors (4 years)':2,'Masters':3,'Doctorate/PhD':4}
    Education =int(dkt1[Education])
    # EducationIsComputerRelated=st.selectbox("EducationIsComputerRelated",["Yes","No","N/A"])
    Certifications=st.selectbox('Certifications', set(df["Certifications"]))
    # Certifications=st.selectbox("Certifications",["Yes, and they're currently valid","Yes, but they expired","No, I never have"])
    dkt2={"Yes, and they're currently valid":0,"Yes, but they expired":1,"No, I never have":2}
    Certifications = int(dkt2[Certifications])
    HoursWorkedPerWeek=st.number_input("HoursWorkedPerWeek")
    # EmploymentSector=st.selectbox("EmploymentSector",["Private business","Non-profit","Local government","State/province government","Education (K-12, college, university)"]) 

    # st.camera_input("", key=None, help=None, on_change=None, args=None, kwargs=None,  disabled=False)
    lm=pickle.load(open('FirstModel.sav','rb'))


    #result=lm.predict([[salary,int(EmploymentStatus),YearsWithThisTypeOfJob,int(Education),int(Certifications),HoursWorkedPerWeek]])[0]
    #st.write(result)

    if st.button("Make Prediction"):
        result=lm.predict([[salary,int(EmploymentStatus),YearsWithThisTypeOfJob,int(Education),int(Certifications),HoursWorkedPerWeek]])[0]
        if result==0:
            st.success("No")
        elif result==1:
            st.success("Yes, actively looking for something else")
        elif result==2:
            st.success("Yes, but only passively (just curious)")
        else:
            pass
       

    # uploaded_files = st.file_uploader("", accept_multiple_files=True)
    # for uploaded_file in uploaded_files:
    #     bytes_data = uploaded_file.read()
    #     st.write("FirstModel.sav", uploaded_file.name)
    #     st.write(bytes_data)

# with st.form(key="form1"):
#     name=st.text_input(label="Enter the model name")
#     number=st.slider("Enter you experience",min_value=10,max_value=50)
#     submit =st.form_submit_button(label="Submit this form")
    
if page=="Analysis": 
    st.subheader("Project description")
    st.write("This project is for helping Human Resource department to predict whether a certain employ tends to leave their current jobs or are to stay. It predicts that basing on different features including employee's salary,country,job title, education... . This will also help the finance department to take decisions accordingly.   ")
    st.subheader("Visualization")
    
    import matplotlib.pyplot as plt
    import numpy as np


    fig, ax = plt.subplots()
    ax.hist(df.SalaryUSD, bins=20)
    st.pyplot(fig)

    st.write("new chart")
    
    st.bar_chart(df['Education'].value_counts()) 
    st.line_chart(df[['Education', 'SalaryUSD']])