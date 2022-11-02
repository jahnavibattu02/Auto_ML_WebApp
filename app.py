import os
import pandas as pd
import streamlit as st
import pandas_profiling
from operator import index
import plotly.express as px
from streamlit_pandas_profiling import st_profile_report 
from pycaret.regression import *
# from pycaret.classification *

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Auto Machine Learning Web Application")
    choice = st.radio("Navigation", ["Upload", "Profile", "Machine Learning", "Download"])

if os.path.exists("data.csv"):
    df = pd.read_csv("data.csv", index_col=None)

if choice == "Upload":
    st.title("Upload")
    st.write("Upload your data here")
    file = st.file_uploader("Upload your data here", type=["csv", "xlsx"])
    if file is not None:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("data.csv", index=False)
        st.dataframe(df) 


elif choice == "Profile":
    st.title("Profile")
    st.write("Profile your data here")
    if st.button("Profile"):
        pr = df.profile_report()
        st_profile_report(pr)

elif choice == "Machine Learning":
    st.title("Machine Learning")
    st.write("Machine Learning your data here")
    target = st.selectbox("Select your target", df.columns)
    if st.button("Train Model"):
        setup(df, target=target,silent=True)
        setup_df = pull()
        st.info("Setup Complete")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.write("Final Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, "best_model")

elif choice == "Download":
    st.title("Download")
    st.write("Download your data here")
    if st.button("Download"):
        with open("best_model.pkl", "wb") as f:
            st.download_button(label="Download Model", data=f, file_name="best_model.pkl")
