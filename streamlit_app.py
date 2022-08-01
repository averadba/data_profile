import pandas as pd
import pandas_profiling
import streamlit as st

from streamlit_pandas_profiling import st_profile_report

st.title("Data Profiler")
st.write("by: A. Vera")
st.markdown("The data profiler is a simple tool to take a quick glance at your data.")

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], help="Only csv files are accepted. Make sure the first row contains column names.")
if uploaded_file is not None:
     # To read file as bytes:
     bytes_data = uploaded_file.getvalue()
     st.write(bytes_data)


df =pd.read_csv(uploaded_file)
#df = pd.DataFrame(uploaded_file,sep=",")

if st.button("Generate Profile Report"):
    pr = df.profile_report()

    st_profile_report(pr)