import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)


st.markdown('''
# **Data Profiler**

*A. Vera*

Use this app to quickly generate a profile report from your input data.  This app uses the  [pandas-profiling](https://pandas-profiling.ydata.ai/docs/master/pages/getting_started/overview.html) library.

---
''')

# Upload CSV data
with st.sidebar.header('Upload your data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input Data**')
    st.write(df)
    st.write('---')
    st.header('**Data Profile**')
    st_profile_report(pr)
    st.balloons()

 # Add download button
    report_name = f"{uploaded_file.name}_report.html"
    report_html = pr.to_html()
    st.download_button(
        label="Download Report",
        data=report_html,
        file_name=report_name,
        mime="text/html"
else:
    st.info('Please, upload your CSV file when ready.')

