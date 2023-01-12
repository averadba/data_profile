import pandas as pd
import pandas_profiling
import streamlit as st

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

def main():
    st.title("Data Profile Generator")
    st.write("By: A. Vera")
    file_upload = st.file_uploader("Upload a CSV file", type="csv")
    if file_upload is not None:
        df = pd.read_csv(file_upload)
        st.write("Data Shape: ", df.shape)
        st.write("Data Columns: ", df.columns)
        profile = pandas_profiling.ProfileReport(df, minimal=True)
        st.markdown(profile.to_html(), unsafe_allow_html=True)

if __name__ == '__main__':
    main()