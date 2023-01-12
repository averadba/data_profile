import pandas as pd
import pandas_profiling
import streamlit as st

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

@st.cache
def load_data(file):
    return pd.read_csv(file)

def main():
    st.title("Data Profiler")
    st.write("By: A. Vera")
    st.write("Use this app to generate a quick profile of the data contained in your CSV file.")

    # Select the CSV file
    file = st.file_uploader("Upload your CSV file", type="csv")
    if file is None:
        st.error("Please upload a CSV file.")
        return

    # Load the data
    data = load_data(file)

    # Display the data profile
    st.write("Data Profile")
    profile = pandas_profiling.ProfileReport(data)
    st.write(profile)

if __name__== "__main__":
    main()