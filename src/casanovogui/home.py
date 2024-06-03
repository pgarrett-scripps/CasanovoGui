import streamlit as st
import torch

st.title("Casanovo Gui")

st.markdown("Welcome to the Casanovo Gui. This is a simple web interface to interact with Casanovo.")

st.header("Check if GPU is available")
st.caption("This is a simple check to see if a GPU is available.")
with st.echo():
    if torch.cuda.is_available():
        st.write("GPU is available")
    else:
        st.write("GPU is not available")

