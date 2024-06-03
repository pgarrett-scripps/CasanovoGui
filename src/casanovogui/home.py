import streamlit as st
import torch

from simple_db import CasanovoDB
from utils import get_database_session

db = get_database_session()

st.title("Casanovo Gui")

st.markdown("Welcome to the Casanovo Gui. This is a simple web interface to interact with Casanovo.")

st.header("Check if GPU is available")
st.caption("This is a simple check to see if a GPU is available.")
with st.echo():
    if torch.cuda.is_available():
        st.write("GPU is available")
    else:
        st.write("GPU is not available")


st.subheader("Queue", divider='blue')

st.caption("Current Tasks in the queue")
st.write(db.current_task)

st.caption("Queued Tasks")
st.write(db.get_queued_tasks())


