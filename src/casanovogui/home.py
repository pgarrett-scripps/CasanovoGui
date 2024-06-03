import streamlit as st
import torch

from simple_db import CasanovoDB
from utils import get_database_session

db = get_database_session()

st.title("Casanovo Gui")

st.markdown("Welcome to the Casanovo Gui. This is a simple web interface to interact with Casanovo.")

st.subheader("GPU Availability", divider='blue')
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

st.subheader("Download Example Data", divider='blue')

c1, c2 = st.columns([1, 1])

file_path = r"..\..\tests\data\sample_preprocessed_2spectra.mgf"

c1.download_button(
    label="sample_preprocessed_2spectra.mgf",
    data=file_path,
    file_name="sample_preprocessed_2spectra.mgf",
    mime="text/plain",
    use_container_width=True,
)

file_path = r"..\..\tests\data\sample_preprocessed_128spectra.mgf"

c2.download_button(
    label="sample_preprocessed_128spectra.mgf",
    data=file_path,
    file_name="sample_preprocessed_128spectra.mgf",
    mime="text/plain",
    use_container_width=True,
)


