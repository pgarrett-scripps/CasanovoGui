# This page should show the current processes

import streamlit as st

from casanovogui.simple_db import CasanovoDB
from casanovogui.streamlit_app.utils import DATA_FOLDER

db = CasanovoDB(DATA_FOLDER)

st.title("Processes")

processes = db.get_queued_tasks()

st.write(db.queue.qsize())

for process in processes:
    st.write(process)

