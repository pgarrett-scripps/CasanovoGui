import os
import tempfile
import uuid
from datetime import date

import requests
import streamlit as st
import torch

from advanced_db import ModelFileMetadata, ConfigFileMetadata, SpectraFileMetadata
from utils import get_database_session, get_storage_path
from login import display_login_ui, get_current_user

db = get_database_session()

st.title("Casanovo Gui")

st.markdown(
    "Welcome to the Casanovo Gui. This is a simple web interface to interact with Casanovo.")

st.subheader("Database", divider='blue')
st.caption(
    "The database is used to store all the files and metadata used by Casanovo.")
st.write(f'{get_storage_path()}')

st.subheader("GPU Availability", divider='blue')
st.caption("Check if a GPU is available on the current system.")
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


with st.sidebar:
    display_login_ui()

user_data = get_current_user()


st.header("Add Default Data")
if user_data is None:
    st.warning("Please log in to access this page.")
    st.stop()


def download_file(url, filename):
    with st.spinner(f"Downloading {filename}..."):
        response = requests.get(url)
        with open(filename, 'wb') as file:
            file.write(response.content)
    st.success(f"{filename} downloaded!")


def get_128_spectra():
    url = 'https://raw.githubusercontent.com/Noble-Lab/casanovo/main/sample_data/sample_preprocessed_spectra.mgf'

    if not os.path.exists('sample_preprocessed_spectra.mgf'):
        download_file(url, 'sample_preprocessed_spectra.mgf')


def get_2_spectra():
    get_128_spectra()

    # keep first 81 lines
    with open('sample_preprocessed_spectra.mgf', 'r') as f:
        lines = f.readlines()
        with open('sample_preprocessed_2spectra.mgf', 'w') as f2:
            f2.writelines(lines[:81])


@st.fragment
def default_spectra():
    st.subheader("Spectra", divider='blue')

    get_128_spectra()
    get_2_spectra()

    c1, c2 = st.columns([1, 1])
    if c1.button('2 spectra mgf', use_container_width=True):
        metadata = SpectraFileMetadata(
            file_id=str(uuid.uuid4()),
            file_name='Sample 2 Spectra',
            description='From Casanovo Github Repository',
            file_type='mgf',
            date=date.today(),
            tags=['default'],
            enzyme='Unknown',
            instrument='Unknown',
            annotated=True,
        )
        db.spectra_files_manager.add_file(
            'sample_preprocessed_2spectra.mgf', metadata, copy=True, owner_id=user_data['id'])
        st.toast("Spectra added successfully", icon="✅")

    if c2.button('128 spectra mgf', use_container_width=True):
        metadata = SpectraFileMetadata(
            file_id=str(uuid.uuid4()),
            file_name='Sample 128 Spectra',
            description='From Casanovo Github Repository',
            file_type='mgf',
            date=date.today(),
            tags=['default'],
            enzyme='Unknown',
            instrument='Unknown',
            annotated=True,
        )
        db.spectra_files_manager.add_file(
            'sample_preprocessed_spectra.mgf', metadata, copy=True, owner_id=user_data['id'])
        st.toast("Spectra added successfully", icon="✅")


default_spectra()


@st.fragment
def default_models():
    st.subheader('Models', divider='blue')
    nontryptic_link = 'https://github.com/Noble-Lab/casanovo/releases/download/v4.0.0/casanovo_nontryptic.ckpt'
    tryptic_link = 'https://github.com/Noble-Lab/casanovo/releases/download/v4.0.0/casanovo_massivekb.ckpt'

    c1, c2 = st.columns([1, 1])
    if c1.button('Default Tryptic Model', use_container_width=True):
        with st.spinner(f"Downloading {tryptic_link}..."):
            response = requests.get(tryptic_link)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        metadata = ModelFileMetadata(
            file_id=str(uuid.uuid4()),
            file_name='Default Tryptic Model',
            description='v4.0.0',
            file_type='ckpt',
            date=date(2023, 12, 23),
            tags=['default', 'tryptic'],
            source='uploaded',
            status='completed',
            config=None,
        )
        db.models_manager.add_file(
            tmp_path, metadata, copy=False, owner_id=user_data['id'])
        st.toast("Model added successfully", icon="✅")

    if c2.button('Default Nontryptic Model', use_container_width=True):
        with st.spinner(f"Downloading {nontryptic_link}..."):
            response = requests.get(nontryptic_link)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        metadata = ModelFileMetadata(
            file_id=str(uuid.uuid4()),
            file_name='Default Nontryptic Model',
            description='v4.2.0',
            file_type='ckpt',
            date=date(2024, 5, 25),
            tags=['default', 'nontryptic'],
            source='uploaded',
            status='completed',
            config=None,
        )

        db.models_manager.add_file(
            tmp_path, metadata, copy=False, owner_id=user_data['id'])
        st.toast("Model added successfully", icon="✅")


default_models()


@st.fragment
def default_config():
    st.subheader("Config", divider='blue')
    if st.button("Default Config", use_container_width=True):
        config_url = 'https://raw.githubusercontent.com/Noble-Lab/casanovo/main/casanovo/config.yaml'

        with st.spinner(f"Downloading {config_url}..."):
            response = requests.get(config_url)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        metadata = ConfigFileMetadata(
            file_id=str(uuid.uuid4()),
            file_name='Default Config',
            description='From Casanovo Github Repository',
            file_type='yaml',
            date=date.today(),
            tags=['default']
        )

        print(metadata)

        db.config_manager.add_file(tmp_path, metadata, copy=False,
                                   owner_id=user_data['id'])
        st.toast("Config added successfully", icon="✅")


default_config()
