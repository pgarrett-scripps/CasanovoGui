import os
import tempfile
import uuid
from datetime import date

import streamlit as st
import pandas as pd

from simple_db import ModelFileMetadata
from utils import refresh_de_key, get_database_session, filter_by_tags

PAGE_KEY = 'MODELS'
PAGE_DE_KEY = f"{PAGE_KEY}_de_key"
SUPPORTED_FILES = ['.ckpt']

# Set up the Streamlit page configuration
st.set_page_config(page_title=f"Model", layout="wide")
st.title(f"Model")
st.caption('Default models can be downloaded from the home page. Otherwise models can be uploaded or trained. '
           'See available models at https://github.com/Noble-Lab/casanovo/releases.')

if PAGE_DE_KEY not in st.session_state:
    refresh_de_key(PAGE_DE_KEY)

db = get_database_session()
manager = db.models_manager

# Streamlit app layout

# Get all file metadata entries
entries = manager.get_all_metadata()
entries = map(lambda e: e.dict(), entries)
df = pd.DataFrame(entries)


@st.experimental_dialog("Train Model", width='large')
def train_option():
    # select multiple annotated files

    t1, t2, t3 = st.tabs(["Metadata", "Spectra", "Config"])
    selected_spectra_ids = []
    selected_config = None

    with t1:
        c1, c2 = st.columns([7, 2])
        file_name = c1.text_input("File Name", value='', disabled=False)
        file_type = c2.text_input("File Type", value='ckpt', disabled=True)

        description = st.text_area("Description")
        date_input = st.date_input("Date", value=date.today())
        tags = st.text_input("Tags (comma-separated)").split(",")

    with t2:
        st.caption("Select annotated spectra to train the model.")
        spectra_metadata = db.spectra_files_manager.get_all_metadata()

        spectra_df = pd.DataFrame(map(lambda e: e.dict(), spectra_metadata))
        if len(spectra_df) > 0:
            spectra_df = spectra_df[spectra_df['annotated'] == True]

        spectra_df = filter_by_tags(spectra_df, 'tags', key='Dialog_Model_Spectra_Filter')

        selection = st.dataframe(spectra_df, on_select='rerun', selection_mode='multi-row', use_container_width=True)
        selected_rows = list(selection['selection']['rows'])
        selected_spectra_ids = spectra_df.iloc[selected_rows]['file_id'].tolist()

    with t3:
        st.caption("Select a config file to train the model.")
        config_metadata = db.config_manager.get_all_metadata()
        config_df = pd.DataFrame(map(lambda e: e.dict(), config_metadata))
        config_df = filter_by_tags(config_df, 'tags', key='Dialog_Model_Config_Filter')
        selection = st.dataframe(config_df, on_select='rerun', selection_mode='single-row', use_container_width=True)
        selected_row = selection['selection']['rows'][0] if len(selection['selection']['rows']) > 0 else None
        selected_config = config_df.iloc[selected_row]['file_id'] if selected_row is not None else None

    if not selected_spectra_ids:
        st.warning("No annotated spectra selected.")

    if not selected_config:
        st.warning("No config selected.")

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True,
                 disabled=len(selected_spectra_ids) == 0 or selected_config is None):
        # db.train(self, spectra_ids: list[str], config_id: Optional[str], model_metadata: ModelFileMetadata) -> str:
        db.train(selected_spectra_ids, selected_config, ModelFileMetadata(
            file_id=str(uuid.uuid4()),
            file_name=file_name,
            description=description,
            file_type=file_type,
            date=date_input,
            tags=tags,
            source='trained',
            status='pending',
            config=selected_config,
        ))

        refresh_de_key(PAGE_DE_KEY)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


@st.experimental_dialog("Add Model")
def add_option():
    uploaded_file = st.file_uploader("Upload Model", type=SUPPORTED_FILES)

    if uploaded_file:
        st.subheader("Model Metadata", divider='blue')
        base_file_name, file_extension = os.path.splitext(uploaded_file.name)
        file_extension = file_extension.lstrip(".")
        c1, c2 = st.columns([7, 2])
        file_name = c1.text_input("File Name", value=base_file_name, disabled=False)
        file_type = c2.text_input("File Type", value=file_extension, disabled=True)

        description = st.text_area("Description")
        date_input = st.date_input("Date", value=date.today())
        tags = st.text_input("Tags (comma-separated)").split(",")

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True, disabled=not uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        metadata = ModelFileMetadata(
            file_id=str(uuid.uuid4()),
            file_name=file_name,
            description=description,
            file_type=file_type,
            date=date_input,
            tags=tags,
            source='uploaded',
            status='completed',
            config=None,
        )

        manager.add_file(tmp_path, metadata)
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


@st.experimental_dialog("Edit Model Metadata")
def edit_option(entry: ModelFileMetadata):
    st.subheader("Model Metadata", divider='blue')

    c1, c2 = st.columns([7, 2])
    entry.file_name = c1.text_input("File Name", value=entry.file_name, disabled=False)
    entry.file_type = c2.text_input("File Type", value=entry.file_type, disabled=True)

    entry.description = st.text_area("Description", value=entry.description)
    entry.date = st.date_input("Date", value=entry.date)
    entry.tags = st.text_input("Tags (comma-separated)", value=",".join(entry.tags)).split(",")

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True):
        manager.update_file_metadata(entry)
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


@st.experimental_dialog("Delete Model")
def delete_option(file_id: str):
    st.write("Are you sure you want to delete this entry?")
    c1, c2 = st.columns([1, 1])
    if c1.button("Delete", use_container_width=True):
        manager.delete_file(file_id)
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()
    if c2.button("Cancel", type='primary', use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


@st.experimental_dialog("Download Model")
def download_option(file_id: str):
    st.write("Download the file:")
    file_path = manager.retrieve_file_path(file_id)
    c1, c2 = st.columns([1, 1])
    with open(file_path, "rb") as file:
        btn = c1.download_button(
            label="Download",
            data=file,
            file_name=os.path.basename(file_path),
            mime='application/octet-stream',
            use_container_width=True,
            type='primary'
        )
    if btn:
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()
    if c2.button("Cancel", use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


# Display buttons for add and refresh
c1, c2, c3 = st.columns([2, 2, 1])
if c1.button("Add Model", use_container_width=True, type='primary'):
    add_option()
if c2.button("Train Model", use_container_width=True):
    train_option()
if c3.button("Refresh", use_container_width=True):
    refresh_de_key(PAGE_DE_KEY)
    st.rerun()

if df.empty:
    st.write("No entries found.")
    st.stop()

rename_map = {
    "file_id": "ID",
    "file_name": "Name",
    "description": "Description",
    "date": "Date",
    "tags": "Tags",
    "source": "Source",
    "status": "Status",
    "config": "Config"
}

# Customize the dataframe for display
df.rename(columns=rename_map, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df["✏️"] = False
df['🗑️'] = False
df['📥'] = False
df['👁️'] = False

df = filter_by_tags(df)

# Display the editable dataframe
edited_df = st.data_editor(df,
                           hide_index=True,
                           column_order=["✏️", "🗑️", "📥", "👁️", "Name", "Description", "Date", "Tags", "Source",
                                         "Status", "Config"],
                           column_config={
                               "✏️": st.column_config.CheckboxColumn(disabled=False, width='small'),
                               "🗑️": st.column_config.CheckboxColumn(disabled=False, width='small'),
                               "📥": st.column_config.CheckboxColumn(disabled=False, width='small'),
                               "👁️": st.column_config.CheckboxColumn(disabled=False, width='small'),
                               "Name": st.column_config.TextColumn(disabled=True, width='medium'),
                               "Description": st.column_config.TextColumn(disabled=True, width='medium'),
                               "Date": st.column_config.DateColumn(disabled=True, width='small'),
                               "Tags": st.column_config.ListColumn(width='small'),
                               "Source": st.column_config.TextColumn(disabled=True, width='small'),
                               "Status": st.column_config.TextColumn(disabled=True, width='small'),
                               "Config": st.column_config.TextColumn(disabled=True, width='small')
                           },
                           key=st.session_state[PAGE_DE_KEY],
                           use_container_width=True)

# Handle edited rows
edited_rows = st.session_state[st.session_state[PAGE_DE_KEY]]['edited_rows']
if len(edited_rows) == 0:
    pass
elif len(edited_rows) == 1:

    row_index, edited_row = list(edited_rows.items())[0]

    if len(edited_row) > 1:
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()

    if "✏️" in edited_row and edited_row["✏️"] is True:
        entry_to_edit = df.iloc[row_index].to_dict()
        entry = manager.get_file_metadata(entry_to_edit["ID"])
        edit_option(entry)

    if "🗑️" in edited_row and edited_row["🗑️"] is True:
        delete_option(df.iloc[row_index]["ID"])

    if "📥" in edited_row and edited_row["📥"] is True:
        download_option(df.iloc[row_index]["ID"])

    if "👁️" in edited_row and edited_row["👁️"] is True:
        entry_to_edit = df.iloc[row_index].to_dict()
        entry = manager.get_file_metadata(entry_to_edit["ID"])

        if 'viewed_file' not in st.session_state:
            st.session_state['viewed_file'] = entry.file_id

        refresh_de_key(PAGE_DE_KEY)
        st.rerun()

else:
    refresh_de_key(PAGE_DE_KEY)
    st.rerun()

if 'viewed_file' in st.session_state:
    entry = manager.get_file_metadata(st.session_state['viewed_file'])
    file_path = manager.retrieve_file_path(entry.file_id)
    log_file = file_path.replace('.ckpt', '.log')

    del st.session_state['viewed_file']

    if not os.path.exists(log_file):
        st.write("No log file found.")
        st.stop()

    st.subheader(f"Name: {entry.file_name}.log", divider='blue')
    with open(log_file, "rb") as file:
        st.code(file.read().decode(), language='txt')
