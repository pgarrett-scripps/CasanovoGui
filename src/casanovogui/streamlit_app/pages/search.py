import os
import tempfile
import uuid
from datetime import date

import streamlit as st
import pandas as pd

from casanovogui.simple_db import SearchMetadata
from casanovogui.streamlit_app.utils import refresh_de_key, get_database_session

PAGE_KEY = 'SEARCH'
PAGE_DE_KEY = f"{PAGE_KEY}_de_key"
SUPPORTED_FILES = ['.mztab']

# Set up the Streamlit page configuration
st.set_page_config(page_title="Search", layout="wide")
st.title("Search")

if PAGE_DE_KEY not in st.session_state:
    refresh_de_key(PAGE_DE_KEY)

db = get_database_session()
manager = db.searches_manager

# Streamlit app layout

# Get all file metadata entries
entries = manager.get_all_metadata()
entries = map(lambda e: e.dict(), entries)
df = pd.DataFrame(entries)

def single_upload(uploaded_file):
    base_file_name, file_extension = os.path.splitext(uploaded_file.name)
    file_extension = file_extension.lstrip(".")

    c1, c2 = st.columns([8, 2])
    file_name = c1.text_input("Base File Name", value=base_file_name, disabled=False)
    file_type = c2.text_input("File Extension", value=file_extension, disabled=True)

    description = st.text_area("Description")
    date_input = st.date_input("Date", value=date.today())
    tags = st.text_input("Tags (comma-separated)").split(",")

    model_ids = db.models_manager.get_all_metadata()
    df = pd.DataFrame(map(lambda e: e.dict(), model_ids))
    selection = st.dataframe(df, selection_mode='single-row', on_select='rerun', hide_index=True)
    selected_index = selection['selection']['rows'][0] if selection['selection']['rows'] else None
    if selected_index is None:
        model_id = None
    else:
        model_id = model_ids[selected_index].file_id

    spectra_ids = db.spectra_files_manager.get_all_metadata()
    df = pd.DataFrame(map(lambda e: e.dict(), spectra_ids))
    selection = st.dataframe(df, selection_mode='single-row', on_select='rerun', hide_index=True)
    selected_index = selection['selection']['rows'][0] if selection['selection']['rows'] else None
    if selected_index is None:
        spectra_id = None
    else:
        spectra_id = spectra_ids[selected_index].file_id

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        metadata = SearchMetadata(
            file_id=str(uuid.uuid4()),
            file_name=file_name,
            description=description,
            file_type=file_type,
            date=date_input,
            tags=tags,
            model_id=model_id,
            spectra_id=spectra_id,
            status="completed"
        )

        manager.add_file(tmp_path, metadata)
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


def batch_upload(uploaded_files):
    file_name = st.text_input("File Suffix", value='', disabled=False)
    description = st.text_area("Description")
    date_input = st.date_input("Date", value=date.today())
    tags = st.text_input("Tags (comma-separated)").split(",")

    c1, c2 = st.columns(2)
    if c1.button("Submit", type='primary', use_container_width=True, disabled=len(uploaded_files) == 0):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            base_file_name, file_extension = os.path.splitext(uploaded_file.name)
            file_extension = file_extension.lstrip(".")

            metadata = SearchMetadata(
                file_id=str(uuid.uuid4()),  # str(uuid.uuid4()
                file_name=file_name + base_file_name,
                description=description,
                file_type=file_extension,
                date=date_input,
                tags=tags,
                model_id=None,
                spectra_id=None,
                status="completed"
            )

            manager.add_file(tmp_path, metadata)

        refresh_de_key(PAGE_DE_KEY)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


@st.experimental_dialog("Upload File")
def upload_option():
    uploaded_file = st.file_uploader("Upload File", type=SUPPORTED_FILES, accept_multiple_files=True)
    if uploaded_file and len(uploaded_file) == 1:
        single_upload(uploaded_file[0])
    elif uploaded_file and len(uploaded_file) > 1:
        batch_upload(uploaded_file)
    else:
        st.warning("Please upload a file.")


@st.experimental_dialog("New Search")
def add_option():

    st.subheader("Search Metadata", divider='blue')

    c1, c2 = st.columns([8, 2])
    file_name = c1.text_input("Base File Name", value='My Search', disabled=False)
    file_type = c2.text_input("File Extension", value='mztab', disabled=True)
    description = st.text_area("Description")
    date_input = st.date_input("Date", value=date.today())
    tags = st.text_input("Tags (comma-separated)").split(",")

    model_id = None
    spectra_id = None

    st.subheader("Select Model", divider='blue')
    model_ids = db.models_manager.get_all_metadata()
    df = pd.DataFrame(map(lambda e: e.dict(), model_ids))
    selection = st.dataframe(df, selection_mode='single-row', on_select='rerun', hide_index=True)
    selected_index = selection['selection']['rows'][0] if selection['selection']['rows'] else None
    if selected_index is None:
        st.info("Selected Model: Default")
    else:
        model_id = model_ids[selected_index].file_id
        model_name = model_ids[selected_index].file_name
        st.info(f"Selected Model: {model_name}")

    st.subheader("Select Spectra", divider='blue')
    spectra_ids = db.spectra_files_manager.get_all_metadata()
    df = pd.DataFrame(map(lambda e: e.dict(), spectra_ids))
    selection = st.dataframe(df, selection_mode='single-row', on_select='rerun', hide_index=True)
    selected_index = selection['selection']['rows'][0] if selection['selection']['rows'] else None
    if selected_index is None:
        st.warning("Please select a spectra file.")
    else:
        spectra_id = spectra_ids[selected_index].file_id
        spectra_name = spectra_ids[selected_index].file_name
        st.info(f"Selected Spectra: {spectra_name}")

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True, disabled=not spectra_id):
        metadata = SearchMetadata(
            file_id=str(uuid.uuid4()),
            file_name=file_name,
            description=description,
            file_type=file_type,
            date=date_input,
            tags=tags,
            model_id=model_id,
            spectra_id=spectra_id,
            status="Pending"
        )

        db.search(metadata)
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()
    if c2.button("Cancel", use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


@st.experimental_dialog("Edit Search Metadata")
def edit_option(entry: SearchMetadata):

    st.subheader("Search Metadata", divider='blue')
    c1, c2 = st.columns([8, 2])
    entry.file_name = c1.text_input("File Name", value=entry.file_name, disabled=False)
    entry.file_type = c2.text_input("File Type", value=entry.file_type, disabled=True)
    entry.description = st.text_input("Description", value=entry.description)
    entry.date = st.date_input("Date", value=pd.to_datetime(entry.date).date())
    entry.tags = st.text_input("Tags (comma-separated)", value=",".join(entry.tags)).split(",")

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True):
        manager.update_file_metadata(entry)
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()
    if c2.button("Cancel", use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


@st.experimental_dialog("Delete Entry")
def delete_option(file_id: str):
    st.write("Are you sure you want to delete this entry?")
    c1, c2 = st.columns([1, 1])
    if c1.button("Delete", use_container_width=True):
        db.delete_search(file_id)
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()
    if c2.button("Cancel", type='primary', use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


@st.experimental_dialog("Download Results")
def download_option(file_id: str):
    file_path = manager.retrieve_file_path(file_id)
    file_metadata = manager.get_file_metadata(file_id)

    file_option = st.radio("File Type", ["result", "log"], horizontal=True)

    if file_option == "result":
        download_name = st.text_input("Download Name", value=file_metadata.file_name + ".mztab")
    elif file_option == "log":
        file_path = file_path.replace(".mztab", ".log")
        download_name = st.text_input("Download Name", value=file_metadata.file_name + ".log")

    path_exists = os.path.exists(file_path)
    # check if path exists
    if not path_exists:
        st.error(f"File not found at: {file_path}")

    c1, c2 = st.columns([1, 1])
    btn = c1.download_button(
        label="Download",
        data=open(file_path, "rb") if path_exists else '',
        file_name=download_name,
        mime='application/octet-stream',
        use_container_width=True,
        type='primary',
        disabled=not path_exists
    )
    if btn:
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()
    if c2.button("Cancel", use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


# Display buttons for add and refresh
c1, c2, c3 = st.columns([2,2,1])
if c1.button("New Search", use_container_width=True, type='primary'):
    add_option()
if c2.button("Refresh", use_container_width=True):
    refresh_de_key(PAGE_DE_KEY)
    st.rerun()
if c3.button("Upload Files", use_container_width=True):
    upload_option()

if df.empty:
    st.write("No entries found.")
    st.stop()

rename_map = {
    "file_id": "File ID",
    "file_name": "Name",
    "description": "Description",
    "date": "Date",
    "tags": "Tags",
    "model_id": "Model ID",
    "spectra_id": "Spectra ID",
    "status": "Status"
}

# Customize the dataframe for display
df.rename(columns=rename_map, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df["✏️"] = False
df['🗑️'] = False
df['📥'] = False
df['👁️'] = False

df['Model Name'] = df['Model ID'].apply(lambda x: db.models_manager.get_file_metadata(x).file_name if x else None)
df['Spectra Name'] = df['Spectra ID'].apply(lambda x: db.spectra_files_manager.get_file_metadata(x).file_name)

# Display the editable dataframe
edited_df = st.data_editor(df,
                           hide_index=True,
                           column_order=["✏️", "🗑️", "📥", "👁️", "Name", "Description", "Date", "Tags", "Model Name",
                                         "Spectra Name", "Status"],
                           column_config={
                               "✏️": st.column_config.CheckboxColumn(disabled=False, width='small'),
                               "🗑️": st.column_config.CheckboxColumn(disabled=False, width='small'),
                               "📥": st.column_config.CheckboxColumn(disabled=False, width='small'),
                               "👁️": st.column_config.CheckboxColumn(disabled=False, width='small'),
                               "Name": st.column_config.TextColumn(disabled=True, width='medium'),
                               "Description": st.column_config.TextColumn(disabled=True, width='medium'),
                               "Date": st.column_config.DateColumn(disabled=True, width='small'),
                               "Tags": st.column_config.ListColumn(width='small'),
                               "Model Name": st.column_config.TextColumn(disabled=True, width='small'),
                               "Spectra Name": st.column_config.TextColumn(disabled=True, width='small'),
                               "Status": st.column_config.TextColumn(disabled=True, width='small')
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
        entry = manager.get_file_metadata(entry_to_edit["File ID"])
        edit_option(entry)

    if "🗑️" in edited_row and edited_row["🗑️"] is True:
        delete_option(df.iloc[row_index]["File ID"])

    if "📥" in edited_row and edited_row["📥"] is True:
        download_option(df.iloc[row_index]["File ID"])

    if "👁️" in edited_row and edited_row["👁️"] is True:
        # switch to the viewer page
        refresh_de_key(PAGE_DE_KEY)
        st.session_state['viewer_Search_id'] = df.iloc[row_index]["File ID"]
        st.switch_page("pages/viewer.py")
else:
    refresh_de_key(PAGE_DE_KEY)
    st.rerun()
