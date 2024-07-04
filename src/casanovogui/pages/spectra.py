import os
import tempfile
import uuid
from datetime import date

import streamlit as st
import pandas as pd

from simple_db import SpectraFileMetadata
from utils import refresh_de_key, get_database_session, filter_by_tags

PAGE_KEY = 'FILES'
PAGE_DE_KEY = f"{PAGE_KEY}_de_key"
SUPPORTED_FILES = ['.mgf']

# Set up the Streamlit page configuration
st.set_page_config(page_title="Spectra", layout="wide")
st.title("Spectra")
st.caption('Small testing mgf files are available on the home page. Otherwise use msconvert to convert vendor raw '
           'files to mgf: https://proteowizard.sourceforge.io/download.html and upload them here.')

if PAGE_DE_KEY not in st.session_state:
    refresh_de_key(PAGE_DE_KEY)

db = get_database_session()

# Get all file metadata entries
entries = db.spectra_files_manager.get_all_metadata()
entries = map(lambda e: e.dict(), entries)
df = pd.DataFrame(entries)


def batch_upload_option(uploaded_files):
    st.subheader("Batch Metadata", divider='blue')
    st.caption("Files will be uploaded in batch, and the metadata will be the same for all files.")
    file_name = st.text_input("File Suffix", value='', disabled=False)
    description = st.text_area("Description")
    date_input = st.date_input("Date", value=date.today())
    tags = st.text_input("Tags (comma-separated)").split(",")

    c1, c2 = st.columns([1, 1])
    enzyme = c1.text_input("Enzyme")
    instrument = c2.text_input("Instrument")

    annotated = st.checkbox("Annotated")

    if c1.button("Submit", type='primary', use_container_width=True, disabled=len(uploaded_files) == 0):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            base_file_name, file_extension = os.path.splitext(uploaded_file.name)
            file_extension = file_extension.lstrip(".")

            metadata = SpectraFileMetadata(
                file_id=str(uuid.uuid4()),  # str(uuid.uuid4()
                file_name=file_name + base_file_name,
                description=description,
                file_type=file_extension,
                date=date_input,
                tags=tags,
                enzyme=enzyme,
                instrument=instrument,
                annotated=annotated
            )

            db.spectra_files_manager.add_file(tmp_path, metadata)

        refresh_de_key(PAGE_DE_KEY)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


def single_option(uploaded_file):

    st.subheader("Spectra Metadata", divider='blue')
    c1, c2 = st.columns([7, 2])
    base_file_name, file_extension = os.path.splitext(uploaded_file.name)
    file_extension = file_extension.lstrip(".")
    file_name = c1.text_input("File Name", value=base_file_name, disabled=False)
    file_type = c2.text_input("File Type", value=file_extension, disabled=True)

    description = st.text_area("Description")
    date_input = st.date_input("Date", value=date.today())
    tags = st.text_input("Tags (comma-separated)").split(",")

    c1, c2 = st.columns([1, 1])
    enzyme = c1.text_input("Enzyme")
    instrument = c2.text_input("Instrument")

    annotated = st.checkbox("Annotated")

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True, disabled=uploaded_file is None):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        metadata = SpectraFileMetadata(
            file_id=str(uuid.uuid4()),  # str(uuid.uuid4()
            file_name=file_name,
            description=description,
            file_type=file_type,
            date=date_input,
            tags=tags,
            enzyme=enzyme,
            instrument=instrument,
            annotated=annotated
        )

        db.spectra_files_manager.add_file(tmp_path, metadata)
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


@st.experimental_dialog("Add Files")
def add_option():
    uploaded_files = st.file_uploader("Upload File", type=SUPPORTED_FILES, accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) == 1:
        single_option(uploaded_files[0])
    elif uploaded_files and len(uploaded_files) > 1:
        batch_upload_option(uploaded_files)
    else:
        st.warning("Please upload at least one file.")


@st.experimental_dialog("Edit Metadata")
def edit_option(entry: SpectraFileMetadata):

    st.subheader("Spectra Metadata", divider='blue')
    c1, c2 = st.columns([7, 2])
    entry.file_name = c1.text_input("File Name", value=entry.file_name, disabled=False)
    entry.file_type = c2.text_input("File Type", value=entry.file_type, disabled=True)
    entry.description = st.text_area("Description", value=entry.description)
    entry.date = st.date_input("Date", value=entry.date)
    entry.tags = st.text_input("Tags (comma-separated)", value=",".join(entry.tags)).split(",")
    c1, c2 = st.columns([1, 1])
    entry.enzyme = c1.text_input("Enzyme", value=entry.enzyme)
    entry.instrument = c2.text_input("Instrument", value=entry.instrument)

    entry.annotated = st.checkbox("Annotated", value=entry.annotated)

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True, disabled=False):
        db.spectra_files_manager.update_file_metadata(entry)
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
        db.spectra_files_manager.delete_file(file_id)
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()
    elif c2.button("Cancel", type='primary', use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


@st.experimental_dialog("Download Entry")
def download_option(file_id: str):
    st.write("Download the file:")
    file_path = db.spectra_files_manager.retrieve_file_path(file_id)
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
c1, c2 = st.columns(2)
if c1.button("Add Files", use_container_width=True, type='primary'):
    add_option()
if c2.button("Refresh", use_container_width=True):
    refresh_de_key(PAGE_DE_KEY)
    st.rerun()


if df.empty:
    st.write("No entries found.")
    st.stop()

rename_map = {
    "file_id": "File ID",
    "file_name": "Name",
    "description": "Description",
    "date": "Date",
    "tags": "Tags",
    "enzyme": "Enzyme",
    "instrument": "Instrument",
    "annotated": "Annotated"
}

# Customize the dataframe for display
df.rename(columns=rename_map, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df["✏️"] = False
df['🗑️'] = False
df['📥'] = False


df = filter_by_tags(df)

# Display the editable dataframe
edited_df = st.data_editor(df,
                           hide_index=True,
                           column_order=["✏️", "🗑️", "📥", "Name", "Description", "Date", "Tags", "Enzyme",
                                         "Instrument", "Annotated"],
                           column_config={
                               "✏️": st.column_config.CheckboxColumn(disabled=False, width='small'),
                               "🗑️": st.column_config.CheckboxColumn(disabled=False, width='small'),
                               "📥": st.column_config.CheckboxColumn(disabled=False, width='small'),
                               "Name": st.column_config.TextColumn(disabled=True, width='medium'),
                               "Description": st.column_config.TextColumn(disabled=True, width='medium'),
                               "Date": st.column_config.DateColumn(disabled=True, width='small'),
                               "Tags": st.column_config.ListColumn(width='small'),
                               "Enzyme": st.column_config.TextColumn(disabled=True, width='small'),
                               "Instrument": st.column_config.TextColumn(disabled=True, width='small'),
                                "Annotated": st.column_config.CheckboxColumn(disabled=True, width='small')

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
        entry = db.spectra_files_manager.get_file_metadata(entry_to_edit["File ID"])
        edit_option(entry)

    if "🗑️" in edited_row and edited_row["🗑️"] is True:
        delete_option(df.iloc[row_index]["File ID"])

    if "📥" in edited_row and edited_row["📥"] is True:
        download_option(df.iloc[row_index]["File ID"])
else:
    refresh_de_key(PAGE_DE_KEY)
    st.rerun()