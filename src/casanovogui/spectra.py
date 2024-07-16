import os
import tempfile
import uuid
from datetime import date

import streamlit as st
import pandas as pd

from casanovogui.dialogs import delete_option, tag_option, view_option, download_option
from simple_db import SpectraFileMetadata
from utils import refresh_de_key, get_database_session, filter_by_tags


def batch_upload_option(uploaded_files):
    st.subheader("Batch Metadata", divider='blue')
    st.caption("Files will be uploaded in batch, and the metadata will be the same for all files.")
    file_name = st.text_input("File Suffix", value='', disabled=False)
    description = st.text_area("Description")
    date_input = st.date_input("Date", value=date.today())
    tags = sorted(list(set([tag.strip() for tag in st.text_input("Tags (comma-separated)").split(",") if tag])))

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

            get_database_session().spectra_files_manager.add_file(tmp_path, metadata)

        st.rerun()

    if c2.button("Cancel", use_container_width=True):
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
    tags = sorted(list(set([tag.strip() for tag in st.text_input("Tags (comma-separated)").split(",") if tag])))

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

        get_database_session().spectra_files_manager.add_file(tmp_path, metadata)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        st.rerun()


@st.experimental_dialog("Add Files")
def add_option():
    uploaded_files = st.file_uploader("Upload File", type='.mgf', accept_multiple_files=True)

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
    entry.tags = sorted(list(set([tag.strip() for tag in st.text_input("Tags (comma-separated)", value=",".join(entry.tags)).split(",") if tag])))

    c1, c2 = st.columns([1, 1])
    entry.enzyme = c1.text_input("Enzyme", value=entry.enzyme)
    entry.instrument = c2.text_input("Instrument", value=entry.instrument)

    entry.annotated = st.checkbox("Annotated", value=entry.annotated)

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True, disabled=False):
        get_database_session().spectra_files_manager.update_file_metadata(entry)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        st.rerun()


def run():
    # Set up the Streamlit page configuration
    st.set_page_config(page_title="Spectra", layout="wide")
    c1, c2 = st.columns([5, 3])
    c1.title("Spectra")
    st.caption('Small testing mgf files are available on the home page. Otherwise use msconvert to convert vendor raw '
               'files to mgf: https://proteowizard.sourceforge.io/download.html and upload them here.')

    db = get_database_session()
    manager = db.spectra_files_manager

    # Get all file metadata entries
    entries = db.spectra_files_manager.get_all_metadata()
    entries = map(lambda e: e.dict(), entries)
    df = pd.DataFrame(entries)

    if df.empty:
        st.write("No entries found.")
        st.stop()

    rename_map = {
        "file_id": "ID",
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

    df = filter_by_tags(df)

    # Display the editable dataframe
    selection = st.dataframe(df,
                             hide_index=True,
                             column_order=["Name", "Description", "Date", "Tags", "Enzyme",
                                           "Instrument", "Annotated"],
                             column_config={
                                 "Name": st.column_config.TextColumn(disabled=True, width='medium'),
                                 "Description": st.column_config.TextColumn(disabled=True, width='medium'),
                                 "Date": st.column_config.DateColumn(disabled=True, width='small'),
                                 "Tags": st.column_config.ListColumn(width='small'),
                                 "Enzyme": st.column_config.TextColumn(disabled=True, width='small'),
                                 "Instrument": st.column_config.TextColumn(disabled=True, width='small'),
                                 "Annotated": st.column_config.CheckboxColumn(disabled=True, width='small')

                             },
                             use_container_width=True,
                             selection_mode='multi-row', on_select='rerun')

    selected_rows = selection['selection']['rows']
    selected_ids = df.iloc[selected_rows]["ID"].tolist() if selected_rows else []

    c1, c2, c3, c4, c5, c6, c7 = c2.columns(7)

    if c1.button("🗑️", use_container_width=True, disabled=len(selected_ids) == 0, help="Delete selected entries"):
        delete_option(selected_ids, manager)
    if c2.button("📥", use_container_width=True, disabled=len(selected_ids) == 0, help="Download selected entries"):
        download_option(selected_ids, manager)
    if c3.button("🏷️", use_container_width=True, disabled=len(selected_ids) == 0, help="Tag selected entries"):
        tag_option(selected_ids, manager)
    if c4.button("✏️", use_container_width=True, disabled=len(selected_ids) != 1, help="Edit selected entry"):
        edit_option(manager.get_file_metadata(selected_ids[0]))
    if c5.button("👁️", use_container_width=True, disabled=len(selected_ids) != 1, help="View selected entry"):
        view_option(selected_ids[0], manager, mode='file')
    if c6.button("📤", use_container_width=True, help="Upload Spectra"):
        add_option()
    if c7.button("🔄", use_container_width=True, help="Refresh"):
        st.rerun()
