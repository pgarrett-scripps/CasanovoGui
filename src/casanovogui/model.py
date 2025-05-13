import os
import tempfile
import uuid
from datetime import date

import streamlit as st
import pandas as pd

from login import display_login_ui, get_current_user
from dialogs import download_option, tag_option, view_option, delete_option
from simple_db import ModelFileMetadata
from utils import get_database_session, filter_by_tags, get_config_filename


@st.dialog("Train Model", width='large')
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
        tags = [tag for tag in st.text_input(
            "Tags (comma-separated)").split(",") if tag]

    with t2:
        st.caption("Select annotated spectra to train the model.")
        spectra_metadata = get_database_session(
        ).spectra_files_manager.get_all_metadata(get_current_user()['id'])

        if spectra_metadata:
            spectra_df = pd.DataFrame(
                map(lambda e: e.model_dump(), spectra_metadata))
            spectra_df = spectra_df[spectra_df['annotated'] == True]
        else:
            spectra_df = pd.DataFrame(columns=['file_id', 'annotated', 'tags'])

        spectra_df = filter_by_tags(
            spectra_df, 'tags', key='Dialog_Model_Spectra_Filter')

        selection = st.dataframe(spectra_df, on_select='rerun',
                                 selection_mode='multi-row', use_container_width=True)
        selected_rows = list(selection['selection']['rows'])
        selected_spectra_ids = spectra_df.iloc[selected_rows]['file_id'].tolist(
        )

    with t3:
        st.caption("Select a config file to train the model.")
        config_metadata = get_database_session(
        ).config_manager.get_all_metadata(get_current_user()['id'])
        config_df = pd.DataFrame(map(lambda e: e.model_dump(
        ), config_metadata)) if config_metadata else pd.DataFrame(columns=['file_id', 'tags'])
        config_df = filter_by_tags(
            config_df, 'tags', key='Dialog_Model_Config_Filter')
        selection = st.dataframe(config_df, on_select='rerun',
                                 selection_mode='single-row', use_container_width=True)
        selected_row = selection['selection']['rows'][0] if len(
            selection['selection']['rows']) > 0 else None
        selected_config = config_df.iloc[selected_row]['file_id'] if selected_row is not None else None

    if not selected_spectra_ids:
        st.warning("No annotated spectra selected.")

    if not selected_config:
        st.warning("No config selected.")

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True,
                 disabled=len(selected_spectra_ids) == 0 or selected_config is None):
        # db.train(self, spectra_ids: list[str], config_id: Optional[str], model_metadata: ModelFileMetadata) -> str:
        get_database_session().train(selected_spectra_ids, selected_config, ModelFileMetadata(
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

        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        st.rerun()


@st.dialog("Add Model")
def add_option():
    uploaded_file = st.file_uploader("Upload Model", type=['ckpt'])

    if uploaded_file:
        st.subheader("Model Metadata", divider='blue')
        base_file_name, file_extension = os.path.splitext(uploaded_file.name)
        file_extension = file_extension.lstrip(".")
        c1, c2 = st.columns([7, 2])
        file_name = c1.text_input(
            "File Name", value=base_file_name, disabled=False)
        file_type = c2.text_input(
            "File Type", value=file_extension, disabled=True)

        description = st.text_area("Description")
        date_input = st.date_input("Date", value=date.today())
        tags = sorted(list(set([tag.strip() for tag in st.text_input(
            "Tags (comma-separated)").split(",") if tag])))

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

        get_database_session().models_manager.add_file(
            tmp_path, metadata, True, get_current_user()['id'])
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        st.rerun()


@st.dialog("Edit Model Metadata")
def edit_option(entry: ModelFileMetadata):
    st.subheader("Model Metadata", divider='blue')

    c1, c2 = st.columns([7, 2])
    entry.file_name = c1.text_input(
        "File Name", value=entry.file_name, disabled=False)
    entry.file_type = c2.text_input(
        "File Type", value=entry.file_type, disabled=True)

    entry.description = st.text_area("Description", value=entry.description)
    entry.date = st.date_input("Date", value=entry.date)
    entry.tags = sorted(list(
        set([tag.strip() for tag in st.text_input("Tags (comma-separated)", value=",".join(entry.tags)).split(",") if
             tag])))

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True):
        get_database_session().models_manager.update_file_metadata(entry)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        st.rerun()


def run():
    # Set up the Streamlit page configuration
    st.set_page_config(page_title=f"Model", layout="wide")
    c1, c2 = st.columns([5, 3])
    c1.title(f"Model")
    st.caption('Default models can be downloaded from the home page. Otherwise models can be uploaded or trained. '
               'See available models at https://github.com/Noble-Lab/casanovo/releases.')

    with st.sidebar:
        display_login_ui()

    user_data = get_current_user()

    if user_data is None:
        st.warning("Please log in to access this page.")
        return

    db = get_database_session()
    manager = db.models_manager

    # Get all file metadata entries
    entries = manager.get_all_metadata(user_data['id'])
    entries = map(lambda e: e.dict(), entries)
    df = pd.DataFrame(entries)

    if df.empty:
        st.write("No entries found.")
        df = pd.DataFrame(columns=[
                          "file_id", "file_name", "description", "date", "tags", "source", "status", "config"])

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
    df.rename(columns=rename_map, inplace=True)

    if 'Config' not in df.columns:
        df['Config'] = None
    else:
        df['Config'] = df['Config'].apply(get_config_filename)

    # Customize the dataframe for display
    df['Date'] = pd.to_datetime(df['Date'])

    df = filter_by_tags(df)

    # Display the editable dataframe
    selection = st.dataframe(df,
                             hide_index=True,
                             column_order=["Name", "Description", "Date",
                                           "Tags", "Source", "Status", "Config"],
                             column_config={
                                 "Name": st.column_config.TextColumn(disabled=True, width='medium'),
                                 "Description": st.column_config.TextColumn(disabled=True, width='medium'),
                                 "Date": st.column_config.DateColumn(disabled=True, width='small'),
                                 "Tags": st.column_config.ListColumn(width='small'),
                                 "Source": st.column_config.TextColumn(disabled=True, width='small'),
                                 "Status": st.column_config.TextColumn(disabled=True, width='small'),
                                 "Config": st.column_config.TextColumn(disabled=True, width='small')
                             },
                             use_container_width=True,
                             selection_mode='multi-row',
                             on_select='rerun')

    selected_rows = selection['selection']['rows']
    selected_ids = df.iloc[selected_rows]["ID"].tolist() if selected_rows else [
    ]

    c1, c2, c3, c4, c5, c6, c7, c8 = c2.columns(8)

    if c1.button("🗑️", use_container_width=True, disabled=len(selected_ids) == 0, help="Delete selected entries"):
        delete_option(selected_ids, manager)
    if c2.button("📥", use_container_width=True, disabled=len(selected_ids) == 0, help="Download selected entries"):
        download_option(selected_ids, manager)
    if c3.button("🏷️", use_container_width=True, disabled=len(selected_ids) == 0, help="Tag selected entries"):
        tag_option(selected_ids, manager)
    if c4.button("✏️", use_container_width=True, disabled=len(selected_ids) != 1, help="Edit selected entry"):
        edit_option(manager.get_file_metadata(selected_ids[0]))
    if c5.button("👁️", use_container_width=True, disabled=len(selected_ids) != 1, help="View selected entry"):
        view_option(selected_ids[0], mode='log', manager=manager)
    if c6.button("➕", use_container_width=True, help="Train new model"):
        train_option()
    if c7.button("📤", use_container_width=True, help="Upload model"):
        add_option()
    if c8.button("🔄", use_container_width=True, help="Refresh"):
        st.rerun()
