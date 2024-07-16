import os
import tempfile
import zipfile

import pandas as pd

from simple_db import FileManager
import streamlit as st



@st.experimental_dialog("Download Entries")
def download_option(file_ids: list[str], manager: FileManager):
    f1, f2 = st.columns([3, 1])
    c1, c2 = st.columns([1, 1])

    if len(file_ids) == 1:
        file_id = file_ids[0]
        metadata = manager.get_file_metadata(file_id)
        file_path = manager.retrieve_file_path(file_id)

        file_basename = f1.text_input("File Name", value=metadata.file_name)
        file_extension = f2.text_input("File Type", value=metadata.file_type, disabled=True)
        file_name = f"{file_basename}.{file_extension}"
        with open(file_path, "rb") as file:
            btn = c1.download_button(
                label="Download",
                data=file,
                file_name=file_name,
                mime='application/octet-stream',
                use_container_width=True,
                type='primary'
            )
    else:
        file_paths = [manager.retrieve_file_path(file_id) for file_id in file_ids]
        file_names = [manager.get_file_metadata(file_id).file_name for file_id in file_ids]
        file_extensions = [manager.get_file_metadata(file_id).file_type for file_id in file_ids]

        file_basename = f1.text_input("Zip File Name", value="configs")
        file_extension = f2.text_input("File Type", value="zip", disabled=True)
        file_name = f"{file_basename}.{file_extension}"

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            with zipfile.ZipFile(tmp_file, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for fname, fext, file_path in zip(file_names, file_extensions, file_paths):
                    # Add the file to the zip archive
                    zip_file_name = os.path.basename(file_path)
                    zip_file.write(file_path, arcname=fname + '.' + fext)

            tmp_file.seek(0)
            zip_data = tmp_file.read()

        # Create the download button
        btn = c1.download_button(
            label="Download",
            data=zip_data,
            file_name=file_name,
            mime='application/zip',
            use_container_width=True,
            type='primary'
        )

    if btn:
        st.rerun()
    if c2.button("Cancel", use_container_width=True):
        st.rerun()


@st.experimental_dialog("Tag Entries")
def tag_option(file_ids: list[str], manager: FileManager):
    # add tags to the selected files
    tags = set([tag.strip() for tag in st.text_input("Tags (comma-separated)").split(',') if tag.strip()])

    c1, c2 = st.columns([1, 1])

    if c1.button("Submit", use_container_width=True, type='primary'):

        # add tags to the selected files
        for file_id in file_ids:
            metadata = manager.get_file_metadata(file_id)
            previous_tags = set(metadata.tags)
            previous_tags.update(tags)
            metadata.tags = sorted(list(previous_tags))
            manager.update_file_metadata(metadata)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        st.rerun()


@st.experimental_dialog("Delete Config")
def delete_option(file_ids: list[str], manager: FileManager):
    # Get all file metadata entries
    entries = [manager.get_file_metadata(file_id) for file_id in file_ids]
    entries = map(lambda e: e.dict(), entries)
    df = pd.DataFrame(entries)

    rename_map = {
        "file_id": "ID",
        "file_name": "Name",
        "description": "Description",
        "date": "Date",
        "tags": "Tags"
    }

    # Customize the dataframe for display
    df.rename(columns=rename_map, inplace=True)

    st.write("Are you sure you want to delete the following entries?")
    st.dataframe(df, hide_index=True, column_order=["Name", "Description", "Date", "Tags"], use_container_width=True)

    c1, c2 = st.columns([1, 1])
    if c1.button("Delete", use_container_width=True, key='delete_option_dialog delete'):
        for file_id in file_ids:
            manager.delete_file(file_id)
        st.rerun()
    if c2.button("Cancel", type='primary', use_container_width=True, key='delete_option_dialog cancel'):
        st.rerun()


@st.experimental_dialog("View Config", width="large")
def view_option(file_id: str, manager: FileManager, mode: str = 'file'):

    if mode not in ['file', 'log', 'both']:
        raise ValueError(f"Invalid mode: {mode}")

    if mode == 'both':
        mode = st.selectbox("Select Mode", ['file', 'log'])

    if mode == 'file':
        entry = manager.get_file_metadata(file_id)
        file_path = manager.retrieve_file_path(file_id)

        st.subheader(f"Name: {entry.file_name}", divider='blue')

        try:
            with open(file_path, "rb") as file:
                st.code(file.read(10000).decode(), language='txt')
        except FileNotFoundError:
            st.error("File not found")

    elif mode == 'log':
        entry = manager.get_file_metadata(file_id)
        file_path = manager.retrieve_file_path(file_id)

        # replace extension with .log
        file_extension = entry.file_type
        file_path = file_path.replace(f'.{file_extension}', '.log')

        st.subheader(f"Name: {entry.file_name}", divider='blue')

        try:
            with open(file_path, "rb") as file:
                st.code(file.read().decode(), language='txt')
        except FileNotFoundError:
            st.error("File not found")