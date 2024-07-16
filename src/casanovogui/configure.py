import os
import tempfile
import uuid
from datetime import date

import pandas as pd

import streamlit as st
import yaml

from dialogs import download_option, tag_option, delete_option, view_option
from simple_db import ConfigFileMetadata
from utils import get_database_session, filter_by_tags


@st.experimental_dialog("Create config", width="large")
def create_entry():
    tabs = st.tabs(['Metadata', 'Inference', 'Training', 'Spectrum Processing',
                    'Model Architecture', 'Training/Inference', 'Residue Vocabulary'])

    with tabs[0]:
        c1, c2 = st.columns([7, 2])
        file_name = c1.text_input("File Name", value="config", disabled=False)
        file_type = c2.text_input("File Type", value="yaml", disabled=True)

        description = st.text_area("Description")
        date_input = st.date_input("Date", value=date.today())
        tags = [tag for tag in st.text_input("Tags (comma-separated)").split(",") if tag]

    with tabs[1]:
        c1, c2, c3 = st.columns(3)
        precursor_mass_tol = c1.number_input("Precursor Mass Tolerance (ppm)", value=50)
        isotope_error_min_range = c2.number_input("Isotope Error Min Range", value=0)
        isotope_error_max_range = c3.number_input("Isotope Error Max Range", value=1)
        c1, c2, c3 = st.columns(3)
        min_peptide_len = c1.number_input("Minimum Peptide Length", value=6)
        predict_batch_size = c2.number_input("Predict Batch Size", value=1024)
        n_beams = c3.number_input("Number of Beams", value=1)
        c1, c2, c3 = st.columns(3)
        top_match = c1.number_input("Number of PSMs for Each Spectrum", value=1)
        accelerator = c2.selectbox("Hardware Accelerator", ["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"], index=6)
        devices = c3.text_input("Devices", value="")

    with tabs[2]:
        c1, c2, c3 = st.columns(3)
        random_seed = c1.number_input("Random Seed", value=454)
        n_log = c2.number_input("Logging Frequency", value=1)
        tb_summarywriter = c3.text_input("Tensorboard Directory")
        c1, c2, c3 = st.columns(3)
        save_top_k = c1.number_input("Save Top K Model Checkpoints", value=5)
        model_save_folder_path = c2.text_input("Model Save Folder Path", value="")
        val_check_interval = c3.number_input("Validation Check Interval", value=50000)

    with tabs[3]:
        c1, c2, c3 = st.columns(3)
        n_peaks = c1.number_input("Number of Most Intense Peaks to Retain", value=150)
        min_mz = c2.number_input("Minimum Peak m/z", value=50.0)
        max_mz = c3.number_input("Maximum Peak m/z", value=2500.0)
        c1, c2, c3 = st.columns(3)
        min_intensity = c1.number_input("Minimum Peak Intensity", value=0.01)
        remove_precursor_tol = c2.number_input("Remove Precursor Tolerance", value=2.0)
        max_charge = c3.number_input("Maximum Precursor Charge", value=10)

    with tabs[4]:
        c1, c2, c3 = st.columns(3)
        dim_model = c1.number_input("Dimensionality of Latent Representations", value=512)
        n_head = c2.number_input("Number of Attention Heads", value=8)
        dim_feedforward = c3.number_input("Dimensionality of Fully Connected Layers", value=1024)
        c1, c2, c3 = st.columns(3)
        n_layers = c1.number_input("Number of Transformer Layers", value=9)
        dropout = c2.number_input("Dropout Rate", value=0.0)
        dim_intensity = c3.text_input("Dimensionality for Encoding Peak Intensity", value="")
        c1, c2, c3 = st.columns(3)
        max_length = c1.number_input("Max Decoded Peptide Length", value=100)
        warmup_iters = c2.number_input("Warmup Iterations", value=100000)
        cosine_schedule_period_iters = c3.number_input("Cosine Schedule Period Iterations", value=600000)
        c1, c2, c3 = st.columns(3)
        learning_rate = c1.number_input("Learning Rate", value=5e-4)
        weight_decay = c2.number_input("Weight Decay", value=1e-5)
        train_label_smoothing = c3.number_input("Train Label Smoothing", value=0.01)

    with tabs[5]:
        c1, c2, c3 = st.columns(3)
        train_batch_size = c1.number_input("Training Batch Size", value=32)
        max_epochs = c2.number_input("Max Training Epochs", value=30)
        num_sanity_val_steps = c3.number_input("Number of Sanity Validation Steps", value=0)
        calculate_precision = st.checkbox("Calculate Precision During Training", value=False)

    with tabs[6]:
        residue_df = pd.DataFrame({
            "Residue": ["G", "A", "S", "P", "V", "T", "C+57.021", "L", "I", "N", "D", "Q", "K", "E", "M", "H", "F", "R",
                        "Y", "W", "M+15.995", "N+0.984", "Q+0.984", "+42.011", "+43.006", "-17.027", "+43.006-17.027"],
            "Mass": [57.021464, 71.037114, 87.032028, 97.052764, 99.068414, 101.047670, 160.030649, 113.084064,
                     113.084064,
                     114.042927, 115.026943, 128.058578, 128.094963, 129.042593, 131.040485, 137.058912, 147.068414,
                     156.101111, 163.063329, 186.079313, 147.035400, 115.026943, 129.042594, 42.010565, 43.005814,
                     -17.026549, 25.980265]
        })

        residue_df = st.data_editor(residue_df, num_rows='dynamic', use_container_width=True,
                                    column_config={"Mass": st.column_config.NumberColumn(required=True),
                                                   "Residue": st.column_config.TextColumn(required=True)})

        residue_dict = dict(zip(residue_df["Residue"], residue_df["Mass"]))

    config = {
        "precursor_mass_tol": precursor_mass_tol,
        "isotope_error_range": [isotope_error_min_range, isotope_error_max_range],
        "min_peptide_len": min_peptide_len,
        "predict_batch_size": predict_batch_size,
        "n_beams": n_beams,
        "top_match": top_match,
        "accelerator": accelerator,
        "devices": devices if devices else None,
        "random_seed": random_seed,
        "n_log": n_log,
        "tb_summarywriter": tb_summarywriter if tb_summarywriter else None,
        "save_top_k": save_top_k,
        "model_save_folder_path": model_save_folder_path,
        "val_check_interval": val_check_interval,
        "n_peaks": n_peaks,
        "min_mz": min_mz,
        "max_mz": max_mz,
        "min_intensity": min_intensity,
        "remove_precursor_tol": remove_precursor_tol,
        "max_charge": max_charge,
        "dim_model": dim_model,
        "n_head": n_head,
        "dim_feedforward": dim_feedforward,
        "n_layers": n_layers,
        "dropout": dropout,
        "dim_intensity": dim_intensity if dim_intensity else None,
        "max_length": max_length,
        "warmup_iters": warmup_iters,
        "cosine_schedule_period_iters": cosine_schedule_period_iters,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "train_label_smoothing": train_label_smoothing,
        "train_batch_size": train_batch_size,
        "max_epochs": max_epochs,
        "num_sanity_val_steps": num_sanity_val_steps,
        "calculate_precision": calculate_precision,
        "residues": residue_dict
    }

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True):
        yaml_config = yaml.dump(config, default_flow_style=False, sort_keys=False)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(yaml_config.encode())
            tmp_path = tmp.name

        metadata = ConfigFileMetadata(
            file_id=str(uuid.uuid4()),
            file_name=file_name,
            description=description,
            file_type=file_type,
            date=date_input,
            tags=tags
        )

        get_database_session().config_manager.add_file(tmp_path, metadata)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        st.rerun()


@st.experimental_dialog("Add Config")
def add_option():
    uploaded_file = st.file_uploader("Upload Config", type='yaml')

    if uploaded_file:
        st.subheader("Config Metadata", divider='blue')
        base_file_name, file_extension = os.path.splitext(uploaded_file.name)
        file_extension = file_extension.lstrip(".")
        c1, c2 = st.columns([7, 2])
        file_name = c1.text_input("File Name", value=base_file_name, disabled=False)
        file_type = c2.text_input("File Type", value=file_extension, disabled=True)

        description = st.text_area("Description")
        date_input = st.date_input("Date", value=date.today())
        tags = sorted(list(set([tag.strip() for tag in st.text_input("Tags (comma-separated)").split(",") if tag])))

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True, disabled=not uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        metadata = ConfigFileMetadata(
            file_id=str(uuid.uuid4()),
            file_name=file_name,
            description=description,
            file_type=file_type,
            date=date_input,
            tags=tags
        )

        get_database_session().config_manager.add_file(tmp_path, metadata)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        st.rerun()


@st.experimental_dialog("Edit Config Metadata")
def edit_option(entry: ConfigFileMetadata):
    st.subheader("Config Metadata", divider='blue')

    c1, c2 = st.columns([7, 2])
    entry.file_name = c1.text_input("File Name", value=entry.file_name, disabled=False)
    entry.file_type = c2.text_input("File Type", value=entry.file_type, disabled=True)

    entry.description = st.text_area("Description", value=entry.description)
    entry.date = st.date_input("Date", value=entry.date)
    entry.tags = sorted(list(set([tag.strip() for tag in st.text_input("Tags (comma-separated)", value=",".join(entry.tags)).split(",") if tag])))

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True):
        get_database_session().config_manager.update_file_metadata(entry)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        st.rerun()


def run():
    # Set up the Streamlit page configuration
    st.set_page_config(page_title=f"Config", layout="wide")
    c1, c2 = st.columns([5, 3])
    c1.title(f"Config")

    db = get_database_session()
    manager = db.config_manager

    # Streamlit app layout

    # Get all file metadata entries
    entries = manager.get_all_metadata()
    entries = map(lambda e: e.dict(), entries)
    df = pd.DataFrame(entries)

    if df.empty:
        st.write("No entries found.")
        df = pd.DataFrame(columns=["file_id", "file_name", "description", "date", "tags"])

    rename_map = {
        "file_id": "ID",
        "file_name": "Name",
        "description": "Description",
        "date": "Date",
        "tags": "Tags"
    }

    # Customize the dataframe for display
    df.rename(columns=rename_map, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    df = filter_by_tags(df)

    selection = st.dataframe(df,
                             hide_index=True,
                             column_order=["Name", "Description", "Date", "Tags"],
                             column_config={
                                 "Name": st.column_config.TextColumn(disabled=True, width='medium'),
                                 "Description": st.column_config.TextColumn(disabled=True, width='medium'),
                                 "Date": st.column_config.DateColumn(disabled=True, width='small'),
                                 "Tags": st.column_config.ListColumn(width='small')
                             },
                             selection_mode='multi-row',
                             on_select='rerun',
                             use_container_width=True)

    selected_rows = selection['selection']['rows']
    selected_ids = df.iloc[selected_rows]["ID"].tolist() if selected_rows else []

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
        view_option(selected_ids[0], mode='file', manager=manager)
    if c6.button("➕", use_container_width=True, help="Create new config"):
        create_entry()
    if c7.button("📤", use_container_width=True, help="Upload config"):
        add_option()
    if c8.button("🔄", use_container_width=True, help="Refresh"):
        st.rerun()
