import os
import tempfile
import uuid
from datetime import date

import pandas as pd

import streamlit as st
import yaml

from simple_db import ConfigFileMetadata
from utils import refresh_de_key, get_database_session, filter_by_tags


PAGE_KEY = 'CONFIG'
PAGE_DE_KEY = f"{PAGE_KEY}_de_key"
SUPPORTED_FILES = ['.yaml']

# Set up the Streamlit page configuration
st.set_page_config(page_title=f"Config", layout="wide")
st.title(f"Config")
st.caption('A Default config file is available on the home page. Otherwise use the create config button to '
           'create a new config file.')

if PAGE_DE_KEY not in st.session_state:
    refresh_de_key(PAGE_DE_KEY)

db = get_database_session()
manager = db.config_manager

# Streamlit app layout

# Get all file metadata entries
entries = manager.get_all_metadata()
entries = map(lambda e: e.dict(), entries)
df = pd.DataFrame(entries)


@st.experimental_dialog("Create config", width="large")
def create_entry():

    st.header("Config Metadata", divider='blue')
    c1, c2 = st.columns([7, 2])
    file_name = c1.text_input("File Name", value="config", disabled=False)
    file_type = c2.text_input("File Type", value="yaml", disabled=True)

    description = st.text_area("Description")
    date_input = st.date_input("Date", value=date.today())
    tags = [tag for tag in st.text_input("Tags (comma-separated)").split(",") if tag]

    st.header("Inference/Training Parameters", divider='blue')
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

    st.header("Training Parameters", divider='blue')
    c1, c2, c3 = st.columns(3)
    random_seed = c1.number_input("Random Seed", value=454)
    n_log = c2.number_input("Logging Frequency", value=1)
    tb_summarywriter = c3.text_input("Tensorboard Directory")
    c1, c2, c3 = st.columns(3)
    save_top_k = c1.number_input("Save Top K Model Checkpoints", value=5)
    model_save_folder_path = c2.text_input("Model Save Folder Path", value="")
    val_check_interval = c3.number_input("Validation Check Interval", value=50000)

    st.header("Spectrum Processing Options", divider='blue')
    c1, c2, c3 = st.columns(3)
    n_peaks = c1.number_input("Number of Most Intense Peaks to Retain", value=150)
    min_mz = c2.number_input("Minimum Peak m/z", value=50.0)
    max_mz = c3.number_input("Maximum Peak m/z", value=2500.0)
    c1, c2, c3 = st.columns(3)
    min_intensity = c1.number_input("Minimum Peak Intensity", value=0.01)
    remove_precursor_tol = c2.number_input("Remove Precursor Tolerance", value=2.0)
    max_charge = c3.number_input("Maximum Precursor Charge", value=10)

    st.header("Model Architecture Options", divider='blue')
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

    st.header("Training/Inference Options", divider='blue')
    c1, c2, c3 = st.columns(3)
    train_batch_size = c1.number_input("Training Batch Size", value=32)
    max_epochs = c2.number_input("Max Training Epochs", value=30)
    num_sanity_val_steps = c3.number_input("Number of Sanity Validation Steps", value=0)
    calculate_precision = st.checkbox("Calculate Precision During Training", value=False)

    st.header("Amino Acid and Modification Vocabulary", divider='blue')

    residue_df = pd.DataFrame({
        "Residue": ["G", "A", "S", "P", "V", "T", "C+57.021", "L", "I", "N", "D", "Q", "K", "E", "M", "H", "F", "R",
                    "Y", "W", "M+15.995", "N+0.984", "Q+0.984", "+42.011", "+43.006", "-17.027", "+43.006-17.027"],
        "Mass": [57.021464, 71.037114, 87.032028, 97.052764, 99.068414, 101.047670, 160.030649, 113.084064, 113.084064,
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

        manager.add_file(tmp_path, metadata)
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()


@st.experimental_dialog("Add Config")
def add_option():
    uploaded_file = st.file_uploader("Upload Config", type=SUPPORTED_FILES)

    if uploaded_file:
        st.subheader("Config Metadata", divider='blue')
        base_file_name, file_extension = os.path.splitext(uploaded_file.name)
        file_extension = file_extension.lstrip(".")
        c1, c2 = st.columns([7, 2])
        file_name = c1.text_input("File Name", value=base_file_name, disabled=False)
        file_type = c2.text_input("File Type", value=file_extension, disabled=True)

        description = st.text_area("Description")
        date_input = st.date_input("Date", value=date.today())
        tags = [tag for tag in st.text_input("Tags (comma-separated)").split(",") if tag]

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

        manager.add_file(tmp_path, metadata)
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()

@st.experimental_dialog("Edit Config Metadata")
def edit_option(entry: ConfigFileMetadata):

    st.subheader("Config Metadata", divider='blue')

    c1, c2 = st.columns([7, 2])
    entry.file_name = c1.text_input("File Name", value=entry.file_name, disabled=False)
    entry.file_type = c2.text_input("File Type", value=entry.file_type, disabled=True)

    entry.description = st.text_area("Description", value=entry.description)
    entry.date = st.date_input("Date", value=entry.date)
    entry.tags = [tag for tag in st.text_input("Tags (comma-separated)", value=",".join(entry.tags)).split(",") if tag]

    c1, c2 = st.columns([1, 1])
    if c1.button("Submit", type='primary', use_container_width=True):
        manager.update_file_metadata(entry)
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()

    if c2.button("Cancel", use_container_width=True):
        refresh_de_key(PAGE_DE_KEY)
        st.rerun()



@st.experimental_dialog("Delete Config")
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


@st.experimental_dialog("Download Config")
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
if c1.button("Add Config", use_container_width=True, type='primary'):
    add_option()
if c2.button("Create Config", use_container_width=True, type='primary'):
    create_entry()
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
    "tags": "Tags"
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
                           column_order=["✏️", "🗑️", "📥", "👁️", "Name", "Description", "Date", "Tags"],
                           column_config={
                               "✏️": st.column_config.CheckboxColumn(disabled=False, width='small'),
                               "🗑️": st.column_config.CheckboxColumn(disabled=False, width='small'),
                               "📥": st.column_config.CheckboxColumn(disabled=False, width='small'),
                                 "👁️": st.column_config.CheckboxColumn(disabled=False, width='small'),
                               "Name": st.column_config.TextColumn(disabled=True, width='medium'),
                               "Description": st.column_config.TextColumn(disabled=True, width='medium'),
                               "Date": st.column_config.DateColumn(disabled=True, width='small'),
                               "Tags": st.column_config.ListColumn(width='small')
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
        file_path = manager.retrieve_file_path(entry.file_id)

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

    st.subheader(f"Name: {entry.file_name}.yaml", divider='blue')

    with open(file_path, "rb") as file:
        st.code(file.read().decode(), language='yaml')

    # remove the viewed_file key
    del st.session_state['viewed_file']


