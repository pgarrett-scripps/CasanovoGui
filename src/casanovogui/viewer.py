import os.path
from typing import Optional

import pandas as pd
import streamlit as st
import plotly.express as px
from matplotlib import pyplot as plt
from pyteomics import mgf
import peptacular as pt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils import (generate_annonated_spectra_plotly, get_database_session, filter_by_tags, get_model_filename,
                   get_spectra_filename)

def run():

    st.set_page_config(page_title="Results Viewer", layout="wide")
    st.title('Results Viewer')

    st.subheader('Selected Search', divider=True)

    if 'selected_search_id' not in st.session_state:
        st.session_state.selected_search_id = None

    selected_search_id = st.session_state.selected_search_id


    @st.experimental_dialog(title="Select Search", width='large')
    def update_current_search():
        # Get all file metadata entries
        db = get_database_session()
        entries = db.searches_manager.get_all_metadata()
        df = pd.DataFrame(map(lambda e: e.dict(), entries))

        rename_map = {
            "file_id": "ID",
            "file_name": "Name",
            "description": "Description",
            "date": "Date",
            "tags": "Tags",
            "model": "Model",
            "spectra": "Spectra",
            "status": "Status"
        }

        df = df.rename(columns=rename_map)

        if 'Model' not in df.columns:
            df['Model'] = None

        if 'Spectra' not in df.columns:
            df['Spectra'] = None

        df['Model'] = df['Model'].apply(get_model_filename)
        df['Spectra'] = df['Spectra'].apply(get_spectra_filename)

        # keep only Comleted searches
        df = df[df['Status'] == 'completed']

        # Customize the dataframe for display
        df['Date'] = pd.to_datetime(df['Date'])

        df = filter_by_tags(df, key='Main_Page_Filter')
        selection = st.dataframe(df,
                                 selection_mode='single-row',
                                 on_select='rerun',
                                 hide_index=True,
                                 use_container_width=True,
                                 column_order=['Name', 'Description', 'Date', 'Tags', 'Model ID', 'Spectra ID', 'Status'],
                                 column_config={
                                     "Name": st.column_config.TextColumn(disabled=True, width='medium'),
                                     "Description": st.column_config.TextColumn(disabled=True, width='medium'),
                                     "Date": st.column_config.DateColumn(disabled=True, width='small'),
                                     "Tags": st.column_config.ListColumn(width='small'),
                                     "Model": st.column_config.TextColumn(disabled=True, width='small'),
                                     "Spectra": st.column_config.TextColumn(disabled=True, width='small'),
                                 },
                                    )

        selected_row = selection['selection']['rows'][0] if selection['selection']['rows'] else None

        selected_search_id = df.iloc[selected_row]['ID'] if selected_row is not None else None

        c1, c2 = st.columns([1, 1])
        if c1.button('Sumbit', type='primary', use_container_width=True):
            st.session_state.selected_search_id = selected_search_id
            st.rerun()

        elif c2.button('Cancel', use_container_width=True):
            st.stop()


    if st.button('Update', type='primary', use_container_width=True):
        update_current_search()

    if selected_search_id is not None:
        db = get_database_session()
        entry = db.searches_manager.get_file_metadata(selected_search_id)
        df = pd.DataFrame(map(lambda e: e.dict(), [entry]))

        rename_map = {
            "file_id": "ID",
            "file_name": "Name",
            "description": "Description",
            "date": "Date",
            "tags": "Tags",
            "model": "Model",
            "spectra": "Spectra",
            "status": "Status"
        }

        df = df.rename(columns=rename_map)

        if 'Model' not in df.columns:
            df['Model'] = None

        if 'Spectra' not in df.columns:
            df['Spectra'] = None

        df['Model'] = df['Model'].apply(get_model_filename)
        df['Spectra'] = df['Spectra'].apply(get_spectra_filename)

        st.dataframe(df,
                     hide_index=True,
                     use_container_width=True,
                     column_order=['Name', 'Description', 'Date', 'Tags', 'Model ID', 'Spectra ID', 'Status'],
                     column_config={
                                   "Name": st.column_config.TextColumn(disabled=True, width='medium'),
                                   "Description": st.column_config.TextColumn(disabled=True, width='medium'),
                                   "Date": st.column_config.DateColumn(disabled=True, width='small'),
                                   "Tags": st.column_config.ListColumn(width='small'),
                                    "Model": st.column_config.TextColumn(disabled=True, width='small'),
                                    "Spectra": st.column_config.TextColumn(disabled=True, width='small'),
                               },
                     )



    db = get_database_session()
    manager = db.searches_manager

    class MGF_Index(mgf.MGF):
        def get_spectrum_by_index(self, index: int) -> dict:
            s_line_index = 0
            for line in self._source:
                sline = line.strip()
                if sline[:5] == 'TITLE':

                    if s_line_index == index:
                        return self._read_spectrum()

                    s_line_index += 1

        def __getitem__(self, index: int) -> dict:
            return self.get_spectrum_by_index(index)


    def read_mgf_index_file(source) -> MGF_Index:
        return MGF_Index(source, use_header=True, read_charges=False)


    def get_spectrum_by_index(mgf_file, index) -> dict:
        with open(mgf_file) as f:
            mgf_reader = read_mgf_index_file(f)
            return mgf_reader[index]


    def casanovo_to_df(file):
        header, data, intro = None, [], []
        for line in file:
            if line == '\n':
                continue
            elif line.startswith('PSH'):
                header = line.strip().split('\t')
            elif line.startswith('PSM'):
                data.append(line.strip().split('\t'))
            else:
                intro.append(line)
        df = pd.DataFrame(data, columns=header)
        return df, intro


    @st.cache_data()
    def get_search_df(search_id):
        search_path = db.searches_manager.retrieve_file_path(search_id)

        if search_path is None or not os.path.exists(search_path):
            st.error('Search file not found')
            st.error('Search could have failed or is still bring processed. Please try again later.')
            st.stop()

        with open(search_path) as f:
            search_df, intro = casanovo_to_df(f)

        # spectra_ref: ms_run[1]:index=118 (fix?)
        search_df['ref_index'] = search_df['spectra_ref'].str.extract(r'index=(\d+)').astype(int)

        search_df['proforma_sequence'] = search_df['sequence'].apply(pt.convert_casanovo_sequence)
        search_df['calc_mass_to_charge'] = search_df['calc_mass_to_charge'].astype(float)
        search_df['exp_mass_to_charge'] = search_df['exp_mass_to_charge'].astype(float)

        search_df['ppm_error'] = search_df.apply(lambda x: pt.ppm_error(x['calc_mass_to_charge'], x['exp_mass_to_charge']),
                                                 axis=1)
        search_df['dalton_error'] = search_df.apply(lambda x: pt.dalton_error(x['calc_mass_to_charge'], x['exp_mass_to_charge']), axis=1)

        rename_map = {
            "proforma_sequence": "Sequence",
            "charge": "Charge",
            "search_engine_score[1]": "Score",
            "exp_mass_to_charge": "Experimental m/z",
            "calc_mass_to_charge": "Theoretical m/z",
            "dalton_error": "Dalton Error",
            "ppm_error": "PPM Error",
            "retention_time": "Retention Time",
        }
        search_df = search_df.rename(columns=rename_map)

        return search_df


    # get the log file and show it
    if selected_search_id is None:
        st.warning('No search selected')
        st.stop()


    try:
        search_metadata = manager.get_file_metadata(selected_search_id)
        spectra_id = search_metadata.spectra
    except FileNotFoundError as e:
        st.error('Search Metadata not found')
        st.stop()

    try:
        spectra_metadata = db.spectra_files_manager.get_file_metadata(spectra_id)
        spectra_path = db.spectra_files_manager.retrieve_file_path(spectra_id)
    except FileNotFoundError as e:
        st.error('Spectra Metadata not found')
        st.error('Spectra file could have been deleted or moved. Please re-upload the file.')
        st.stop()

    if spectra_metadata.file_type == 'mzML':
        st.error('mzML files are not supported yet')
    elif spectra_metadata.file_type == 'mgf':
        pass
    else:
        raise ValueError(f'Unsupported file type: {spectra_metadata.file_type}')

    search_id = search_metadata.file_id
    search_path = db.searches_manager.retrieve_file_path(search_id)

    search_df = get_search_df(search_id)

    # plotly scatter plot of ppm_error vs score
    st.subheader('Search Results', divider=True)
    st.caption('Select a point to view the spectrum, or use the lass or box tool to select multiple points')


    c1, c2 = st.columns([1, 1])
    min_ppm_error = c1.number_input('Min PPM Error', value=-50)
    max_ppm_error = c2.number_input('Max PPM Error', value=50)
    filter_by_best_peptide = c1.checkbox('Filter by best (peptide, charge) pair', value=True)

    search_df = search_df[(search_df['PPM Error'] > min_ppm_error) & (search_df['PPM Error'] < max_ppm_error)]

    if filter_by_best_peptide:
        search_df = search_df.sort_values('Score', ascending=False).groupby(['Sequence', 'Charge']).head(1)

    st.divider()

    fig = px.scatter(search_df, x='PPM Error', y='Score',
                     hover_data=['Sequence', 'Charge', 'Experimental m/z', 'Theoretical m/z', 'Retention Time',
                                 'Dalton Error',
                                 'PPM Error'])

    selection = st.plotly_chart(fig, on_select='rerun')
    selected_indices = selection['selection']['point_indices'] if selection['selection']['point_indices'] else None

    if selected_indices is not None:
        search_df = search_df.iloc[selected_indices]

    selection = st.dataframe(search_df, selection_mode='single-row', on_select='rerun', hide_index=True,
                             column_order=['Sequence', 'Charge', 'Score', 'Experimental m/z', 'Theoretical m/z',
                                           'Retention Time', 'Dalton Error', 'PPM Error'],
                             use_container_width=True)
    selected_index = selection['selection']['rows'][0] if selection['selection']['rows'] else None

    if selected_index is not None:
        selected_row = search_df.iloc[selected_index]
        selected_index = selected_row['ref_index']
    elif len(search_df) == 1:
        selected_index = search_df.iloc[0]['ref_index']
        selected_row = search_df.iloc[0]

    if selected_index is not None:

        selected_spectrum = get_spectrum_by_index(spectra_path, selected_index)

        peptide = selected_row['Sequence']
        charge = int(float(selected_row['Charge']))
        mz_spectra = selected_spectrum['m/z array']
        intensity_spectra = selected_spectrum['intensity array']

        st.subheader(f'Spectra Viewer: {peptide}/{charge}', divider=True)

        scores = list(map(float, selected_row['opt_ms_run[1]_aa_scores'].split(',')))
        unmod_peptide = pt.strip_mods(peptide)

        # Normalize scores to the range [0, 1]
        norm = plt.Normalize(0, 1)
        colors = plt.cm.viridis(norm(scores))

        colored_html = ''.join(
            [f'<span title="{score}" style="color: rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)}); '
             f'background-color: rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 0.2); padding: 2px; font-size: 24px;">{aa}</span>'
             for aa, score, (r, g, b, _) in zip(unmod_peptide, scores, colors)]
        )

        # Wrap the colored HTML in a centered div
        centered_html = f'<div style="text-align: center;">{colored_html}</div>'


        generate_annonated_spectra_plotly(peptide, charge, mz_spectra, intensity_spectra)

        st.subheader('AA Scores', divider=True)
        # Display the centered colored peptide string using Streamlit
        st.markdown(centered_html, unsafe_allow_html=True)

        score_df = pd.DataFrame({'Amino Acid': list(unmod_peptide), 'Score': scores})
        st.dataframe(score_df, use_container_width=True, hide_index=True)