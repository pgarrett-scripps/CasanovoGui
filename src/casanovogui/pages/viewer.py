import pandas as pd
import streamlit as st
import plotly.express as px
from pyteomics import mgf
import peptacular as pt

from utils import generate_annonated_spectra_plotly, get_database_session

st.set_page_config(page_title="Results Viewer", layout="wide")
st.title('Results Viewer')


selected_search_id = st.session_state.get('viewer_Search_id', None)
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

    with open(search_path) as f:
        search_df, intro = casanovo_to_df(f)

    # spectra_ref: ms_run[1]:index=118 (fix?)
    search_df['ref_index'] = search_df['spectra_ref'].str.extract(r'index=(\d+)').astype(int)

    search_df['proforma_sequence'] = search_df['sequence'].apply(pt.convert_casanovo_sequence)
    search_df['calc_mass_to_charge'] = search_df['calc_mass_to_charge'].astype(float)
    search_df['exp_mass_to_charge'] = search_df['exp_mass_to_charge'].astype(float)

    search_df['ppm_error'] = search_df.apply(lambda x: pt.ppm_error(x['calc_mass_to_charge'], x['exp_mass_to_charge']),
                                             axis=1)
    search_df['dalton_error'] = search_df.apply(lambda x: x['calc_mass_to_charge'] - x['exp_mass_to_charge'], axis=1)

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

search_metadata = manager.get_file_metadata(selected_search_id)
spectra_id = search_metadata.spectra_id
spectra_metadata = db.spectra_files_manager.get_file_metadata(spectra_id)
spectra_path = db.spectra_files_manager.retrieve_file_path(spectra_id)

if spectra_metadata.file_type == 'mzML':
    st.error('mzML files are not supported yet')
elif spectra_metadata.file_type == 'mgf':
    pass
else:
    raise ValueError(f'Unsupported file type: {spectra_metadata.file_type}')

search_id = search_metadata.file_id
search_path = db.searches_manager.retrieve_file_path(search_id)

search_df = get_search_df(search_id)

c1, c2 = st.columns([1, 1])
min_ppm_error = c2.number_input('Min PPM Error', value=-50)
max_ppm_error = c1.number_input('Max PPM Error', value=50)
filter_by_best_peptide = c1.checkbox('Filter by best (peptide, charge) pair', value=True)

search_df = search_df[(search_df['PPM Error'] > min_ppm_error) & (search_df['PPM Error'] < max_ppm_error)]

if filter_by_best_peptide:
    search_df = search_df.sort_values('Score', ascending=False).groupby(['Sequence', 'Charge']).head(1)

# plotly scatter plot of ppm_error vs score
st.subheader('Search Results', divider=True)
st.caption('Select a point to view the spectrum, or use the lass or box tool to select multiple points')


fig = px.scatter(search_df, x='PPM Error', y='Score',
                 hover_data=['Sequence', 'Charge', 'Experimental m/z', 'Theoretical m/z', 'Retention Time', 'Dalton Error',
                             'PPM Error'])

selection = st.plotly_chart(fig, on_select='rerun')
selected_indices = selection['selection']['point_indices'] if selection['selection']['point_indices'] else None

if selected_indices is not None:
    search_df = search_df.iloc[selected_indices]

selection = st.dataframe(search_df, selection_mode='single-row', on_select='rerun', hide_index=True,
                         column_order=['Sequence', 'Charge', 'Score', 'Experimental m/z', 'Theoretical m/z',
                                       'Retention Time','Dalton Error', 'PPM Error'],
                         use_container_width=True)
selected_index = selection['selection']['rows'][0] if selection['selection']['rows'] else None

if selected_index is not None:
    selected_row = search_df.iloc[selected_index]
    selected_index = selected_row['ref_index']
elif len(search_df) == 1:
    selected_index = search_df.iloc[0]['ref_index']
    selected_row = search_df.iloc[0]

if selected_index is not None:

    #selected_index
    selected_spectrum = get_spectrum_by_index(spectra_path, selected_index)
    #selected_spectrum

    peptide = selected_row['Sequence']
    charge = int(float(selected_row['Charge']))
    mz_spectra = selected_spectrum['m/z array']
    intensity_spectra = selected_spectrum['intensity array']

    st.subheader(f'Spectra Viewer: {peptide}/{charge}', divider=True)

    generate_annonated_spectra_plotly(peptide, charge, mz_spectra, intensity_spectra)

