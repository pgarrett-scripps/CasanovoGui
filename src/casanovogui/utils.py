import uuid
from typing import List

import streamlit as st
import peptacular as pt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simple_db import CasanovoDB

DATA_FOLDER = "data"


def refresh_de_key(de_key: str):
    if de_key in st.session_state:
        del st.session_state[de_key]
    st.session_state[de_key] = str(uuid.uuid4())


@st.cache_resource
def get_database_session():
    # Create a database session object that points to the URL.
    db = CasanovoDB(DATA_FOLDER)
    return db


def generate_annonated_spectra_plotly(peptide: str, charge: int, mz_spectra: List[float],
                                      intensity_spectra: List[float]):
    COLOR_DICT = {'b': 'blue', 'y': 'red', 'a': 'green', 'x': 'purple', 'c': 'orange', 'z': 'brown', "": 'grey'}

    c1, c2 = st.columns([7, 3])
    peptide = c1.text_input("Peptide", peptide)
    max_charge = c2.number_input("Max Charge", value=charge, min_value=1, max_value=charge)
    charges = list(range(1, max_charge + 1))

    # Get params
    c1, c2 = st.columns(2)
    ion_types = c1.multiselect("Ion Types", ['a', 'b', 'c', 'x', 'y', 'z'], default=['b', 'y'])

    # losses
    losses = c2.multiselect("Losses", ['H2O', 'NH3'], default=[])
    water_loss = 'H2O' in losses
    ammonia_loss = 'NH3' in losses

    c1, c2, c3, c4 = st.columns(4)
    mass_tolerance = c1.number_input("Mass Tolerance", value=10.0, format="%.4f", step=1.0)
    mass_tolerance_type = c2.selectbox("Mass Tolerance Type", ["ppm", "th"])
    mass_type = c3.selectbox("Mass Type", ['monoisotopic', 'average'])
    peak_assignment = c4.selectbox("Peak Assignment", ["closest", "largest"])

    c1, c2 = st.columns(2)
    min_mz = c1.number_input("Min M/Z", value=min(mz_spectra) - 10, format="%.4f", step=1.0)
    max_mz = c2.number_input("Max M/Z", value=max(mz_spectra) + 10, format="%.4f", step=1.0)

    fragments = pt.fragment(peptide, ion_types, charges, monoisotopic=mass_type == 'monoisotopic')
    # Not available in current version of peptacular
    #water_loss=water_loss, ammonia_loss=ammonia_loss
    fragment_df = pd.DataFrame([f.to_dict() for f in fragments])
    fragment_df['color'] = fragment_df['ion_type'].apply(lambda x: COLOR_DICT.get(x, 'grey'))

    fragment_matches = pt.get_fragment_matches(fragments,
                                               mz_spectra,
                                               intensity_spectra,
                                               mass_tolerance,
                                               mass_tolerance_type,
                                               peak_assignment)
    fragment_matches.sort(key=lambda x: abs(x.error), reverse=True)
    fragment_matches = {fm.mz: fm for fm in fragment_matches}  # keep the best fragment match for each mz

    match_data = []
    data = []
    for mz, i in zip(mz_spectra, intensity_spectra):
        fm = fragment_matches.get(mz, None)
        if fm:
            match_data.append(fm.to_dict())
        else:
            fm = pt.FragmentMatch(fragment=None, mz=mz, intensity=i)
            data.append(fm.to_dict())

    spectra_df = pd.DataFrame(data)
    spectra_df['matched'] = False
    match_df = pd.DataFrame(match_data)
    match_df['matched'] = True
    spectra_df = pd.concat([spectra_df, match_df])

    spectra_df['color'] = spectra_df['ion_type'].apply(lambda x: COLOR_DICT.get(x, 'grey'))

    max_intensity = max(intensity_spectra)
    fragment_df['intensity'] = max_intensity * .10

    spectra_df['color'] = spectra_df['ion_type'].apply(lambda x: COLOR_DICT[x])

    spectra_figs = []
    # Loop through each unique color in the spectra dataframe
    for color in spectra_df['color'].unique():
        tmp_df = spectra_df[spectra_df['color'] == color]

        hover_text = [
            f"Label: {label}<br>M/Z: {mz}<br>Intensity: {intensity}"
            for label, mz, intensity in zip(tmp_df['label'], tmp_df['mz'], tmp_df['intensity'])
        ]

        # Create a Scatter plot for the current color group with error bars
        tmp_fig = go.Scatter(
            x=tmp_df['mz'].values,
            y=tmp_df['intensity'].values,
            mode='markers',
            marker=dict(color=color, size=1),
            text=tmp_df['label'],

            opacity=0.5 if color == 'grey' else 1,

            hoverinfo='text',
            hovertext=hover_text,

            error_y=dict(
                type='data',
                symmetric=False,
                arrayminus=tmp_df['intensity'].values,
                array=[0] * len(tmp_df['intensity'].values),
                width=0,
                color=color
            ),
            name=f'Spectra ({color})'
        )

        # Add the Scatter plot to the figure
        spectra_figs.append(tmp_fig)

    fragment_figs = []
    # Loop through each unique color in the fragment dataframe
    for color in fragment_df['color'].unique():
        tmp_df = fragment_df[fragment_df['color'] == color]

        # Create a Scatter plot for the current color group
        tmp_fig = go.Scatter(
            x=tmp_df['mz'].values,
            y=-tmp_df['intensity'].values,
            mode='markers',
            opacity=0.3,
            marker=dict(color=color, size=1),
            text=tmp_df['label'],
            name=f'Fragments ({color})',
            error_y=dict(
                type='data',
                symmetric=False,
                arrayminus=-tmp_df['intensity'].values,
                array=[0] * len(tmp_df['intensity'].values),
                width=0,
                color=color
            ),
        )

        # Add the Scatter plot to the figure
        fragment_figs.append(tmp_fig)

    # Create the Error scatter plot
    matched_df = spectra_df[spectra_df['matched']]
    error_fig = go.Scatter(
        x=matched_df['mz'],
        y=matched_df['error_ppm'],
        mode='markers+text',
        marker_color=matched_df['color'],
        name='Error',
        text=matched_df['label'],
        textposition='top center',
        textfont=dict(size=13, color=matched_df['color']),
    )

    spectra_scatter = go.Scatter(
        x=matched_df['mz'],
        y=matched_df['intensity'],
        marker_color=matched_df['color'],
        mode='text',
        text=matched_df['label'],
        textposition='top center',
        textfont=dict(size=13, color=matched_df['color']),
    )

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.3, 0.7],  # Adjust the heights to make the error plot smaller
        subplot_titles=["Error", "Spectra and Fragments"],
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    # Add traces to the same subplot for Spectra and Fragments
    for spectra_fig in spectra_figs:
        fig.add_trace(spectra_fig, row=2, col=1)
    for fragment_fig in fragment_figs:
        fig.add_trace(fragment_fig, row=2, col=1)
    fig.add_trace(spectra_scatter, row=2, col=1)

    # Add the Error scatter plot to the second subplot
    fig.add_trace(error_fig, row=1, col=1)

    # Update layout with annotations
    fig.update_layout(
        height=800,
        title_text="Spectra, Fragments, and Error Analysis",
        showlegend=False,
        xaxis=dict(
            range=[min_mz, max_mz]  # Replace min_x_value and max_x_value with your desired values
        )
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    annotation = pt.parse(peptide)
    peptide_length = len(annotation)

    # Separate theoretical and experimental data
    theo_df = fragment_df
    expt_df = matched_df

    theo_df['column_label'] = theo_df.apply(lambda x: f'{"+" * x.charge}{x.ion_type}', axis=1)
    expt_df['column_label'] = expt_df.apply(lambda x: f'{"+" * x.charge}{x.ion_type}', axis=1)

    # remove losses
    theo_df = theo_df[theo_df['loss'] == 0]
    expt_df = expt_df[expt_df['loss'] == 0]

    # Pivot dataframes to have fragment ordinals as index and column labels as columns
    theo_df = theo_df.pivot(index='number', columns='column_label', values='mz').round(4)
    expt_df = expt_df.pivot(index='number', columns='column_label', values='mz').round(4)

    # Reindex experimental dataframe to ensure all ordinals up to peptide_length are included
    expt_df = expt_df.reindex(range(1, peptide_length + 1))

    # drop ordinal col and reindex
    theo_df = theo_df.reset_index(drop=True)
    expt_df = expt_df.reset_index(drop=True)

    # ensure expt has all cols
    for col in theo_df.columns:
        if col not in expt_df.columns:
            expt_df[col] = [None] * len(expt_df)

    # ensure that the dfs have the same order of cols (and I have the rows to be sorted by ion type)
    columns_order = sorted(theo_df.columns, key=lambda x: (x.split('+')[-1], x.count('+')))
    theo_df = theo_df[columns_order]
    expt_df = expt_df[columns_order]

    # Reverse the data for columns containing 'x', 'y', or 'z'
    for col in theo_df.columns:
        if any(x in col for x in ['x', 'y', 'z']):
            theo_df[col] = theo_df[col][::-1].reset_index(drop=True)
            expt_df[col] = expt_df[col][::-1].reset_index(drop=True)

    # Generate fill colors and text colors based on COLOR_DICT
    # Generate fill colors and text colors based on COLOR_DICT
    fill_colors = []
    text_colors = []

    for col in theo_df.columns:
        fill_colors.append([
            'lavender' if pd.notna(expt_df.at[ordinal, col]) else 'white'
            for ordinal in theo_df.index
        ])
        text_colors.append([COLOR_DICT.get(col[-1], 'black')] * peptide_length)

    # Add the new columns for the peptide length
    left_col = list(range(1, peptide_length + 1))
    right_col = list(range(peptide_length, 0, -1))

    middle_col = list(annotation.sequence)

    # Find index of first column that contains x, y, or z
    first_xyz_col = next((i for i, col in enumerate(theo_df.columns) if any(x in col for x in ['x', 'y', 'z'])), None)

    # Insert the middle column at the first index that contains x, y, or z
    theo_df.insert(first_xyz_col, 'Peptide', middle_col)

    # Insert the new columns into the DataFrame
    theo_df.insert(0, '(abc)', left_col)
    theo_df['(xyz)'] = right_col

    # Update fill_colors and text_colors for the new columns
    fill_colors.insert(first_xyz_col, ['white'] * peptide_length)
    fill_colors.insert(0, ['white'] * peptide_length)
    fill_colors.append(['white'] * peptide_length)
    text_colors.insert(first_xyz_col, ['black'] * peptide_length)
    text_colors.insert(0, ['black'] * peptide_length)
    text_colors.append(['black'] * peptide_length)

    # Create the figure
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(theo_df.columns),
                    fill_color='lightgrey',
                    align='center'),  # Center the header text
        cells=dict(
            values=[[val if pd.notna(val) else '' for val in theo_df[col].values] for col in theo_df.columns],
            fill_color=fill_colors,
            font=dict(color=text_colors),  # Make the text bold
            align='center',
        )  # Center the text in the cells
    )])

    guess_height = 230 + 20 * peptide_length

    fig.update_layout(
        width=800,  # Width of the figure in pixels
        height=guess_height,  # Height of the figure in pixels
        showlegend=False,
        title_text="Fragment Chart",
    )

    st.plotly_chart(fig, use_container_width=True)
