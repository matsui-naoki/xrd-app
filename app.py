"""
XRD Analyzer - Streamlit Application
A comprehensive tool for XRD pattern analysis using NMF and clustering
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import io
from typing import Dict, List, Optional

# Local imports
from tools.data_loader import load_xrd_file, validate_xrd_data, convert_to_dict_format, dict_to_files_format
from tools.preprocessing import (
    normalize_xrd, trim_2theta, remove_background,
    remove_xrd_nan, negative_to_zero, sort_dict_by_key,
    smooth_xrd, interpolate_xrd, preprocess_pipeline
)
from tools.analysis import (
    run_nmf, norm_array, calculate_dtw_matrix,
    calculate_cosine_matrix, calculate_correlation_matrix,
    dim_reduction, run_dbscan, merge_cluster_ratio,
    get_argmax_prob, normalize_coefficient_by_integral,
    find_optimal_n_components, analyze_xrd_pipeline, xrd_to_matrix
)
from components.plots import (
    create_xrd_plot, create_heatmap, create_nmf_summary_plot,
    create_cluster_scatter, create_ternary_plot,
    create_probability_heatmap, create_reconstruction_error_plot,
    create_basis_pattern_plot, create_coefficient_plot,
    create_sample_comparison_plot, COLORS
)
from components.styles import inject_custom_css
from utils.help_texts import get_help
from utils.logger import (
    log_info, log_warning, log_error,
    log_analysis_start, log_analysis_complete,
    log_preprocessing_start, log_preprocessing_complete,
    log_file_load, log_file_error
)


# Page configuration
st.set_page_config(
    page_title="XRD Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
inject_custom_css()


def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'files': {},
        'original_files': {},
        'processed_files': {},
        'selected_file': None,
        'analysis_results': None,
        'preprocessing_applied': False,

        # Preprocessing settings
        'preprocess_settings': {
            'normalize': True,
            'trim_range': [[10, 60]],
            'remove_bg': True,
            'remove_nan': True,
            'zero_negative': True,
            'smooth': False,
            'smooth_window': 11,
            'interpolate': False
        },

        # Analysis settings
        'analysis_settings': {
            'n_components': 10,
            'distance_method': 'DTW',
            'dtw_window': 30,
            'dim_reduction_method': 'MDS',
            'dbscan_eps': 20.0,
            'dbscan_min_samples': 1
        },

        # Mapping settings
        'mapping_settings': {
            'compositions': {},
            'axis_labels': ['A', 'B', 'C']
        },

        # UI state
        'current_tab': 'Data',
        'show_original': False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sidebar_header():
    """Render sidebar header"""
    st.sidebar.title("XRD Analyzer")
    st.sidebar.markdown("---")


def sidebar_file_upload():
    """Render file upload section in sidebar"""
    st.sidebar.header("1. Data Upload")

    with st.sidebar.expander("Upload XRD Files", expanded=True):
        uploaded_files = st.file_uploader(
            "Select XRD files",
            type=['xy', 'txt', 'csv', 'ras', 'dat'],
            accept_multiple_files=True,
            help=get_help('file_upload')
        )

        if uploaded_files:
            process_uploaded_files(uploaded_files)

        # Show loaded files count
        n_files = len(st.session_state['files'])
        if n_files > 0:
            st.success(f"{n_files} files loaded")

    # Session load/save
    with st.sidebar.expander("Session Management"):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Save Session", use_container_width=True):
                save_session()

        with col2:
            session_file = st.file_uploader(
                "Load Session",
                type=['json'],
                key='session_loader'
            )
            if session_file:
                load_session(session_file)


def process_uploaded_files(uploaded_files):
    """Process uploaded files and add to session state"""
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state['files']:
            try:
                # Read file content
                content = uploaded_file.read()
                uploaded_file.seek(0)  # Reset for potential re-read

                # Load XRD data
                data = load_xrd_file(io.BytesIO(content), uploaded_file.name)

                # Validate
                is_valid, error_msg = validate_xrd_data(data)
                if is_valid:
                    st.session_state['files'][uploaded_file.name] = data
                    st.session_state['original_files'][uploaded_file.name] = data.copy()
                    log_file_load(uploaded_file.name, len(data['two_theta']))
                else:
                    st.sidebar.error(f"Invalid file {uploaded_file.name}: {error_msg}")
                    log_file_error(uploaded_file.name, error_msg)

            except Exception as e:
                st.sidebar.error(f"Error loading {uploaded_file.name}: {str(e)}")
                log_file_error(uploaded_file.name, str(e))


def sidebar_file_manager():
    """Render file manager in sidebar"""
    if not st.session_state['files']:
        return

    st.sidebar.header("2. File Manager")

    with st.sidebar.expander("Loaded Files", expanded=True):
        files_to_remove = []

        for filename in st.session_state['files'].keys():
            col1, col2 = st.columns([4, 1])

            with col1:
                st.text(filename[:25] + "..." if len(filename) > 25 else filename)

            with col2:
                if st.button("×", key=f"remove_{filename}"):
                    files_to_remove.append(filename)

        # Remove marked files
        for filename in files_to_remove:
            del st.session_state['files'][filename]
            if filename in st.session_state['original_files']:
                del st.session_state['original_files'][filename]
            st.rerun()

        if st.button("Clear All Files", use_container_width=True):
            st.session_state['files'] = {}
            st.session_state['original_files'] = {}
            st.session_state['processed_files'] = {}
            st.session_state['analysis_results'] = None
            st.rerun()


def sidebar_preprocessing():
    """Render preprocessing settings in sidebar"""
    if not st.session_state['files']:
        return

    st.sidebar.header("3. Preprocessing")

    settings = st.session_state['preprocess_settings']

    with st.sidebar.expander("Preprocessing Options", expanded=True):
        # Normalize
        settings['normalize'] = st.checkbox(
            "Normalize to max=100",
            value=settings['normalize'],
            help=get_help('normalize')
        )

        # Trim 2theta range
        st.markdown("**2θ Range**")
        col1, col2 = st.columns(2)
        with col1:
            trim_min = st.number_input(
                "Min",
                value=float(settings['trim_range'][0][0]),
                min_value=0.0,
                max_value=180.0,
                step=1.0
            )
        with col2:
            trim_max = st.number_input(
                "Max",
                value=float(settings['trim_range'][0][1]),
                min_value=0.0,
                max_value=180.0,
                step=1.0
            )
        settings['trim_range'] = [[trim_min, trim_max]]

        # Background removal
        settings['remove_bg'] = st.checkbox(
            "Remove Background",
            value=settings['remove_bg'],
            help=get_help('background_removal')
        )

        # Additional options
        settings['remove_nan'] = st.checkbox(
            "Remove NaN values",
            value=settings['remove_nan']
        )

        settings['zero_negative'] = st.checkbox(
            "Set negative to zero",
            value=settings['zero_negative']
        )

        settings['smooth'] = st.checkbox(
            "Apply Smoothing",
            value=settings['smooth'],
            help=get_help('smoothing')
        )

        if settings['smooth']:
            settings['smooth_window'] = st.slider(
                "Smoothing Window",
                min_value=3,
                max_value=51,
                value=settings['smooth_window'],
                step=2
            )

        settings['interpolate'] = st.checkbox(
            "Interpolate to uniform spacing",
            value=settings['interpolate']
        )

        # Apply button
        if st.button("Apply Preprocessing", type="primary", use_container_width=True):
            apply_preprocessing()


def apply_preprocessing():
    """Apply preprocessing to loaded files"""
    if not st.session_state['files']:
        st.sidebar.error("No files loaded")
        return

    settings = st.session_state['preprocess_settings']
    n_files = len(st.session_state['files'])
    log_preprocessing_start(n_files)

    with st.spinner("Applying preprocessing..."):
        try:
            # Convert to dict format
            xrds = convert_to_dict_format(st.session_state['files'])

            # Apply preprocessing pipeline
            processed = preprocess_pipeline(
                xrds,
                normalize=settings['normalize'],
                trim_range=settings['trim_range'] if settings['trim_range'][0][0] < settings['trim_range'][0][1] else None,
                remove_bg=settings['remove_bg'],
                remove_nan=settings['remove_nan'],
                zero_negative=settings['zero_negative'],
                smooth=settings['smooth'],
                interpolate=settings['interpolate'],
                show_progress=False
            )

            # Convert back and store
            st.session_state['processed_files'] = dict_to_files_format(
                processed,
                st.session_state['files']
            )
            st.session_state['preprocessing_applied'] = True
            log_preprocessing_complete()
            st.sidebar.success("Preprocessing complete!")

        except Exception as e:
            log_error(f"Preprocessing failed: {str(e)}", exc_info=True)
            st.sidebar.error(f"Preprocessing failed: {str(e)}")


def sidebar_analysis():
    """Render analysis settings in sidebar"""
    if not st.session_state['files']:
        return

    st.sidebar.header("4. Analysis")

    settings = st.session_state['analysis_settings']

    with st.sidebar.expander("NMF Settings", expanded=True):
        settings['n_components'] = st.slider(
            "Number of Components",
            min_value=2,
            max_value=30,
            value=settings['n_components'],
            help=get_help('nmf_components')
        )

        if st.button("Find Optimal Components"):
            find_optimal_components()

    with st.sidebar.expander("Clustering Settings"):
        settings['distance_method'] = st.selectbox(
            "Distance Method",
            options=['DTW', 'cosine', 'correlation'],
            index=['DTW', 'cosine', 'correlation'].index(settings['distance_method']),
            help=get_help('distance_method')
        )

        if settings['distance_method'] == 'DTW':
            settings['dtw_window'] = st.slider(
                "DTW Window Size",
                min_value=5,
                max_value=100,
                value=settings['dtw_window'],
                help=get_help('dtw_window')
            )

        settings['dim_reduction_method'] = st.selectbox(
            "Dimension Reduction",
            options=['MDS', 'tSNE', 'UMAP'],
            index=['MDS', 'tSNE', 'UMAP'].index(settings['dim_reduction_method']),
            help=get_help('dim_reduction')
        )

        settings['dbscan_eps'] = st.slider(
            "DBSCAN eps",
            min_value=1.0,
            max_value=100.0,
            value=float(settings['dbscan_eps']),
            step=1.0,
            help=get_help('dbscan_eps')
        )

        settings['dbscan_min_samples'] = st.slider(
            "DBSCAN min_samples",
            min_value=1,
            max_value=10,
            value=settings['dbscan_min_samples']
        )

    # Run analysis button
    if st.sidebar.button("Run Analysis", type="primary", use_container_width=True):
        run_analysis()


def find_optimal_components():
    """Find optimal number of NMF components"""
    data = get_current_data()
    if not data:
        st.sidebar.error("No data available")
        return

    xrds = convert_to_dict_format(data)

    with st.spinner("Finding optimal components..."):
        errors = find_optimal_n_components(xrds, max_components=20)

    if errors:
        fig = create_reconstruction_error_plot(errors)
        st.sidebar.plotly_chart(fig, use_container_width=True)


def run_analysis():
    """Run full XRD analysis pipeline"""
    data = get_current_data()
    if not data:
        st.sidebar.error("No data available. Please load files first.")
        return

    settings = st.session_state['analysis_settings']
    xrds = convert_to_dict_format(data)
    n_samples = len(xrds)

    log_analysis_start(n_samples, settings['n_components'], settings['distance_method'])

    with st.spinner("Running analysis..."):
        try:
            results = analyze_xrd_pipeline(
                xrds,
                n_components=settings['n_components'],
                distance_method=settings['distance_method'],
                dim_reduction_method=settings['dim_reduction_method'],
                dbscan_eps=settings['dbscan_eps'],
                dbscan_min_samples=settings['dbscan_min_samples'],
                window_size=settings['dtw_window'],
                show_progress=False
            )

            st.session_state['analysis_results'] = results
            log_analysis_complete(results['n_clusters'], results['reconstruction_error'])
            st.sidebar.success(f"Analysis complete! Found {results['n_clusters']} clusters.")

        except ValueError as e:
            log_error(f"Invalid parameters: {str(e)}")
            st.sidebar.error(f"Invalid parameters: {str(e)}")
        except MemoryError as e:
            log_error(f"Memory error - dataset too large: {str(e)}")
            st.sidebar.error("Memory error: Dataset may be too large. Try reducing sample count.")
        except Exception as e:
            log_error(f"Analysis failed: {str(e)}", exc_info=True)
            st.sidebar.error(f"Analysis failed: {str(e)}")


def get_current_data():
    """Get current data (processed if available, otherwise original)"""
    if st.session_state['preprocessing_applied'] and st.session_state['processed_files']:
        return st.session_state['processed_files']
    return st.session_state['files']


def main_panel():
    """Render main panel with tabs"""
    st.title("XRD Pattern Analyzer")

    if not st.session_state['files']:
        st.info("Please upload XRD files using the sidebar to get started.")
        show_welcome_message()
        return

    # Create tabs
    tabs = st.tabs(["📊 Data", "🔧 Preprocessing", "🔬 Analysis", "🗺️ Mapping", "📋 Results"])

    with tabs[0]:
        show_data_tab()

    with tabs[1]:
        show_preprocessing_tab()

    with tabs[2]:
        show_analysis_tab()

    with tabs[3]:
        show_mapping_tab()

    with tabs[4]:
        show_results_tab()


def show_welcome_message():
    """Show welcome message and instructions"""
    st.markdown("""
    ## Welcome to XRD Analyzer

    This application provides comprehensive tools for X-ray diffraction pattern analysis:

    ### Features
    - **Data Loading**: Support for multiple XRD file formats
    - **Preprocessing**: Normalization, background removal, smoothing
    - **NMF Analysis**: Non-negative matrix factorization for pattern decomposition
    - **Clustering**: DBSCAN clustering with DTW distance
    - **Visualization**: Interactive plots and phase diagrams

    ### Getting Started
    1. Upload your XRD files using the sidebar
    2. Apply preprocessing to clean and normalize data
    3. Run NMF analysis to extract basis patterns
    4. Explore clustering results and phase mappings

    ### Supported File Formats
    - `.xy`, `.txt`: Two-column format (2θ, intensity)
    - `.csv`: CSV with 2θ and intensity columns
    - `.ras`: Rigaku RAS format
    """)


def show_data_tab():
    """Show data visualization tab"""
    st.header("XRD Data Visualization")

    data = get_current_data()
    if not data:
        st.warning("No data loaded")
        return

    # Convert to dict format for plotting
    xrds = convert_to_dict_format(data)

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Plot Options")
        plot_type = st.radio(
            "Plot Type",
            options=["Stacked Patterns", "Heatmap", "Overlay"]
        )

        show_original = st.checkbox(
            "Show Original Data",
            value=st.session_state['show_original'],
            disabled=not st.session_state['preprocessing_applied']
        )
        st.session_state['show_original'] = show_original

        # Sample selection for large datasets
        sample_ids = sorted(xrds.keys())
        if len(sample_ids) > 20:
            selected_range = st.slider(
                "Sample Range",
                min_value=min(sample_ids),
                max_value=max(sample_ids),
                value=(min(sample_ids), min(sample_ids) + 20)
            )
            selected_samples = [s for s in sample_ids if selected_range[0] <= s <= selected_range[1]]
        else:
            selected_samples = sample_ids

    with col1:
        if show_original and st.session_state['original_files']:
            original_xrds = convert_to_dict_format(st.session_state['original_files'])
            display_data = original_xrds
            title_suffix = " (Original)"
        else:
            display_data = xrds
            title_suffix = " (Processed)" if st.session_state['preprocessing_applied'] else ""

        if plot_type == "Stacked Patterns":
            fig = create_xrd_plot(
                display_data,
                selected_samples=selected_samples,
                title=f"XRD Patterns{title_suffix}",
                offset_mode=True
            )
        elif plot_type == "Heatmap":
            fig = create_heatmap(
                display_data,
                title=f"XRD Heatmap{title_suffix}"
            )
        else:  # Overlay
            fig = create_xrd_plot(
                display_data,
                selected_samples=selected_samples,
                title=f"XRD Patterns{title_suffix}",
                offset_mode=False
            )

        st.plotly_chart(fig, use_container_width=True)

    # Data statistics
    st.subheader("Data Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Number of Samples", len(xrds))

    with col2:
        first_sample = list(xrds.values())[0]
        st.metric("Data Points per Sample", len(first_sample[0]))

    with col3:
        two_theta_min = min(min(v[0]) for v in xrds.values())
        two_theta_max = max(max(v[0]) for v in xrds.values())
        st.metric("2θ Range", f"{two_theta_min:.1f}° - {two_theta_max:.1f}°")

    with col4:
        status = "Processed" if st.session_state['preprocessing_applied'] else "Raw"
        st.metric("Data Status", status)


def show_preprocessing_tab():
    """Show preprocessing comparison tab"""
    st.header("Preprocessing Results")

    if not st.session_state['preprocessing_applied']:
        st.info("Apply preprocessing from the sidebar to see results here.")
        return

    if not st.session_state['original_files'] or not st.session_state['processed_files']:
        st.warning("No preprocessing data available")
        return

    original_xrds = convert_to_dict_format(st.session_state['original_files'])
    processed_xrds = convert_to_dict_format(st.session_state['processed_files'])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Data")
        fig1 = create_heatmap(original_xrds, title="Original XRD Patterns")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Processed Data")
        fig2 = create_heatmap(processed_xrds, title="Processed XRD Patterns")
        st.plotly_chart(fig2, use_container_width=True)

    # Individual sample comparison
    st.subheader("Sample Comparison")

    sample_ids = sorted(original_xrds.keys())
    selected_sample = st.selectbox(
        "Select Sample",
        options=sample_ids,
        format_func=lambda x: f"Sample {x}"
    )

    if selected_sample:
        fig = create_sample_comparison_plot(
            processed_xrds,
            original_xrds,
            selected_sample
        )
        st.plotly_chart(fig, use_container_width=True)


def show_analysis_tab():
    """Show analysis results tab"""
    st.header("NMF Analysis Results")

    results = st.session_state['analysis_results']

    if results is None:
        st.info("Run analysis from the sidebar to see results here.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Number of Components", len(results['basis_vector']))

    with col2:
        st.metric("Number of Clusters", results['n_clusters'])

    with col3:
        st.metric("Reconstruction Error", f"{results['reconstruction_error']:.2f}%")

    with col4:
        st.metric("Number of Samples", len(results['sample_ids']))

    st.markdown("---")

    # NMF Summary Plot
    st.subheader("NMF and Clustering Summary")
    fig = create_nmf_summary_plot(
        results['labels'],
        results['norm_basis_vector'],
        results['embedding'],
        results['two_theta']
    )
    st.plotly_chart(fig, use_container_width=True)

    # Additional plots
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basis Patterns")
        fig = create_basis_pattern_plot(
            results['norm_basis_vector'],
            results['two_theta'],
            labels=results['labels']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cluster Distribution")
        fig = create_cluster_scatter(
            results['embedding'],
            results['labels']
        )
        st.plotly_chart(fig, use_container_width=True)

    # Coefficient heatmap
    st.subheader("NMF Coefficients")
    fig = create_coefficient_plot(
        results['coefficient'],
        results['sample_ids']
    )
    st.plotly_chart(fig, use_container_width=True)

    # Probability heatmap
    st.subheader("Cluster Probabilities")
    fig = create_probability_heatmap(
        np.array(results['probabilities']),
        results['sample_ids']
    )
    st.plotly_chart(fig, use_container_width=True)


def show_mapping_tab():
    """Show composition mapping tab"""
    st.header("Composition Mapping")

    results = st.session_state['analysis_results']

    if results is None:
        st.info("Run analysis first to access mapping features.")
        return

    st.subheader("Ternary Phase Diagram")
    st.markdown("""
    Enter composition data for each sample to create a ternary phase diagram.
    Compositions should be in the format: A, B, C (values will be normalized to sum to 1).
    """)

    # Composition input
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("**Axis Labels**")
        axis_a = st.text_input("Component A", value="A")
        axis_b = st.text_input("Component B", value="B")
        axis_c = st.text_input("Component C", value="C")

        st.markdown("**Quick Input**")
        use_sample_data = st.checkbox("Use sample data")

    with col1:
        compositions = st.session_state['mapping_settings']['compositions']

        if use_sample_data:
            # Generate sample compositions
            n_samples = len(results['sample_ids'])
            sample_compositions = []
            for i in range(n_samples):
                a = np.random.random()
                b = np.random.random() * (1 - a)
                c = 1 - a - b
                sample_compositions.append([a, b, c])
            compositions = {sid: comp for sid, comp in zip(results['sample_ids'], sample_compositions)}
        else:
            st.markdown("**Enter compositions (comma-separated: a, b, c)**")

            # Create input fields for each sample
            for sid in results['sample_ids'][:10]:  # Limit to first 10 for UI
                default_val = compositions.get(sid, [0.33, 0.33, 0.34])
                comp_str = st.text_input(
                    f"Sample {sid}",
                    value=f"{default_val[0]:.3f}, {default_val[1]:.3f}, {default_val[2]:.3f}",
                    key=f"comp_{sid}"
                )
                try:
                    parts = [float(x.strip()) for x in comp_str.split(',')]
                    if len(parts) == 3:
                        compositions[sid] = parts
                except:
                    pass

        st.session_state['mapping_settings']['compositions'] = compositions

    # Create ternary plot if compositions are available
    if compositions and len(compositions) >= 3:
        # Build composition list matching sample order
        comp_list = []
        labels_list = []

        for i, sid in enumerate(results['sample_ids']):
            if sid in compositions:
                comp_list.append(compositions[sid])
                labels_list.append(results['argmax'][i])

        if comp_list:
            fig = create_ternary_plot(
                comp_list,
                labels_list,
                axis_labels=[axis_a, axis_b, axis_c],
                title="Ternary Phase Diagram"
            )
            st.plotly_chart(fig, use_container_width=True)


def show_results_tab():
    """Show results export tab"""
    st.header("Results Summary & Export")

    results = st.session_state['analysis_results']

    if results is None:
        st.info("Run analysis first to see results.")
        return

    # Results table
    st.subheader("Sample Classification Results")

    df_data = {
        'Sample ID': results['sample_ids'],
        'Cluster': [x + 1 for x in results['argmax']],
        'Max Probability': [max(p) for p in results['probabilities']]
    }

    # Add probability columns
    n_clusters = results['n_clusters']
    for i in range(n_clusters):
        df_data[f'Cluster {i+1} Prob'] = [p[i] if i < len(p) else 0 for p in results['probabilities']]

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)

    # Export buttons
    st.subheader("Export")

    col1, col2, col3 = st.columns(3)

    with col1:
        # CSV export
        csv = df.to_csv(index=False)
        st.download_button(
            "Download Results CSV",
            data=csv,
            file_name="xrd_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # Basis vectors export
        basis_df = pd.DataFrame(
            results['basis_vector'],
            columns=[f"2θ_{i}" for i in range(len(results['two_theta']))],
            index=[f"Component_{i+1}" for i in range(len(results['basis_vector']))]
        )
        basis_csv = basis_df.to_csv()
        st.download_button(
            "Download Basis Vectors",
            data=basis_csv,
            file_name="nmf_basis_vectors.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col3:
        # Coefficients export
        coef_df = pd.DataFrame(
            results['coefficient'],
            columns=[f"Component_{i+1}" for i in range(len(results['basis_vector']))],
            index=[f"Sample_{sid}" for sid in results['sample_ids']]
        )
        coef_csv = coef_df.to_csv()
        st.download_button(
            "Download Coefficients",
            data=coef_csv,
            file_name="nmf_coefficients.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Cluster summary
    st.subheader("Cluster Summary")

    cluster_counts = {}
    for label in results['argmax']:
        cluster_counts[label + 1] = cluster_counts.get(label + 1, 0) + 1

    cluster_df = pd.DataFrame({
        'Cluster': list(cluster_counts.keys()),
        'Count': list(cluster_counts.values()),
        'Percentage': [f"{100*c/len(results['argmax']):.1f}%" for c in cluster_counts.values()]
    })

    st.dataframe(cluster_df, use_container_width=True)


def save_session():
    """Save current session to JSON"""
    session_data = {
        'files': {k: {
            'two_theta': v['two_theta'].tolist() if hasattr(v['two_theta'], 'tolist') else v['two_theta'],
            'intensity': v['intensity'].tolist() if hasattr(v['intensity'], 'tolist') else v['intensity'],
            'filename': v.get('filename', k)
        } for k, v in st.session_state['files'].items()},
        'preprocess_settings': st.session_state['preprocess_settings'],
        'analysis_settings': st.session_state['analysis_settings'],
        'preprocessing_applied': st.session_state['preprocessing_applied']
    }

    # Add processed files if available
    if st.session_state['processed_files']:
        session_data['processed_files'] = {k: {
            'two_theta': v['two_theta'].tolist() if hasattr(v['two_theta'], 'tolist') else v['two_theta'],
            'intensity': v['intensity'].tolist() if hasattr(v['intensity'], 'tolist') else v['intensity'],
            'filename': v.get('filename', k)
        } for k, v in st.session_state['processed_files'].items()}

    json_str = json.dumps(session_data, indent=2)

    st.sidebar.download_button(
        "Download Session File",
        data=json_str,
        file_name="xrd_session.json",
        mime="application/json",
        use_container_width=True
    )


def load_session(session_file):
    """Load session from JSON file"""
    try:
        session_data = json.load(session_file)

        # Restore files
        st.session_state['files'] = {k: {
            'two_theta': np.array(v['two_theta']),
            'intensity': np.array(v['intensity']),
            'filename': v.get('filename', k)
        } for k, v in session_data.get('files', {}).items()}

        st.session_state['original_files'] = dict(st.session_state['files'])

        # Restore processed files
        if 'processed_files' in session_data:
            st.session_state['processed_files'] = {k: {
                'two_theta': np.array(v['two_theta']),
                'intensity': np.array(v['intensity']),
                'filename': v.get('filename', k)
            } for k, v in session_data['processed_files'].items()}

        # Restore settings
        st.session_state['preprocess_settings'] = session_data.get(
            'preprocess_settings',
            st.session_state['preprocess_settings']
        )
        st.session_state['analysis_settings'] = session_data.get(
            'analysis_settings',
            st.session_state['analysis_settings']
        )
        st.session_state['preprocessing_applied'] = session_data.get(
            'preprocessing_applied',
            False
        )

        st.sidebar.success("Session loaded successfully!")
        st.rerun()

    except Exception as e:
        st.sidebar.error(f"Failed to load session: {str(e)}")


def main():
    """Main application entry point"""
    initialize_session_state()

    # Sidebar
    sidebar_header()
    sidebar_file_upload()
    sidebar_file_manager()
    sidebar_preprocessing()
    sidebar_analysis()

    # Main panel
    main_panel()


if __name__ == "__main__":
    main()
