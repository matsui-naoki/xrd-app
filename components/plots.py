"""
XRD Visualization Components
Plotly-based interactive plots for XRD analysis
with publication-ready figure export capabilities
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union
import io

# Color palette
COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
    '#FF0000',  # red
    '#0000FF',  # blue
    '#008000',  # green
    '#800080',  # purple
    '#ffa500',  # orange
]

# Publication-ready styling settings
PUBLICATION_STYLE = {
    'font_family': 'Arial',
    'font_size': 14,
    'axis_line_width': 1,
    'tick_font_size': 12,
    'title_font_size': 16,
    'legend_font_size': 11,
    'line_width': 1.5,
    'marker_size': 8,
}


def apply_publication_style(fig: go.Figure,
                            width: int = 800,
                            height: int = 600,
                            show_grid: bool = False) -> go.Figure:
    """
    Apply publication-ready styling to a plotly figure

    Args:
        fig: Plotly figure to style
        width: Figure width in pixels
        height: Figure height in pixels
        show_grid: Whether to show grid lines

    Returns:
        Styled figure
    """
    style = PUBLICATION_STYLE

    fig.update_layout(
        font=dict(
            family=style['font_family'],
            size=style['font_size'],
            color='black'
        ),
        width=width,
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=70, r=20, t=50, b=60),
    )

    # Update axes
    axis_settings = dict(
        showgrid=show_grid,
        gridcolor='lightgray' if show_grid else None,
        showline=True,
        linewidth=style['axis_line_width'],
        linecolor='black',
        tickcolor='black',
        tickfont=dict(
            family=style['font_family'],
            size=style['tick_font_size'],
            color='black'
        ),
        title_font=dict(
            family=style['font_family'],
            size=style['title_font_size'],
            color='black'
        ),
        mirror=True,
        ticks='inside',
        ticklen=5,
        zeroline=False,
    )

    fig.update_xaxes(**axis_settings)
    fig.update_yaxes(**axis_settings)

    # Update legend
    fig.update_layout(
        legend=dict(
            font=dict(
                family=style['font_family'],
                size=style['legend_font_size'],
                color='black'
            ),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1,
        )
    )

    return fig


def export_figure_bytes(fig: go.Figure,
                        format: str = 'png',
                        width: int = 800,
                        height: int = 600,
                        scale: int = 3) -> bytes:
    """
    Export figure to bytes for download

    Args:
        fig: Plotly figure
        format: Export format ('png', 'svg', 'pdf')
        width: Figure width
        height: Figure height
        scale: Scale factor for resolution (3 = 300 DPI at 100% size)

    Returns:
        Image bytes
    """
    try:
        import kaleido
        return fig.to_image(
            format=format,
            width=width,
            height=height,
            scale=scale
        )
    except ImportError:
        raise ImportError("kaleido is required for image export. Install with: pip install kaleido")


def create_xrd_plot(xrd_data: Dict[int, List],
                    selected_samples: Optional[List[int]] = None,
                    title: str = "XRD Patterns",
                    offset_mode: bool = True,
                    show_legend: bool = True) -> go.Figure:
    """
    Create XRD pattern plot

    Args:
        xrd_data: Dictionary with sample IDs as keys and [two_theta, intensity] as values
        selected_samples: List of sample IDs to plot (None = all)
        title: Plot title
        offset_mode: Whether to stack patterns with offset
        show_legend: Whether to show legend

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    samples = selected_samples if selected_samples else sorted(xrd_data.keys())
    offset = 0

    for i, sample_id in enumerate(samples):
        if sample_id not in xrd_data:
            continue

        two_theta = xrd_data[sample_id][0]
        intensity = np.array(xrd_data[sample_id][1])

        if offset_mode:
            y_data = intensity + offset
            offset += np.max(intensity) * 1.1
        else:
            y_data = intensity

        fig.add_trace(go.Scatter(
            x=two_theta,
            y=y_data,
            mode='lines',
            name=f'Sample {sample_id}',
            line=dict(color=COLORS[i % len(COLORS)], width=1),
            showlegend=show_legend
        ))

    fig.update_layout(
        title=title,
        xaxis_title='2θ (degree)',
        yaxis_title='Intensity (a.u.)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    if offset_mode:
        fig.update_yaxes(showticklabels=False)

    return fig


def create_heatmap(xrd_data: Dict[int, List],
                   title: str = "XRD Heatmap",
                   colorscale: str = "Jet") -> go.Figure:
    """
    Create XRD heatmap

    Args:
        xrd_data: Dictionary with sample IDs as keys and [two_theta, intensity] as values
        title: Plot title
        colorscale: Colorscale name

    Returns:
        Plotly figure
    """
    keys = sorted(xrd_data.keys())
    values = [xrd_data[k] for k in keys]

    # Get 2theta from first sample
    x_axis = np.array(values[0][0])

    # Create intensity matrix
    y_matrix = np.array([v[1] for v in values])

    fig = go.Figure(data=go.Heatmap(
        z=y_matrix,
        x=x_axis,
        y=keys,
        colorscale=colorscale,
        colorbar=dict(title='Intensity')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='2θ (degree)',
        yaxis_title='Sample Number',
        template='plotly_white'
    )

    return fig


def create_nmf_summary_plot(labels: np.ndarray,
                            norm_basis_vector: np.ndarray,
                            embedding: np.ndarray,
                            two_theta: np.ndarray) -> go.Figure:
    """
    Create NMF analysis summary plot (3 subplots)

    Args:
        labels: Cluster labels for each basis vector
        norm_basis_vector: Normalized basis vectors
        embedding: 2D embedding from dimension reduction
        two_theta: 2theta values

    Returns:
        Plotly figure with 3 subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Basis Patterns from NMF',
            'Clustering (DTW → MDS → DBSCAN)',
            'Summed Basis Patterns per Cluster'
        ),
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)

    # Create color map
    colors_map = {label: COLORS[i % len(COLORS)] for i, label in enumerate(unique_labels)}

    # Plot 1: Basis patterns (left, full height)
    offset = 0
    done = []
    for i, label in enumerate(labels):
        if label not in done:
            indices = [j for j, x in enumerate(labels) if x == label]
            for idx, index in enumerate(indices):
                show_legend = idx == 0
                fig.add_trace(go.Scatter(
                    x=two_theta,
                    y=norm_basis_vector[index] + offset,
                    mode='lines',
                    name=f'Cluster {label + 1}',
                    line=dict(color=colors_map[label], width=1),
                    showlegend=show_legend,
                    legendgroup=f'cluster_{label}'
                ), row=1, col=1)
                offset += np.max(norm_basis_vector[index]) * 1.2
            done.append(label)

    # Plot 2: Cluster scatter (top right)
    for label in unique_labels:
        indices = [j for j, x in enumerate(labels) if x == label]
        fig.add_trace(go.Scatter(
            x=embedding[indices, 0],
            y=embedding[indices, 1],
            mode='markers',
            name=f'Cluster {label + 1}',
            marker=dict(color=colors_map[label], size=10),
            showlegend=False,
            legendgroup=f'cluster_{label}'
        ), row=1, col=2)

    # Plot 3: Summed basis patterns (bottom right)
    offset3 = 0
    done3 = []
    for i, label in enumerate(labels):
        if label not in done3:
            indices = [j for j, x in enumerate(labels) if x == label]
            sum_basis = np.mean([norm_basis_vector[idx] for idx in indices], axis=0)

            fig.add_trace(go.Scatter(
                x=two_theta,
                y=sum_basis + offset3,
                mode='lines',
                name=f'Cluster {label + 1}',
                line=dict(color=colors_map[label], width=2),
                showlegend=False,
                legendgroup=f'cluster_{label}'
            ), row=2, col=2)
            offset3 += np.max(sum_basis) * 1.2
            done3.append(label)

    # Update layout
    fig.update_xaxes(title_text='2θ (degree)', row=1, col=1)
    fig.update_yaxes(title_text='Intensity (a.u.)', showticklabels=False, row=1, col=1)

    fig.update_xaxes(title_text='Feature 1', row=1, col=2)
    fig.update_yaxes(title_text='Feature 2', row=1, col=2)

    fig.update_xaxes(title_text='2θ (degree)', row=2, col=2)
    fig.update_yaxes(title_text='Intensity (a.u.)', showticklabels=False, row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )

    return fig


def create_cluster_scatter(embedding: np.ndarray,
                           labels: np.ndarray,
                           title: str = "Cluster Distribution") -> go.Figure:
    """
    Create cluster scatter plot

    Args:
        embedding: 2D embedding array
        labels: Cluster labels
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    unique_labels = sorted(set(labels))

    for label in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == label]
        fig.add_trace(go.Scatter(
            x=embedding[indices, 0],
            y=embedding[indices, 1],
            mode='markers',
            name=f'Cluster {label + 1}',
            marker=dict(
                color=COLORS[label % len(COLORS)],
                size=12,
                opacity=0.8
            )
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig


def create_ternary_plot(compositions: List[List[float]],
                        labels: List[int],
                        axis_labels: List[str] = ['A', 'B', 'C'],
                        title: str = "Ternary Phase Diagram") -> go.Figure:
    """
    Create ternary plot for 3-component systems

    Args:
        compositions: List of [a, b, c] compositions
        labels: Cluster labels for each composition
        axis_labels: Labels for the three axes
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'ternary'}]])

    unique_labels = sorted(set(labels))
    compositions = np.array(compositions)

    for label in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == label]
        fig.add_trace(go.Scatterternary(
            a=compositions[indices, 0],
            b=compositions[indices, 1],
            c=compositions[indices, 2],
            mode='markers',
            marker=dict(
                size=10,
                color=COLORS[label % len(COLORS)],
                opacity=0.7,
                symbol='hexagon2'
            ),
            name=f'Cluster {label + 1}',
            showlegend=True
        ))

    fig.update_layout(
        title=title,
        ternary=dict(
            sum=1,
            aaxis=dict(
                title=axis_labels[0],
                titlefont=dict(size=16),
                tickfont=dict(size=12),
                linecolor='black',
                gridcolor='lightgray'
            ),
            baxis=dict(
                title=axis_labels[1],
                titlefont=dict(size=16),
                tickfont=dict(size=12),
                linecolor='black',
                gridcolor='lightgray'
            ),
            caxis=dict(
                title=axis_labels[2],
                titlefont=dict(size=16),
                tickfont=dict(size=12),
                linecolor='black',
                gridcolor='lightgray'
            ),
            bgcolor='white'
        ),
        template='plotly_white'
    )

    return fig


def create_probability_heatmap(probabilities: np.ndarray,
                               sample_ids: List,
                               title: str = "Cluster Probability Distribution") -> go.Figure:
    """
    Create heatmap of cluster probabilities

    Args:
        probabilities: Probability matrix (n_samples, n_clusters)
        sample_ids: Sample IDs
        title: Plot title

    Returns:
        Plotly figure
    """
    n_clusters = probabilities.shape[1]
    cluster_names = [f'Cluster {i+1}' for i in range(n_clusters)]

    fig = go.Figure(data=go.Heatmap(
        z=probabilities,
        x=cluster_names,
        y=sample_ids,
        colorscale='Blues',
        colorbar=dict(title='Probability')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Cluster',
        yaxis_title='Sample',
        template='plotly_white'
    )

    return fig


def create_reconstruction_error_plot(errors: List[Tuple[int, float]],
                                     title: str = "NMF Reconstruction Error") -> go.Figure:
    """
    Create plot of reconstruction error vs number of components

    Args:
        errors: List of (n_components, error) tuples
        title: Plot title

    Returns:
        Plotly figure
    """
    n_components = [e[0] for e in errors]
    error_values = [e[1] for e in errors]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=n_components,
        y=error_values,
        mode='lines+markers',
        name='Reconstruction Error',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Number of Components',
        yaxis_title='Reconstruction Error (%)',
        template='plotly_white'
    )

    return fig


def create_basis_pattern_plot(basis_vectors: np.ndarray,
                              two_theta: np.ndarray,
                              labels: Optional[np.ndarray] = None,
                              title: str = "NMF Basis Patterns") -> go.Figure:
    """
    Create plot of NMF basis patterns

    Args:
        basis_vectors: Basis vector matrix (n_components, n_features)
        two_theta: 2theta values
        labels: Optional cluster labels for coloring
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    offset = 0
    n_components = len(basis_vectors)

    for i in range(n_components):
        if labels is not None:
            color = COLORS[labels[i] % len(COLORS)]
            name = f'Component {i+1} (Cluster {labels[i]+1})'
        else:
            color = COLORS[i % len(COLORS)]
            name = f'Component {i+1}'

        fig.add_trace(go.Scatter(
            x=two_theta,
            y=basis_vectors[i] + offset,
            mode='lines',
            name=name,
            line=dict(color=color, width=1.5)
        ))
        offset += np.max(basis_vectors[i]) * 1.2

    fig.update_layout(
        title=title,
        xaxis_title='2θ (degree)',
        yaxis_title='Intensity (a.u.)',
        template='plotly_white',
        yaxis=dict(showticklabels=False),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig


def create_coefficient_plot(coefficient: np.ndarray,
                            sample_ids: List,
                            title: str = "NMF Coefficients") -> go.Figure:
    """
    Create heatmap of NMF coefficients

    Args:
        coefficient: Coefficient matrix (n_samples, n_components)
        sample_ids: Sample IDs
        title: Plot title

    Returns:
        Plotly figure
    """
    n_components = coefficient.shape[1]
    component_names = [f'Comp. {i+1}' for i in range(n_components)]

    fig = go.Figure(data=go.Heatmap(
        z=coefficient,
        x=component_names,
        y=sample_ids,
        colorscale='Viridis',
        colorbar=dict(title='Coefficient')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Component',
        yaxis_title='Sample',
        template='plotly_white'
    )

    return fig


def create_sample_comparison_plot(xrd_data: Dict[int, List],
                                  original_data: Dict[int, List],
                                  sample_id: int,
                                  title: str = "Before/After Comparison") -> go.Figure:
    """
    Create comparison plot of original vs processed data

    Args:
        xrd_data: Processed XRD data
        original_data: Original XRD data
        sample_id: Sample ID to compare
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    if sample_id in original_data:
        fig.add_trace(go.Scatter(
            x=original_data[sample_id][0],
            y=original_data[sample_id][1],
            mode='lines',
            name='Original',
            line=dict(color='gray', width=1)
        ))

    if sample_id in xrd_data:
        fig.add_trace(go.Scatter(
            x=xrd_data[sample_id][0],
            y=xrd_data[sample_id][1],
            mode='lines',
            name='Processed',
            line=dict(color='#1f77b4', width=1.5)
        ))

    fig.update_layout(
        title=f'{title} - Sample {sample_id}',
        xaxis_title='2θ (degree)',
        yaxis_title='Intensity (a.u.)',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig


def create_multi_view_nmf_plot(xrd_basis: np.ndarray,
                                comp_basis: np.ndarray,
                                coefficients: np.ndarray,
                                two_theta: np.ndarray,
                                element_names: List[str],
                                title: str = "Multi-view NMF Results") -> go.Figure:
    """
    Create visualization for multi-view NMF results

    Args:
        xrd_basis: XRD basis vectors (n_components, n_2theta)
        comp_basis: Composition basis vectors (n_components, n_elements)
        coefficients: Shared coefficient matrix (n_samples, n_components)
        two_theta: 2theta values
        element_names: List of element names
        title: Plot title

    Returns:
        Plotly figure with subplots
    """
    n_components = xrd_basis.shape[0]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'XRD Basis Patterns',
            'Composition Basis',
            'Sample Coefficients',
            'Component Contributions'
        ),
        specs=[[{"rowspan": 1}, {"type": "bar"}],
               [{"type": "heatmap"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    # Plot 1: XRD Basis Patterns
    offset = 0
    for i in range(n_components):
        fig.add_trace(go.Scatter(
            x=two_theta,
            y=xrd_basis[i] / np.max(xrd_basis[i]) + offset,
            mode='lines',
            name=f'Component {i+1}',
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            legendgroup=f'comp_{i}',
            showlegend=True
        ), row=1, col=1)
        offset += 1.2

    # Plot 2: Composition Basis (bar chart)
    x_positions = np.arange(len(element_names))
    bar_width = 0.8 / n_components

    for i in range(n_components):
        fig.add_trace(go.Bar(
            x=[e + (i - n_components/2 + 0.5) * bar_width for e in x_positions],
            y=comp_basis[i],
            name=f'Component {i+1}',
            marker_color=COLORS[i % len(COLORS)],
            width=bar_width,
            legendgroup=f'comp_{i}',
            showlegend=False
        ), row=1, col=2)

    # Plot 3: Sample Coefficients Heatmap
    fig.add_trace(go.Heatmap(
        z=coefficients,
        x=[f'Comp {i+1}' for i in range(n_components)],
        y=[f'Sample {i+1}' for i in range(len(coefficients))],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Coeff.', x=0.45)
    ), row=2, col=1)

    # Plot 4: Average Component Contributions
    avg_coefficients = np.mean(coefficients, axis=0)
    fig.add_trace(go.Bar(
        x=[f'Comp {i+1}' for i in range(n_components)],
        y=avg_coefficients,
        marker_color=[COLORS[i % len(COLORS)] for i in range(n_components)],
        showlegend=False
    ), row=2, col=2)

    # Update layout
    fig.update_xaxes(title_text='2θ (degree)', row=1, col=1)
    fig.update_yaxes(title_text='Intensity (a.u.)', showticklabels=False, row=1, col=1)

    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(len(element_names))),
        ticktext=element_names,
        row=1, col=2
    )
    fig.update_yaxes(title_text='Composition', row=1, col=2)

    fig.update_xaxes(title_text='Component', row=2, col=1)
    fig.update_yaxes(title_text='Sample', row=2, col=1)

    fig.update_xaxes(title_text='Component', row=2, col=2)
    fig.update_yaxes(title_text='Average Coefficient', row=2, col=2)

    fig.update_layout(
        title=title,
        height=800,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def create_endmember_plot(compositions: np.ndarray,
                           endmembers: np.ndarray,
                           coefficients: np.ndarray,
                           element_names: List[str],
                           title: str = "Endmember Decomposition") -> go.Figure:
    """
    Create visualization for endmember decomposition results

    Args:
        compositions: Original sample compositions (n_samples, n_elements)
        endmembers: Endmember compositions (n_endmembers, n_elements)
        coefficients: Decomposition coefficients (n_samples, n_endmembers)
        element_names: List of element names
        title: Plot title

    Returns:
        Plotly figure
    """
    n_endmembers = endmembers.shape[0]
    n_samples = compositions.shape[0]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Endmember Compositions', 'Sample Decomposition'),
        specs=[[{"type": "bar"}, {"type": "heatmap"}]],
        horizontal_spacing=0.12
    )

    # Plot 1: Endmember Compositions
    x_positions = np.arange(len(element_names))
    bar_width = 0.8 / n_endmembers

    for i in range(n_endmembers):
        fig.add_trace(go.Bar(
            x=[e + (i - n_endmembers/2 + 0.5) * bar_width for e in x_positions],
            y=endmembers[i],
            name=f'Endmember {i+1}',
            marker_color=COLORS[i % len(COLORS)],
            width=bar_width
        ), row=1, col=1)

    # Plot 2: Coefficients Heatmap
    fig.add_trace(go.Heatmap(
        z=coefficients,
        x=[f'EM {i+1}' for i in range(n_endmembers)],
        y=[f'Sample {i+1}' for i in range(n_samples)],
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title='Fraction')
    ), row=1, col=2)

    # Update layout
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(len(element_names))),
        ticktext=element_names,
        row=1, col=1
    )
    fig.update_yaxes(title_text='Composition', row=1, col=1)

    fig.update_xaxes(title_text='Endmember', row=1, col=2)
    fig.update_yaxes(title_text='Sample', row=1, col=2)

    fig.update_layout(
        title=title,
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.25
        )
    )

    return fig


def create_xrd_publication_plot(xrd_data: Dict[int, List],
                                 selected_samples: Optional[List[int]] = None,
                                 title: str = "",
                                 offset_mode: bool = True,
                                 show_legend: bool = True,
                                 width: int = 800,
                                 height: int = 600) -> go.Figure:
    """
    Create publication-ready XRD pattern plot

    Args:
        xrd_data: Dictionary with sample IDs as keys
        selected_samples: List of sample IDs to plot
        title: Plot title
        offset_mode: Whether to stack patterns with offset
        show_legend: Whether to show legend
        width: Figure width
        height: Figure height

    Returns:
        Publication-styled Plotly figure
    """
    fig = create_xrd_plot(xrd_data, selected_samples, title, offset_mode, show_legend)
    fig = apply_publication_style(fig, width, height, show_grid=False)

    # Remove title for publication (often added in figure caption)
    if not title:
        fig.update_layout(title=None)

    return fig


def create_column_preview_plot(df_preview: 'pd.DataFrame',
                                theta_col: int,
                                intensity_col: int) -> go.Figure:
    """
    Create preview plot for column selection

    Args:
        df_preview: Preview dataframe
        theta_col: Selected 2theta column index
        intensity_col: Selected intensity column index

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    try:
        x_data = df_preview.iloc[:, theta_col].astype(float)
        y_data = df_preview.iloc[:, intensity_col].astype(float)

        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers',
            name='Preview',
            line=dict(color='#1f77b4', width=1.5),
            marker=dict(size=4)
        ))

        fig.update_layout(
            title='Data Preview',
            xaxis_title='2θ (selected column)',
            yaxis_title='Intensity (selected column)',
            template='plotly_white',
            height=300
        )
    except Exception:
        fig.update_layout(
            title='Invalid column selection',
            annotations=[dict(
                text='Cannot parse selected columns as numeric data',
                xref='paper', yref='paper',
                x=0.5, y=0.5, showarrow=False
            )]
        )

    return fig
