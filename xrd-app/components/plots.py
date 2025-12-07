"""
XRD Visualization Components
Plotly-based interactive plots for XRD analysis
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union
from matplotlib.colors import ListedColormap

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
