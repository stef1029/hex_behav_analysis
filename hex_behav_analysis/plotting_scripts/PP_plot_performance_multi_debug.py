import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path
from datetime import datetime
import json

def generate_mouse_data():
    """
    Generate synthetic mouse performance data for demonstration.
    
    Returns
    -------
    dict
        Dictionary containing performance data for each mouse.
    """
    np.random.seed(42)  # For reproducibility
    
    # Define angle bins (in degrees)
    angles_deg = np.array([-165, -135, -105, -75, -45, -15, 15, 45, 75, 105, 135, 165])
    
    # Generate data for 5 mice with different performance patterns
    mice_data = {}
    mouse_names = ['mouse001', 'mouse002', 'mouse003', 'mouse004', 'mouse005']
    
    for i, mouse_id in enumerate(mouse_names):
        # Create realistic performance patterns with angle-dependent variation
        base_performance = 0.5 + 0.3 * np.sin(np.radians(angles_deg + i * 30))
        noise = np.random.normal(0, 0.1, len(angles_deg))
        performance = np.clip(base_performance + noise, 0, 1)
        
        mice_data[mouse_id] = {
            'performance': performance.tolist(),  # Convert to list for JSON serialisation
            'sessions': [f'session_{mouse_id}_{j}' for j in range(1, 4)],
            'angles': angles_deg.tolist()
        }
    
    return mice_data


def create_initial_figure(mice_data):
    """
    Create the initial Plotly figure with all traces.
    
    Parameters
    ----------
    mice_data : dict
        Dictionary containing mouse performance data.
        
    Returns
    -------
    plotly.graph_objects.Figure
        The initial figure with all mice and population average.
    """
    fig = go.Figure()
    
    # Get angle data
    angles_deg = np.array(mice_data[list(mice_data.keys())[0]]['angles'])
    angles_rad = np.radians(angles_deg % 360)
    angles_rad_closed = np.append(angles_rad, angles_rad[0])
    angles_deg_closed = np.degrees(angles_rad_closed)
    
    # Add individual mouse traces
    colours = px.colors.qualitative.Set1
    for i, (mouse_id, data) in enumerate(mice_data.items()):
        performance = np.array(data['performance'])
        performance_closed = np.append(performance, performance[0])
        sessions_str = ', '.join(data['sessions'])
        
        fig.add_trace(go.Scatterpolar(
            r=performance_closed,
            theta=angles_deg_closed,
            mode='lines+markers',
            name=mouse_id,
            line=dict(dash='dash', width=2, color=colours[i % len(colours)]),
            marker=dict(size=5),
            opacity=0.7,
            hovertemplate=f'<b>{mouse_id}</b><br>Sessions: {sessions_str}<br>Angle: %{{theta}}°<br>Performance: %{{r:.3f}}<extra></extra>',
            visible=True,
            meta={'mouse_id': mouse_id}  # Store mouse ID for reference
        ))
    
    # Calculate initial population statistics
    visible_performances = [np.array(data['performance']) for data in mice_data.values()]
    mean_performance = np.mean(visible_performances, axis=0)
    sem_performance = np.std(visible_performances, axis=0) / np.sqrt(len(visible_performances))
    
    mean_performance_closed = np.append(mean_performance, mean_performance[0])
    sem_performance_closed = np.append(sem_performance, sem_performance[0])
    
    # Add population average trace
    fig.add_trace(go.Scatterpolar(
        r=mean_performance_closed,
        theta=angles_deg_closed,
        mode='lines+markers',
        name='Population Average',
        line=dict(color='#00adf0', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Population Average</b><br>Angle: %{theta}°<br>Performance: %{r:.3f}<br>N=%{customdata}<extra></extra>',
        customdata=[len(visible_performances)] * len(mean_performance_closed),
        legendgroup='average',
        visible=True
    ))
    
    # Add SEM shading (upper bound)
    fig.add_trace(go.Scatterpolar(
        r=mean_performance_closed + sem_performance_closed,
        theta=angles_deg_closed,
        mode='lines',
        name='Upper SEM',
        line=dict(color='#00adf0', width=0),
        showlegend=False,
        hoverinfo='skip',
        legendgroup='average',
        visible=True
    ))
    
    # Add SEM shading (lower bound)
    fig.add_trace(go.Scatterpolar(
        r=mean_performance_closed - sem_performance_closed,
        theta=angles_deg_closed,
        mode='lines',
        name='Lower SEM',
        line=dict(color='#00adf0', width=0),
        fill='tonext',
        fillcolor='rgba(0, 173, 240, 0.3)',
        showlegend=False,
        hoverinfo='skip',
        legendgroup='average',
        visible=True
    ))
    
    # Configure layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=12)
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=list(range(-180, 181, 30))[:-1],
                ticktext=[f'{int(a)}°' for a in range(-180, 181, 30)][:-1],
                direction='clockwise',
                rotation=90,
                tickfont=dict(size=12)
            )
        ),
        title=dict(
            text='Dynamic Mouse Performance Visualisation',
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=True,
        legend=dict(
            x=1.1,
            y=0.5,
            font=dict(size=12),
            itemclick='toggle',  # Allow toggling
            itemdoubleclick='toggleothers'  # Double-click to isolate
        ),
        width=800,
        height=800
    )
    
    return fig


def create_dash_app(mice_data=None, port=8050):
    """
    Create and run a Dash application with dynamic average updating.
    
    Parameters
    ----------
    mice_data : dict, optional
        Dictionary containing mouse performance data. If None, generates synthetic data.
    port : int, default=8050
        Port number for the Dash server.
        
    Returns
    -------
    dash.Dash
        The Dash application instance.
    """
    # Generate data if not provided
    if mice_data is None:
        mice_data = generate_mouse_data()
    
    # Initialise Dash app
    app = dash.Dash(__name__)
    
    # Create initial figure
    initial_fig = create_initial_figure(mice_data)
    
    # Define app layout
    app.layout = html.Div([
        html.H1("Dynamic Mouse Performance Dashboard", 
                style={'textAlign': 'center', 'marginBottom': 30}),
        
        # Summary statistics panel
        html.Div(id='summary-stats', style={
            'textAlign': 'center',
            'marginBottom': 20,
            'fontSize': '16px'
        }),
        
        # Main graph
        dcc.Graph(
            id='performance-graph',
            figure=initial_fig,
            config={'displayModeBar': True, 'displaylogo': False}
        ),
        
        # Store component to hold mouse data
        dcc.Store(id='mouse-data-store', data=mice_data),
        
        # Instructions
        html.Div([
            html.H3("Instructions:"),
            html.Ul([
                html.Li("Click on mouse names in the legend to toggle visibility"),
                html.Li("Double-click a mouse name to show only that mouse"),
                html.Li("The population average updates automatically based on visible mice"),
                html.Li("Hover over data points for detailed information"),
                html.Li("Use the toolbar to zoom, pan, or save the plot")
            ])
        ], style={'marginTop': 30, 'marginLeft': 50})
    ])
    
    @app.callback(
        [Output('performance-graph', 'figure'),
         Output('summary-stats', 'children')],
        [Input('performance-graph', 'restyleData')],
        [State('performance-graph', 'figure'),
         State('mouse-data-store', 'data')]
    )
    def update_average_on_toggle(restyle_data, current_fig, mice_data):
        """
        Update the population average when mice are toggled on/off.
        
        Parameters
        ----------
        restyle_data : list
            Data about which traces were modified.
        current_fig : dict
            Current figure state.
        mice_data : dict
            Original mouse performance data.
            
        Returns
        -------
        tuple
            Updated figure and summary statistics.
        """
        if restyle_data is None:
            # Initial load - calculate stats for all mice
            n_visible = len(mice_data)
            all_performances = [np.array(data['performance']) for data in mice_data.values()]
            overall_mean = np.mean([np.mean(perf) for perf in all_performances])
            
            stats_text = f"Visible mice: {n_visible} | Overall mean performance: {overall_mean:.3f}"
            return current_fig, stats_text
        
        # Get angles for calculations
        angles_deg = np.array(mice_data[list(mice_data.keys())[0]]['angles'])
        angles_rad = np.radians(angles_deg % 360)
        angles_rad_closed = np.append(angles_rad, angles_rad[0])
        angles_deg_closed = np.degrees(angles_rad_closed)
        
        # Determine which mice are visible
        visible_mice = []
        for i, trace in enumerate(current_fig['data']):
            # Only check mouse traces (not average or SEM traces)
            if 'meta' in trace and 'mouse_id' in trace.get('meta', {}):
                # Check visibility - traces are visible if 'visible' is True or not set
                # They're hidden if 'visible' is 'legendonly'
                is_visible = trace.get('visible', True)
                if is_visible is True or (is_visible != 'legendonly' and is_visible != False):
                    visible_mice.append(trace['meta']['mouse_id'])
        
        # Calculate new average based on visible mice
        if len(visible_mice) > 0:
            visible_performances = [np.array(mice_data[mouse_id]['performance']) 
                                  for mouse_id in visible_mice]
            mean_performance = np.mean(visible_performances, axis=0)
            sem_performance = np.std(visible_performances, axis=0) / np.sqrt(len(visible_performances))
            
            mean_performance_closed = np.append(mean_performance, mean_performance[0])
            sem_performance_closed = np.append(sem_performance, sem_performance[0])
            
            # Update the average trace (find it by name)
            for i, trace in enumerate(current_fig['data']):
                if trace.get('name') == 'Population Average':
                    current_fig['data'][i]['r'] = mean_performance_closed.tolist()
                    current_fig['data'][i]['customdata'] = [len(visible_mice)] * len(mean_performance_closed)
                elif trace.get('name') == 'Upper SEM':
                    current_fig['data'][i]['r'] = (mean_performance_closed + sem_performance_closed).tolist()
                elif trace.get('name') == 'Lower SEM':
                    current_fig['data'][i]['r'] = (mean_performance_closed - sem_performance_closed).tolist()
            
            # Calculate summary statistics
            overall_mean = np.mean(mean_performance)
            stats_text = f"Visible mice: {len(visible_mice)} | Overall mean performance: {overall_mean:.3f}"
        else:
            # No mice visible - hide average traces
            for i, trace in enumerate(current_fig['data']):
                if trace.get('name') in ['Population Average', 'Upper SEM', 'Lower SEM']:
                    current_fig['data'][i]['visible'] = 'legendonly'
            
            stats_text = "No mice selected - select at least one mouse to see population average"
        
        return current_fig, stats_text
    
    return app


def run_dash_app(mice_data=None, port=8050, debug=True, jupyter_mode='external', host='127.0.0.1'):
    """
    Run the Dash application.
    
    Parameters
    ----------
    mice_data : dict, optional
        Dictionary containing mouse performance data. If None, generates synthetic data.
    port : int, default=8050
        Port number for the Dash server.
    debug : bool, default=True
        Whether to run in debug mode.
    jupyter_mode : str, default='external'
        How to display in Jupyter: 'external' (new tab), 'inline' (in notebook), 
        'jupyterlab' (for JupyterLab), or None (standard mode).
    host : str, default='127.0.0.1'
        Host to bind to. Use '0.0.0.0' to bind to all interfaces.
    """
    app = create_dash_app(mice_data, port)
    
    print("\n" + "=" * 50)
    print("STARTING DASH APPLICATION")
    print("=" * 50)
    print(f"Dashboard running on: http://{host}:{port}/")
    if host == '0.0.0.0':
        print(f"Also accessible at: http://<your-machine-ip>:{port}/")
    print("Press CTRL+C to stop the server")
    print("=" * 50 + "\n")
    
    # Run with appropriate mode
    if jupyter_mode:
        app.run(debug=debug, port=port, host=host, jupyter_mode=jupyter_mode)
    else:
        app.run(debug=debug, port=port, host=host)


def save_static_snapshot(mice_data=None, output_path=None, filename_prefix="dash_snapshot"):
    """
    Save a static snapshot of the current visualisation.
    
    Parameters
    ----------
    mice_data : dict, optional
        Dictionary containing mouse performance data.
    output_path : Path or str, optional
        Directory to save the snapshot.
    filename_prefix : str, default='dash_snapshot'
        Prefix for the saved file.
    """
    if mice_data is None:
        mice_data = generate_mouse_data()
    
    fig = create_initial_figure(mice_data)
    
    # Set up output directory
    if output_path is None:
        output_path = Path.cwd()
    else:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_filename = f"{timestamp}_{filename_prefix}.png"
    png_path = output_path / png_filename
    
    try:
        print(f"Saving static snapshot: {png_path}")
        fig.write_image(png_path)
        print("✅ Snapshot saved successfully")
    except Exception as e:
        if "kaleido" in str(e).lower():
            print("⚠️  PNG export requires kaleido: pip install kaleido")
        else:
            print(f"⚠️  PNG export failed: {e}")


def create_standalone_html(mice_data=None, output_path=None, filename="interactive_plot.html"):
    """
    Create a standalone HTML file with interactive plot (no server required).
    
    This creates a Plotly plot with interactive legend but without dynamic
    average recalculation.
    
    Parameters
    ----------
    mice_data : dict, optional
        Dictionary containing mouse performance data.
    output_path : Path or str, optional
        Directory to save the HTML file.
    filename : str, default='interactive_plot.html'
        Name of the output file.
        
    Returns
    -------
    Path
        Path to the created HTML file.
    """
    if mice_data is None:
        mice_data = generate_mouse_data()
    
    # Create figure with all traces
    fig = create_initial_figure(mice_data)
    
    # Add custom JavaScript for interactivity
    fig.update_layout(
        updatemenus=[
            {
                'buttons': [
                    {
                        'label': 'Show All',
                        'method': 'restyle',
                        'args': ['visible', [True] * len(fig.data)]
                    },
                    {
                        'label': 'Hide All Mice',
                        'method': 'restyle',
                        'args': ['visible', ['legendonly'] * (len(mice_data)) + [True] * 3]
                    }
                ],
                'direction': 'left',
                'showactive': False,
                'x': 0.1,
                'xanchor': 'left',
                'y': 1.15,
                'yanchor': 'top'
            }
        ]
    )
    
    # Set up output path
    if output_path is None:
        output_path = Path.cwd()
    else:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Save HTML
    html_path = output_path / filename
    fig.write_html(
        html_path,
        include_plotlyjs='cdn',  # Use CDN for smaller file size
        config={'displayModeBar': True, 'displaylogo': False}
    )
    
    print(f"\n✅ Interactive HTML saved to: {html_path}")
    print("This file can be opened directly in a browser without any server.")
    print("Features: Toggle mice visibility, zoom, pan, hover for details.")
    
    return html_path


# Example usage
if __name__ == "__main__":
    # Generate sample data
    sample_data = generate_mouse_data()
    
    # Run the Dash app
    run_dash_app(sample_data, port=8050, debug=True)