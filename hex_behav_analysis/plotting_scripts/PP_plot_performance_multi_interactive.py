import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path
from datetime import datetime


def plot_performance_by_angle_dash(sessions_input, 
                                   plot_titles=None,
                                   main_title='Mouse Performance by Angle',
                                   bin_mode='manual', 
                                   num_bins=12, 
                                   trials_per_bin=10, 
                                   cue_modes=['all_trials'],
                                   exclusion_mice=[],
                                   port=8050,
                                   host='127.0.0.1',
                                   likelihood_threshold=0.6):
    """
    Create an interactive Dash application for visualising mouse performance by angle.
    
    Parameters
    ----------
    sessions_input : list or dict
        Either:
        - A list of session objects (single condition)
        - A dict of condition_name: session_list pairs (multiple conditions)
    plot_titles : list or dict, optional
        Titles for each condition. If sessions_input is a dict, this should be a dict
        with matching keys. If None, uses condition names from sessions_input keys.
    main_title : str
        Main title for the application
    bin_mode : str
        Binning mode ('manual', 'rice', or 'tpb')
    num_bins : int
        Number of bins for manual mode
    trials_per_bin : int
        Target trials per bin for 'tpb' mode
    cue_modes : list
        List of cue modes to analyse (e.g. ['all_trials', 'visual_trials', 'audio_trials'])
    exclusion_mice : list
        List of mice to exclude from analysis
    port : int
        Port number for the Dash server
    host : str
        Host address (default '127.0.0.1' for localhost)
    likelihood_threshold : float
        Threshold for ear detection likelihood
        
    Returns
    -------
    dash.Dash
        The Dash application instance
    """
    
    # Define colours for different cue modes
    cue_mode_colours = {
        'all_trials': '#00adf0',      # Blue
        'visual_trials': '#ed008c',   # Pink
        'audio_trials': '#ff9700'     # Orange
    }
    
    # Convert single condition to dict format
    if isinstance(sessions_input, list):
        sessions_dict = {'Condition 1': sessions_input}
        if plot_titles is None:
            plot_titles = {'Condition 1': 'Condition 1'}
        elif isinstance(plot_titles, str):
            plot_titles = {'Condition 1': plot_titles}
    else:
        sessions_dict = sessions_input
        if plot_titles is None:
            plot_titles = {k: k for k in sessions_dict.keys()}
    
    def process_sessions(sessions, cue_mode='all_trials'):
        """
        Process session data to extract performance by angle for each mouse.
        
        Parameters
        ----------
        sessions : list
            List of session objects
        cue_mode : str
            Which trials to include
            
        Returns
        -------
        dict
            Dictionary with mouse IDs as keys and performance data as values
        """
        mice_data = {}
        all_trials = []
        
        # Collect trials by mouse
        for session in sessions:
            mouse_id = session.session_dict.get('mouse_id', 'unknown')
            if mouse_id in exclusion_mice:
                continue
                
            if mouse_id not in mice_data:
                mice_data[mouse_id] = {
                    'trials': [],
                    'sessions': []
                }
            
            # Add session identifier
            session_id = session.session_dict.get('session_id', 'unknown_session')
            if session_id not in mice_data[mouse_id]['sessions']:
                mice_data[mouse_id]['sessions'].append(session_id)
            
            # Process trials
            for trial in session.trials:
                # Skip catch trials
                if trial.get('catch', False):
                    continue
                
                # Skip trials without turn data
                if trial.get("turn_data") is None:
                    continue
                
                # Check ear detection likelihood
                if (trial["turn_data"].get("left_ear_likelihood", 1) < likelihood_threshold or
                    trial["turn_data"].get("right_ear_likelihood", 1) < likelihood_threshold):
                    continue
                
                # Filter based on cue mode
                if cue_mode == 'all_trials':
                    mice_data[mouse_id]['trials'].append(trial)
                    all_trials.append(trial)
                elif cue_mode == 'visual_trials' and 'audio' not in trial.get('correct_port', ''):
                    mice_data[mouse_id]['trials'].append(trial)
                    all_trials.append(trial)
                elif cue_mode == 'audio_trials' and 'audio' in trial.get('correct_port', ''):
                    mice_data[mouse_id]['trials'].append(trial)
                    all_trials.append(trial)
        
        # Remove mice with no trials for this cue mode
        mice_to_remove = []
        for mouse_id, mouse_info in mice_data.items():
            if len(mouse_info['trials']) == 0:
                mice_to_remove.append(mouse_id)
        
        for mouse_id in mice_to_remove:
            del mice_data[mouse_id]
        
        # If no mice have trials for this mode, return empty dict
        if not mice_data:
            return {}, 0
        
        # Determine number of bins
        n_trials = len(all_trials)
        if bin_mode == 'manual':
            n_bins = num_bins
        elif bin_mode == 'rice':
            n_bins = int(2 * n_trials ** (1/3))
        elif bin_mode == 'tpb':
            n_bins = int(n_trials / trials_per_bin)
        else:
            n_bins = 12  # Default
        
        # Create angle bins
        angle_range = 360
        bin_size = angle_range / n_bins
        bin_edges = np.arange(-180, 180 + bin_size, bin_size)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate performance for each mouse
        processed_data = {}
        for mouse_id, mouse_info in mice_data.items():
            # Initialise bins
            performance_bins = {i: [] for i in range(n_bins)}
            
            # Bin trials by angle
            for trial in mouse_info['trials']:
                angle = trial["turn_data"]["cue_presentation_angle"]
                
                # Determine if trial was correct
                is_correct = False
                if trial.get("next_sensor"):
                    is_correct = int(trial["correct_port"][-1]) == int(trial["next_sensor"]["sensor_touched"][-1])
                
                # Find appropriate bin
                bin_idx = np.digitize(angle, bin_edges) - 1
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                elif bin_idx < 0:
                    bin_idx = 0
                
                performance_bins[bin_idx].append(1 if is_correct else 0)
            
            # Calculate mean performance per bin
            performance = []
            for i in range(n_bins):
                if performance_bins[i]:
                    performance.append(np.mean(performance_bins[i]))
                else:
                    performance.append(0)  # No data for this bin
            
            processed_data[mouse_id] = {
                'performance': performance,
                'angles': bin_centres.tolist(),
                'sessions': mouse_info['sessions'],
                'n_trials': len(mouse_info['trials'])
            }
        
        return processed_data, n_bins
    
    # Process all conditions and cue modes
    all_data = {}
    for condition_name, sessions in sessions_dict.items():
        all_data[condition_name] = {}
        for cue_mode in cue_modes:
            mice_data, n_bins = process_sessions(sessions, cue_mode)
            all_data[condition_name][cue_mode] = mice_data
    
    # Create initial figure
    def create_figure(mice_data, visible_mice=None, cue_mode='all_trials', condition_name=''):
        """
        Create the Plotly figure with specified mice visible.
        
        Parameters
        ----------
        mice_data : dict
            Processed mouse performance data
        visible_mice : list, optional
            List of mouse IDs to show. If None, show all.
        cue_mode : str
            Current cue mode being displayed
        condition_name : str
            Name of the current condition
            
        Returns
        -------
        plotly.graph_objects.Figure
            The figure object
        """
        fig = go.Figure()
        
        # Handle empty data case
        if not mice_data:
            fig.update_layout(
                polar=dict(
                    bgcolor='#e8e8e8',
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickfont=dict(size=12),
                        tickmode='linear',
                        tick0=0,
                        dtick=0.2,
                        gridcolor='rgba(0,0,0,0.3)',
                        gridwidth=1.5,
                        showgrid=True,
                        showline=True,
                        linecolor='rgba(0,0,0,0.5)',
                        linewidth=2
                    ),
                    angularaxis=dict(
                        tickmode='array',
                        tickvals=list(range(-180, 181, 30)),
                        ticktext=[f'{int(a)}°' for a in range(-180, 181, 30)],
                        direction='counterclockwise',
                        rotation=90,
                        tickfont=dict(size=12),
                        gridcolor='rgba(0,0,0,0.25)',
                        gridwidth=1.2,
                        showgrid=True
                    )
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                title=dict(
                    text=f"{plot_titles.get(condition_name, condition_name)} - {cue_mode.replace('_', ' ').title()} (No Data)",
                    x=0.5,
                    font=dict(size=16)
                ),
                showlegend=False,
                width=800,
                height=800
            )
            return fig
        
        if visible_mice is None:
            visible_mice = list(mice_data.keys())
        
        # Colours for individual mice
        colours = px.colors.qualitative.Set1
        
        # Calculate average performance for each mouse and sort
        mouse_performance_list = []
        for mouse_id, data in mice_data.items():
            avg_performance = np.mean(data['performance'])
            mouse_performance_list.append((mouse_id, data, avg_performance))
        
        # Sort by average performance (highest first)
        mouse_performance_list.sort(key=lambda x: x[2], reverse=True)
        
        # Add individual mouse traces in sorted order
        for i, (mouse_id, data, avg_perf) in enumerate(mouse_performance_list):
            angles_deg = np.array(data['angles'])
            performance = np.array(data['performance'])
            
            # Close the polar plot
            angles_deg_closed = np.append(angles_deg, angles_deg[0])
            performance_closed = np.append(performance, performance[0])
            
            # Format session list for hover
            sessions_str = ', '.join(data['sessions'][:3])  # Show first 3 sessions
            if len(data['sessions']) > 3:
                extra_sessions = len(data['sessions']) - 3
                sessions_str += f' (+{extra_sessions} more)'
            
            fig.add_trace(go.Scatterpolar(
                r=performance_closed,
                theta=angles_deg_closed,
                mode='lines+markers',
                name=mouse_id,
                line=dict(dash='dash', width=2, color=colours[i % len(colours)]),
                marker=dict(size=5),
                opacity=0.7,
                hovertemplate=(f'<b>{mouse_id}</b><br>'
                             f'Sessions: {sessions_str}<br>'
                             f'Total {cue_mode.replace("_", " ")}: {data["n_trials"]}<br>'
                             'Angle: %{theta}°<br>'
                             'Performance: %{r:.3f}<extra></extra>'),
                visible=True if mouse_id in visible_mice else 'legendonly',
                meta={'mouse_id': mouse_id}
            ))
        
        # Calculate population average for visible mice
        if visible_mice:
            visible_performances = [np.array(mice_data[m]['performance']) 
                                   for m in visible_mice if m in mice_data]
            if visible_performances:
                mean_performance = np.mean(visible_performances, axis=0)
                sem_performance = np.std(visible_performances, axis=0) / np.sqrt(len(visible_performances))
                
                angles_deg = np.array(mice_data[list(mice_data.keys())[0]]['angles'])
                angles_deg_closed = np.append(angles_deg, angles_deg[0])
                mean_performance_closed = np.append(mean_performance, mean_performance[0])
                sem_performance_closed = np.append(sem_performance, sem_performance[0])
                
                # Use cue mode colour for population average
                avg_colour = cue_mode_colours.get(cue_mode, '#00adf0')
                
                # Population average trace
                fig.add_trace(go.Scatterpolar(
                    r=mean_performance_closed,
                    theta=angles_deg_closed,
                    mode='lines+markers',
                    name='Population Average',
                    line=dict(color=avg_colour, width=3),
                    marker=dict(size=8),
                    hovertemplate=('<b>Population Average</b><br>'
                                 'Angle: %{theta}°<br>'
                                 'Performance: %{r:.3f}<br>'
                                 'N=%{customdata}<extra></extra>'),
                    customdata=[len(visible_mice)] * len(mean_performance_closed),
                    legendgroup='average'
                ))
                
                # SEM shading
                fig.add_trace(go.Scatterpolar(
                    r=mean_performance_closed + sem_performance_closed,
                    theta=angles_deg_closed,
                    mode='lines',
                    name='Upper SEM',
                    line=dict(color=avg_colour, width=0),
                    showlegend=False,
                    hoverinfo='skip',
                    legendgroup='average'
                ))
                
                # Convert colour to rgba for fill
                if avg_colour.startswith('#'):
                    # Convert hex to rgba
                    hex_colour = avg_colour.lstrip('#')
                    r = int(hex_colour[0:2], 16)
                    g = int(hex_colour[2:4], 16)
                    b = int(hex_colour[4:6], 16)
                    fill_colour = f'rgba({r},{g},{b},0.3)'
                else:
                    # Assume it's already in rgb format, convert to rgba
                    fill_colour = avg_colour.replace('rgb', 'rgba').replace(')', ',0.3)')
                
                fig.add_trace(go.Scatterpolar(
                    r=mean_performance_closed - sem_performance_closed,
                    theta=angles_deg_closed,
                    mode='lines',
                    name='Lower SEM',
                    line=dict(color=avg_colour, width=0),
                    fill='tonext',
                    fillcolor=fill_colour,
                    showlegend=False,
                    hoverinfo='skip',
                    legendgroup='average'
                ))
        
        # Configure layout
        fig.update_layout(
            polar=dict(
                bgcolor='#e8e8e8',
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=12),
                    tickmode='linear',
                    tick0=0,
                    dtick=0.2,
                    gridcolor='rgba(0,0,0,0.3)',
                    gridwidth=1.5,
                    showgrid=True,
                    showline=True,
                    linecolor='rgba(0,0,0,0.5)',
                    linewidth=2
                ),
                angularaxis=dict(
                    tickmode='array',
                    tickvals=list(range(-180, 181, 30)),
                    ticktext=[f'{int(a)}°' for a in range(-180, 181, 30)],
                    direction='counterclockwise',
                    rotation=90,
                    tickfont=dict(size=12),
                    gridcolor='rgba(0,0,0,0.25)',
                    gridwidth=1.2,
                    showgrid=True
                )
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            title=dict(
                text=f"{plot_titles.get(condition_name, condition_name)} - {cue_mode.replace('_', ' ').title()}",
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=True,
            legend=dict(
                x=1.1,
                y=0.5,
                font=dict(size=12)
            ),
            width=800,
            height=800
        )
        
        return fig
    
    # Create Dash app
    app = dash.Dash(__name__)
    
    # Get initial values
    initial_condition = list(sessions_dict.keys())[0]
    initial_cue_mode = cue_modes[0]
    initial_fig = create_figure(
        all_data[initial_condition][initial_cue_mode], 
        cue_mode=initial_cue_mode,
        condition_name=initial_condition
    )
    
    # Define layout
    app.layout = html.Div([
        html.H1(main_title, style={'textAlign': 'center', 'marginBottom': 30}),
        
        # Condition and cue mode selectors
        html.Div([
            # Condition selector
            html.Div([
                html.Label('Select Condition:', style={'marginRight': 10, 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='condition-selector',
                    options=[{'label': plot_titles.get(k, k), 'value': k} for k in sessions_dict.keys()],
                    value=initial_condition,
                    style={'width': '300px'}
                )
            ], style={'display': 'inline-block', 'marginRight': '50px'}),
            
            # Cue mode selector
            html.Div([
                html.Label('Select Cue Mode:', style={'marginRight': 10, 'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='cue-mode-selector',
                    options=[{'label': mode.replace('_', ' ').title(), 'value': mode} for mode in cue_modes],
                    value=initial_cue_mode,
                    inline=True
                )
            ], style={'display': 'inline-block'})
        ], style={'textAlign': 'center', 'marginBottom': 20}),
        
        # Summary statistics
        html.Div(id='summary-stats', style={
            'textAlign': 'center',
            'marginBottom': 20,
            'fontSize': '16px'
        }),
        
        # Container for graph and session info
        html.Div([
            # Main graph
            html.Div([
                dcc.Graph(
                    id='performance-graph',
                    figure=initial_fig,
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            # Session information panel
            html.Div(id='session-info-container', style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
        ]),
        
        # Store for all data
        dcc.Store(id='all-data-store', data=all_data),
        dcc.Store(id='plot-titles-store', data=plot_titles),
        
        # Instructions
        html.Div([
            html.H3("Instructions:"),
            html.Ul([
                html.Li("Select different conditions using the dropdown menu"),
                html.Li("Select different cue modes using the radio buttons"),
                html.Li("Click on mouse names in the legend to toggle visibility"),
                html.Li("Double-click a mouse name to show only that mouse"),
                html.Li("The population average updates automatically based on visible mice"),
                html.Li("Hover over data points for detailed information"),
                html.Li("Session IDs can be copied from the panel on the right")
            ])
        ], style={'marginTop': 30, 'marginLeft': 50})
    ])
    
    @app.callback(
        [Output('performance-graph', 'figure'),
         Output('summary-stats', 'children'),
         Output('session-info-container', 'children')],
        [Input('condition-selector', 'value'),
         Input('cue-mode-selector', 'value'),
         Input('performance-graph', 'restyleData')],
        [State('performance-graph', 'figure'),
         State('all-data-store', 'data'),
         State('plot-titles-store', 'data')]
    )
    def update_display(selected_condition, selected_cue_mode, restyle_data, current_fig, all_data, plot_titles):
        """Update the display when condition, cue mode changes or mice are toggled."""
        
        # Get the current data
        mice_data = all_data[selected_condition][selected_cue_mode]
        
        # Convert figure to dictionary format for easier processing
        if hasattr(current_fig, 'to_dict'):
            current_fig_dict = current_fig.to_dict()
        else:
            current_fig_dict = current_fig
        
        # Check what triggered the update
        ctx = dash.ctx
        triggered_id = ctx.triggered_id if hasattr(ctx, 'triggered_id') else None
        
        # If condition or cue mode changed, create new figure
        if triggered_id in ['condition-selector', 'cue-mode-selector']:
            new_fig = create_figure(mice_data, cue_mode=selected_cue_mode, condition_name=selected_condition)
        else:
            # Otherwise, it's a toggle event - determine which mice are visible
            visible_mice = []
            for trace in current_fig_dict['data']:
                if isinstance(trace, dict) and 'meta' in trace and isinstance(trace['meta'], dict) and 'mouse_id' in trace['meta']:
                    is_visible = trace.get('visible', True)
                    if is_visible is True or (is_visible != 'legendonly' and is_visible != False):
                        visible_mice.append(trace['meta']['mouse_id'])
            
            # Create new figure with updated average
            new_fig = create_figure(mice_data, visible_mice, cue_mode=selected_cue_mode, condition_name=selected_condition)
            
            # Preserve visibility states from current figure
            for i, trace in enumerate(current_fig_dict['data']):
                if i < len(new_fig['data']) and 'visible' in trace:
                    new_fig['data'][i]['visible'] = trace['visible']
        
        # Calculate summary statistics
        visible_mice = []
        # Convert new_fig to dict if needed
        if hasattr(new_fig, 'to_dict'):
            new_fig_dict = new_fig.to_dict()
        else:
            new_fig_dict = new_fig
            
        for trace in new_fig_dict['data']:
            if isinstance(trace, dict) and 'meta' in trace and isinstance(trace['meta'], dict) and 'mouse_id' in trace['meta']:
                is_visible = trace.get('visible', True)
                if is_visible is True or (is_visible != 'legendonly' and is_visible != False):
                    visible_mice.append(trace['meta']['mouse_id'])
        
        if visible_mice and mice_data:  # Check that mice_data is not empty
            all_performances = []
            total_trials = 0
            for mouse_id in visible_mice:
                if mouse_id in mice_data:
                    all_performances.extend(mice_data[mouse_id]['performance'])
                    total_trials += mice_data[mouse_id]['n_trials']
            
            overall_mean = np.mean(all_performances) if all_performances else 0
            condition_label = plot_titles.get(selected_condition, selected_condition)
            stats_text = f"{condition_label} | {selected_cue_mode.replace('_', ' ').title()} | Visible mice: {len(visible_mice)} | Overall mean: {overall_mean:.3f} | Total trials: {total_trials}"
        elif not mice_data:
            condition_label = plot_titles.get(selected_condition, selected_condition)
            stats_text = f"{condition_label} | {selected_cue_mode.replace('_', ' ').title()} | No mice have trials for this cue mode"
        else:
            condition_label = plot_titles.get(selected_condition, selected_condition)
            stats_text = f"{condition_label} | {selected_cue_mode.replace('_', ' ').title()} | No mice selected"
        
        # Create session info panel
        if mice_data:  # Only create panel if there's data
            mouse_performance_sorted = []
            for mouse_id, data in mice_data.items():
                avg_performance = np.mean(data['performance'])
                mouse_performance_sorted.append((mouse_id, data, avg_performance))
            mouse_performance_sorted.sort(key=lambda x: x[2], reverse=True)
            
            session_info = html.Div([
                html.H3("Session Information", style={'marginBottom': 15}),
                html.Div([
                    html.Div([
                        html.B(f"{mouse_id} ({avg_perf:.1%}): ", style={'color': px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]}),
                        html.Span(', '.join(data['sessions']), style={'fontFamily': 'monospace', 'fontSize': '12px'}),
                        html.Br()
                    ], style={'marginBottom': 10})
                    for i, (mouse_id, data, avg_perf) in enumerate(mouse_performance_sorted)
                ], style={
                    'backgroundColor': '#f8f9fa',
                    'padding': '15px',
                    'borderRadius': '5px',
                    'maxHeight': '600px',
                    'overflowY': 'auto',
                    'fontSize': '14px',
                    'userSelect': 'text',
                    'cursor': 'text'
                })
            ])
        else:
            session_info = html.Div([
                html.H3("Session Information", style={'marginBottom': 15}),
                html.P(f"No mice have {selected_cue_mode.replace('_', ' ')} in their sessions.", 
                      style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
            ])
        
        return new_fig, stats_text, session_info
    
    print(f"\n{'=' * 50}")
    print("STARTING DASH APPLICATION")
    print(f"{'=' * 50}")
    print(f"Dashboard running on: http://{host}:{port}/")
    print("Press CTRL+C to stop the server")
    print(f"{'=' * 50}\n")
    
    # Run the app
    app.run(debug=True, port=port, host=host)
    
    return app