import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd
import re

def plot_performance_across_conditions(session_group_results, 
                                      figsize=(12, 8),
                                      colors=None,
                                      show_group_mean=True,
                                      show_overall_mean=False,
                                      show_individual_mice=True,
                                      title=None,
                                      xlabel=None,
                                      ylabel=None,
                                      y_range=None,
                                      grid=True,
                                      line_width=1.5,
                                      line_alpha=0.7,
                                      mean_line_width=3,
                                      save_path=None,
                                      dpi=300,
                                      show_legend=True,
                                      connect_points=True,
                                      marker='o',
                                      marker_size=6):
    """
    Plot mouse performance across different drug dosages with separate lines for Group 1 and Group 2.
    
    Args:
        session_group_results (dict): Results dictionary from analyze_session_groups function
        figsize (tuple): Figure size in inches (width, height) (default: (12, 8))
        colors (dict): Dictionary with keys for group names and values for colors
                      (default: {1: 'magenta', 2: 'cyan'})
        show_group_mean (bool): Whether to show mean line for each group (default: True)
        show_overall_mean (bool): Whether to show overall mean line across all mice (default: False)
        show_individual_mice (bool): Whether to show individual mouse lines (default: True)
        title (str): Plot title (default: "Performance Across Drug Dosages")
        xlabel (str): X-axis label (default: "Drug Dosage")
        ylabel (str): Y-axis label (default: "Attempt Proportion (%)")
        y_range (tuple): Range of y-axis values (min, max) (default: None, auto-scaled)
        grid (bool): Whether to show grid (default: True)
        line_width (float): Width of individual mouse lines (default: 1.5)
        line_alpha (float): Transparency of individual mouse lines (default: 0.7)
        mean_line_width (float): Width of mean lines (default: 3)
        save_path (str): Path to save figure (default: None, not saved)
        dpi (int): DPI for saved figure (default: 300)
        show_legend (bool): Whether to show legend (default: True)
        connect_points (bool): Whether to connect points with lines (default: True)
        marker (str): Marker style for data points (default: 'o')
        marker_size (int): Size of markers (default: 6)
        
    Returns:
        tuple: (figure, axes) - The matplotlib figure and axes objects
    """
    # Default colors if not provided
    if colors is None:
        colors = {1: 'magenta', 2: 'cyan'}
    
    # Set default labels and title if not provided
    if xlabel is None:
        xlabel = 'Drug Dosage'
    if ylabel is None:
        ylabel = 'Attempt Proportion (%)'
    if title is None:
        title = 'Performance Across Drug Dosages'
    
    # Extract data and organize by mouse, dosage, and group
    data = []
    for group_name, group_data in session_group_results.items():
        # Extract mouse group (1 or 2) and dosage from group name
        match = re.match(r'Group (\d+) (.+)', group_name)
        if match:
            group_num = int(match.group(1))
            dosage = match.group(2)
            
            # Extract test data
            if group_data["test"]:
                for mouse_id, proportion in group_data["test"]['mice'].items():
                    data.append({
                        'dosage': dosage,
                        'mouse_id': mouse_id,
                        'proportion': proportion * 100,  # Convert to percentage
                        'group': group_num
                    })
    
    # Create pandas DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Check if we have data
    if df.empty:
        print("No data found in the specified groups")
        return plt.figure(), plt.gca()
    
    # Define dosage order (saline is control, then increasing dosages)
    dosage_order = ['saline', '10ng', '100ng', '500ng', '1000ng']
    
    # Make sure all dosages in the data are in the order
    for dosage in df['dosage'].unique():
        if dosage not in dosage_order:
            dosage_order.append(dosage)
    
    # Create a categorical type with the desired order
    df['dosage_cat'] = pd.Categorical(df['dosage'], categories=dosage_order, ordered=True)
    
    # Sort the dataframe
    df = df.sort_values(['group', 'mouse_id', 'dosage_cat'])
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique mouse IDs for each group
    group1_mice = df[df['group'] == 1]['mouse_id'].unique()
    group2_mice = df[df['group'] == 2]['mouse_id'].unique()
    
    # Plot individual mouse lines for Group 1
    if show_individual_mice:
        for mouse_id in group1_mice:
            mouse_data = df[(df['mouse_id'] == mouse_id) & (df['group'] == 1)]
            if not mouse_data.empty:
                ax.plot(mouse_data['dosage_cat'], mouse_data['proportion'], 
                        marker=marker, markersize=marker_size,
                        color=colors.get(1, 'magenta'), 
                        linewidth=line_width, alpha=line_alpha,
                        linestyle='-')
        
        # Plot individual mouse lines for Group 2
        for mouse_id in group2_mice:
            mouse_data = df[(df['mouse_id'] == mouse_id) & (df['group'] == 2)]
            if not mouse_data.empty:
                ax.plot(mouse_data['dosage_cat'], mouse_data['proportion'], 
                        marker=marker, markersize=marker_size,
                        color=colors.get(2, 'cyan'), 
                        linewidth=line_width, alpha=line_alpha,
                        linestyle='-')
    
    # Plot group means
    if show_group_mean:
        # Group 1 mean
        group1_df = df[df['group'] == 1]
        if not group1_df.empty:
            group1_means = group1_df.groupby('dosage_cat')['proportion'].mean().reset_index()
            ax.plot(group1_means['dosage_cat'], group1_means['proportion'], 
                    marker=marker, markersize=marker_size+2,
                    color=colors.get(1, 'magenta'), 
                    linewidth=mean_line_width, linestyle='-',
                    label='Group 1 Mean')
        
        # Group 2 mean
        group2_df = df[df['group'] == 2]
        if not group2_df.empty:
            group2_means = group2_df.groupby('dosage_cat')['proportion'].mean().reset_index()
            ax.plot(group2_means['dosage_cat'], group2_means['proportion'], 
                    marker=marker, markersize=marker_size+2,
                    color=colors.get(2, 'cyan'), 
                    linewidth=mean_line_width, linestyle='-',
                    label='Group 2 Mean')
    
    # Plot overall mean
    if show_overall_mean:
        # Calculate mean for each dosage across all mice
        overall_means = df.groupby('dosage_cat')['proportion'].mean().reset_index()
        ax.plot(overall_means['dosage_cat'], overall_means['proportion'], 
                marker=marker, markersize=marker_size+4,
                color='black', linewidth=mean_line_width, 
                linestyle='--', label='Overall Mean')
    
    # Print statistics 
    print("\n===== PERFORMANCE STATISTICS =====")
    
    # Overall statistics
    print("\nOVERALL STATISTICS:")
    overall_mean = df['proportion'].mean()
    overall_median = df['proportion'].median()
    overall_std = df['proportion'].std()
    print(f"  Mean across all conditions: {overall_mean:.2f}%")
    print(f"  Median across all conditions: {overall_median:.2f}%")
    print(f"  Standard deviation: {overall_std:.2f}%")
    
    # Statistics by dosage
    print("\nDOSAGE STATISTICS:")
    for dosage in dosage_order:
        dosage_data = df[df['dosage'] == dosage]
        if not dosage_data.empty:
            dosage_mean = dosage_data['proportion'].mean()
            dosage_median = dosage_data['proportion'].median()
            dosage_std = dosage_data['proportion'].std()
            print(f"  {dosage}:")
            print(f"    Mean: {dosage_mean:.2f}%")
            print(f"    Median: {dosage_median:.2f}%")
            print(f"    Standard deviation: {dosage_std:.2f}%")
    
    # Statistics by group and dosage
    print("\nGROUP AND DOSAGE STATISTICS:")
    for group_num in [1, 2]:
        print(f"  Group {group_num}:")
        group_data = df[df['group'] == group_num]
        group_mean = group_data['proportion'].mean()
        group_median = group_data['proportion'].median()
        group_std = group_data['proportion'].std()
        print(f"    Overall Mean: {group_mean:.2f}%")
        print(f"    Overall Median: {group_median:.2f}%")
        print(f"    Overall Standard deviation: {group_std:.2f}%")
        
        # By dosage within group
        print(f"    By dosage:")
        for dosage in dosage_order:
            dosage_group_data = group_data[group_data['dosage'] == dosage]
            if not dosage_group_data.empty:
                dg_mean = dosage_group_data['proportion'].mean()
                dg_median = dosage_group_data['proportion'].median()
                dg_std = dosage_group_data['proportion'].std()
                print(f"      {dosage}:")
                print(f"        Mean: {dg_mean:.2f}%")
                print(f"        Median: {dg_median:.2f}%")
                print(f"        Standard deviation: {dg_std:.2f}%")
    
    # Set axis labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Set y-axis range if specified
    if y_range:
        ax.set_ylim(y_range)
    
    # Add grid if requested
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend if requested
    if show_legend:
        # Create custom legend
        legend_elements = []
        
        # Add group means
        if show_group_mean:
            legend_elements.append(
                Line2D([0], [0], color=colors.get(1, 'magenta'), 
                       linewidth=mean_line_width, marker=marker,
                       markersize=marker_size+2, label='Group 1 Mean')
            )
            legend_elements.append(
                Line2D([0], [0], color=colors.get(2, 'cyan'), 
                       linewidth=mean_line_width, marker=marker,
                       markersize=marker_size+2, label='Group 2 Mean')
            )
        
        # Add overall mean
        if show_overall_mean:
            legend_elements.append(
                Line2D([0], [0], color='black', linewidth=mean_line_width, 
                       linestyle='--', marker=marker, markersize=marker_size+4,
                       label='Overall Mean')
            )
        
        # Add individual mice
        if show_individual_mice:
            legend_elements.append(
                Line2D([0], [0], color=colors.get(1, 'magenta'), 
                       linewidth=line_width, alpha=line_alpha,
                       marker=marker, markersize=marker_size,
                       label='Group 1 Mouse')
            )
            legend_elements.append(
                Line2D([0], [0], color=colors.get(2, 'cyan'), 
                       linewidth=line_width, alpha=line_alpha,
                       marker=marker, markersize=marker_size,
                       label='Group 2 Mouse')
            )
        
        # Create legend
        ax.legend(handles=legend_elements, loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Return the figure and axes for further customization if needed
    return fig, ax