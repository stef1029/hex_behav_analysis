import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

def plot_control_histogram(session_group_results, 
                           condition="control", 
                           value_range=(0, 100), 
                           num_bins=20, 
                           figsize=(10, 6), 
                           colors=None, 
                           show_mean=True, 
                           show_median=False, 
                           show_stats=True, 
                           title=None, 
                           xlabel=None, 
                           ylabel=None, 
                           grid=True, 
                           save_path=None, 
                           dpi=300, 
                           specific_groups=None, 
                           normalize=False, 
                           cumulative=False, 
                           alpha=0.7,
                           edgecolor='black'):
    """
    Extract session values for each mouse across selected groups and plot a histogram.
    
    Args:
        session_group_results (dict): Results dictionary from analyze_session_groups function
        condition (str): Which condition to plot - "control", "test", or "both" (default: "control")
        value_range (tuple): Range of x-axis values (min, max) (default: (0, 100))
        num_bins (int): Number of histogram bins (default: 20)
        figsize (tuple): Figure size in inches (width, height) (default: (10, 6))
        colors (dict): Dictionary with keys 'control' and 'test' for bar colors 
                      (default: {'control': 'magenta', 'test': 'cyan'})
        show_mean (bool): Whether to show mean line (default: True)
        show_median (bool): Whether to show median line (default: False)
        show_stats (bool): Whether to print statistics (default: True)
        title (str): Plot title (default: auto-generated based on condition)
        xlabel (str): X-axis label (default: "Attempt Proportion (%)")
        ylabel (str): Y-axis label (default: "Number of Mice" or "Frequency" if normalized)
        grid (bool): Whether to show grid (default: True)
        save_path (str): Path to save figure (default: None, not saved)
        dpi (int): DPI for saved figure (default: 300)
        specific_groups (list): List of group names to include (default: None, all groups)
        normalize (bool): Whether to normalize histogram (default: False)
        cumulative (bool): Whether to show cumulative distribution (default: False)
        alpha (float): Transparency of histogram bars (default: 0.7)
        edgecolor (str): Color of histogram bar edges (default: 'black')
        
    Returns:
        tuple: (figure, axes, stats_dict) - The matplotlib figure, axes objects, and statistics dictionary
    """
    # Default colors if not provided
    if colors is None:
        colors = {'control': '#FF8A00', 'test': 'cyan'}
    
    # Compile values for the specified condition(s)
    condition_values = {'control': [], 'test': []}
    
    # Determine which groups to include
    groups_to_process = specific_groups if specific_groups else session_group_results.keys()
    
    # Extract values from each session group
    for group_name in groups_to_process:
        if group_name in session_group_results:
            group_data = session_group_results[group_name]
            
            # Extract control values if needed
            if condition in ["control", "both"] and group_data["control"]:
                condition_values['control'].extend(group_data["control"]["mice"].values())
            
            # Extract test values if needed
            if condition in ["test", "both"] and group_data["test"]:
                condition_values['test'].extend(group_data["test"]["mice"].values())
    
    # Convert proportion values to percentages (0-100 scale)
    for cond in condition_values:
        condition_values[cond] = [value * 100 for value in condition_values[cond]]
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bins
    bins = np.linspace(value_range[0], value_range[1], num_bins + 1)
    
    # Plot histograms for selected conditions
    if condition == "both":
        # Create slightly offset bins for control and test to see both
        width = (value_range[1] - value_range[0]) / num_bins
        
        # Plot control histogram
        n_ctrl, bins_ctrl, patches_ctrl = ax.hist(
            condition_values['control'], 
            bins=bins, 
            alpha=alpha, 
            color=colors['control'], 
            edgecolor=edgecolor,
            label='Control',
            density=normalize,
            cumulative=cumulative,
            histtype='bar',
            align='mid',
            rwidth=0.8
        )
        
        # Plot test histogram with transparency
        n_test, bins_test, patches_test = ax.hist(
            condition_values['test'], 
            bins=bins, 
            alpha=alpha * 0.8, 
            color=colors['test'], 
            edgecolor=edgecolor,
            label='Test',
            density=normalize,
            cumulative=cumulative,
            histtype='bar', 
            align='mid',
            rwidth=0.5
        )
        
        # Add legend
        ax.legend(loc='best')
        
    else:
        # Plot a single histogram
        n, bins, patches = ax.hist(
            condition_values[condition], 
            bins=bins, 
            color=colors[condition], 
            edgecolor=edgecolor, 
            alpha=alpha,
            density=normalize,
            cumulative=cumulative,
            histtype='bar',
            align='mid'
        )
    
    # Set default labels if not provided
    if xlabel is None:
        xlabel = 'Attempt Proportion (%)'
    if ylabel is None:
        ylabel = 'Frequency' if normalize else 'Number of Sessions'
    if title is None:
        if condition == "both":
            title = 'Distribution of Attempt Proportions (Control vs Test)'
        else:
            title = f'Distribution of {condition.capitalize()} Attempt Proportions'
    
    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=20)
    
    # Set the x-axis limits and ticks
    ax.set_xlim(value_range)
    ax.set_xticks(np.linspace(value_range[0], value_range[1], 11))
    # Set fontsize for tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=15)  # Controls x-axis tick label font size
    ax.tick_params(axis='y', labelsize=15)  # Controls y-axis tick label font size
    
    # Add percentage formatting to y-axis if normalized
    if normalize:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    
    # Add grid if requested
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a vertical line for the mean and/or median for each condition
    if condition == "both":
        conditions_to_plot = ['control', 'test']
    else:
        conditions_to_plot = [condition]
    
    # Dictionary to store statistics
    stats_dict = {}
    
    for cond in conditions_to_plot:
        if not condition_values[cond]:
            continue
            
        # Calculate statistics
        cond_values = condition_values[cond]
        cond_mean = np.mean(cond_values)
        cond_median = np.median(cond_values)
        cond_std = np.std(cond_values)
        cond_min = min(cond_values)
        cond_max = max(cond_values)
        cond_n = len(cond_values)
        
        # Store statistics in dictionary
        stats_dict[cond] = {
            'number_of_mice': cond_n,
            'mean': cond_mean,
            'median': cond_median,
            'std_dev': cond_std,
            'min': cond_min,
            'max': cond_max
        }
        
        # Add mean line if requested
        if show_mean:
            ax.axvline(cond_mean, color=colors[cond], linestyle='dashed', linewidth=2)
            ax.text(cond_mean+1, ax.get_ylim()[1]*(0.9 - 0.05*conditions_to_plot.index(cond)), 
                    f'{cond.capitalize()} Mean: {cond_mean:.2f}%', 
                    color=colors[cond], fontweight='bold')
        
        # Add median line if requested
        if show_median:
            ax.axvline(cond_median, color=colors[cond], linestyle='dotted', linewidth=2)
            offset = 0 if not show_mean else 0.05
            ax.text(cond_median+1, ax.get_ylim()[1]*(0.85 - (0.05*conditions_to_plot.index(cond) + offset)), 
                    f'{cond.capitalize()} Median: {cond_median:.2f}%', 
                    color=colors[cond], fontweight='bold')
    
    # Print statistics if requested
    if show_stats:
        print("\n===== STATISTICS =====")
        for cond in conditions_to_plot:
            if cond in stats_dict:
                stats = stats_dict[cond]
                print(f"\n{cond.upper()} CONDITION:")
                print(f"  Number of mice: {stats['number_of_mice']}")
                print(f"  Mean: {stats['mean']:.2f}%")
                print(f"  Median: {stats['median']:.2f}%")
                print(f"  Standard Deviation: {stats['std_dev']:.2f}%")
                print(f"  Range: {stats['min']:.2f}% - {stats['max']:.2f}%")
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Return the figure, axes, and statistics dictionary
    return fig, ax, stats_dict