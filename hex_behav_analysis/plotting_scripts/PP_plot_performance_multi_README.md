# Angular Performance Plotter Documentation

## Overview

The Angular Performance Plotter is a sophisticated data visualisation and analysis tool designed for behavioural experiments involving angular responses. It analyses how subjects (typically mice) perform tasks based on cue presentation angles, providing comprehensive visualisations and statistical analyses of performance metrics, response biases, and signal detection measures.

## Table of Contents

1. [Installation & Requirements](#installation--requirements)
2. [Quick Start](#quick-start)
3. [Function Parameters](#function-parameters)
4. [Plot Types Explained](#plot-types-explained)
5. [Statistical Analyses](#statistical-analyses)
6. [Usage Examples](#usage-examples)
7. [Interpreting Results](#interpreting-results)
8. [Troubleshooting](#troubleshooting)

## Installation & Requirements

### Dependencies
```python
matplotlib >= 3.0.0
numpy >= 1.19.0
scipy >= 1.5.0
pathlib (standard library)
datetime (standard library)
dataclasses (Python 3.7+)
```

### Required Custom Modules
```python
from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
from hex_behav_analysis.utils.Session_nwb import Session
```

## Quick Start

### Basic Usage
```python
from hex_behav_analysis.plotting_scripts.PP_plot_performance_multi import plot_performance_by_angle

# Single dataset plotting
plot_performance_by_angle(
    sessions,  # List of session objects
    plot_title='Mouse Performance by Angle',
    plot_type='hit_rate',
    plot_mode='radial'
)

# Multiple group comparison
sessions_dict = {
    'Control': control_sessions,
    'Test': test_sessions
}
plot_performance_by_angle(
    sessions_dict,
    plot_title='Control vs Test Performance',
    plot_type='bias_corrected',
    show_circular_stats=True
)
```

## Function Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sessions_input` | list or dict | Required | List of sessions or dictionary of session lists for group comparisons |
| `plot_title` | str | 'title' | Main title for the plot |
| `x_title` | str | '' | X-axis label (linear plots only) |
| `y_title` | str | '' | Y-axis label |

### Binning Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bin_mode` | str | 'manual' | Binning strategy: 'manual', 'rice', or 'tpb' |
| `num_bins` | int | 12 | Number of bins for 'manual' mode |
| `trials_per_bin` | int | 10 | Target trials per bin for 'tpb' mode |

**Binning Modes Explained:**
- **'manual'**: Fixed number of bins specified by `num_bins`
- **'rice'**: Rice's rule: `bins = 2 * n^(1/3)` where n is number of trials
- **'tpb'**: Trials per bin: `bins = total_trials / trials_per_bin`

### Visualisation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plot_mode` | str | 'radial' | Plot style: 'radial' or 'linear_comparison' |
| `plot_type` | str | 'hit_rate' | Type of data to plot (see Plot Types section) |
| `cue_modes` | list | ['visual_trials'] | Trial types to analyse |
| `error_bars` | str | 'SEM' | Error bar type (currently only 'SEM' supported) |
| `plot_individual_mice` | bool | False | Show individual mouse data |

### Data Filtering Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exclusion_mice` | list | [] | Mouse IDs to exclude from analysis |
| `likelihood_threshold` | float | 0.6 | Minimum ear detection confidence |
| `timeout_handling` | str/None | None | How to handle timeout trials |
| `min_trial_duration` | float/None | None | Minimum trial duration in seconds |

**Timeout Handling Options:**
- `None`: Include timeouts as incorrect trials (default)
- `'exclude'`: Exclude from hit rate but keep in trial count
- `'exclude_total'`: Remove from all calculations entirely

### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | Path | None | Directory to save plots |
| `plot_save_name` | str | 'untitled_plot' | Base filename for saved plots |
| `draft` | bool | True | If True, adds timestamp to filename |

### Statistical Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_circular_stats` | bool | True | Calculate and display circular statistics |
| `plot_circular_means` | bool | True | Plot mean vectors on radial plots |

## Plot Types Explained

### 1. Hit Rate (`plot_type='hit_rate'`)
**What it shows:** Raw performance accuracy at each angle bin.

**Calculation:** 
```
hit_rate = correct_trials / total_trials per angle bin
```

**Interpretation:**
- Values range from 0 to 1 (0-100% accuracy)
- Higher values indicate better performance at that angle
- Useful for identifying angular preferences or task difficulty by location

### 2. Bias (`plot_type='bias'`)
**What it shows:** Distribution of all responses across angles, regardless of correctness.

**Calculation:**
```
bias = touches_at_angle / total_touches across all angles
```

**Interpretation:**
- Shows where subjects tend to respond, independent of task demands
- Values sum to 1 across all angles
- High values indicate response preference for that location
- Reveals motor biases or spatial preferences

### 3. Bias Corrected (`plot_type='bias_corrected'`)
**What it shows:** Performance adjusted for response bias.

**Calculation:**
```
bias_corrected = hit_rate / (bias + ε)
```
Where ε is a small constant to avoid division by zero.

**Interpretation:**
- Removes the influence of spatial preferences from performance
- Values > 1 indicate performance better than expected given bias
- Values < 1 indicate performance worse than expected
- Useful for distinguishing true perceptual ability from motor preferences

### 4. Bias Incorrect (`plot_type='bias_incorrect'`)
**What it shows:** Distribution of responses from incorrect trials only.

**Calculation:**
```
bias_incorrect = incorrect_touches_at_angle / total_incorrect_touches
```

**Interpretation:**
- Reveals systematic error patterns
- Shows where subjects go when they make mistakes
- Can indicate confusion between similar angles
- Useful for understanding failure modes

### 5. D-Prime (`plot_type='dprime'`)
**What it shows:** Signal detection sensitivity measure.

**Calculation:**
Based on signal detection theory:
```
d' = Z(hit_rate) - Z(false_alarm_rate)
```
Where Z is the inverse normal cumulative distribution function.

**Interpretation:**
- Measures ability to discriminate signal from noise
- Higher values indicate better discrimination
- d' = 0: No discrimination ability
- d' = 1: Moderate discrimination
- d' > 2: Good discrimination
- Can be negative if performance is below chance
- Independent of response bias

## Statistical Analyses

### 1. Circular Statistics

The plotter calculates several circular statistics for angular data:

#### Circular Mean
- **What it is:** The average direction weighted by performance
- **Calculation:** Uses vector addition of angles weighted by performance values
- **Interpretation:** Shows the "centre of mass" of performance across angles

#### Resultant Vector Length (R)
- **What it is:** Measure of concentration around the circular mean
- **Range:** 0 to 1
- **Interpretation:**
  - R ≈ 0: Uniform distribution (no preferred direction)
  - R ≈ 1: Highly concentrated around mean direction

#### Rayleigh Test
- **What it tests:** Whether data is uniformly distributed around the circle
- **Null hypothesis:** Data is uniformly distributed
- **Interpretation:**
  - p < 0.05: Significant directional preference
  - p ≥ 0.05: No significant directional preference

### 2. Watson-Williams Test

Used for comparing circular means between groups.

- **What it tests:** Whether two or more groups have different mean directions
- **Assumptions:** Similar to ANOVA but for circular data
- **Interpretation:**
  - p < 0.05: Groups have significantly different mean angles
  - F-statistic: Larger values indicate greater differences

### 3. Pairwise Comparisons

When multiple groups are provided, the plotter automatically performs all pairwise Watson-Williams tests and reports:
- Individual group means
- F-statistics for each comparison
- Corrected p-values
- Summary of significant pairs

## Usage Examples

### Example 1: Basic Single Group Analysis
```python
# Analyse hit rate with radial plot
plot_performance_by_angle(
    sessions,
    plot_title='Mouse Performance - Hit Rate',
    plot_type='hit_rate',
    plot_mode='radial',
    num_bins=12,
    show_circular_stats=True,
    output_path=Path('./results'),
    plot_save_name='hit_rate_analysis'
)
```

### Example 2: Group Comparison with Bias Correction
```python
# Compare control vs treatment with bias-corrected performance
sessions_dict = {
    'Control': control_sessions,
    'Treatment': treatment_sessions
}

plot_performance_by_angle(
    sessions_dict,
    plot_title='Bias-Corrected Performance Comparison',
    plot_type='bias_corrected',
    plot_mode='radial',
    show_circular_stats=True,
    plot_individual_mice=True,
    output_path=Path('./results'),
    plot_save_name='group_comparison'
)
```

### Example 3: Multi-Modal Analysis
```python
# Analyse visual vs audio trials separately
plot_performance_by_angle(
    sessions,
    plot_title='Performance by Modality',
    plot_type='hit_rate',
    cue_modes=['visual_trials', 'audio_trials'],
    plot_mode='linear_comparison',
    error_bars='SEM',
    output_path=Path('./results')
)
```

### Example 4: Signal Detection Analysis with Filtering
```python
# D-prime analysis with quality filters
plot_performance_by_angle(
    sessions,
    plot_title='Signal Detection Sensitivity',
    plot_type='dprime',
    plot_mode='radial',
    likelihood_threshold=0.8,  # Stricter tracking quality
    min_trial_duration=0.5,    # Exclude very short trials
    timeout_handling='exclude', # Don't count timeouts as errors
    exclusion_mice=['mouse_01', 'mouse_02'],  # Exclude specific subjects
    output_path=Path('./results'),
    draft=False  # Final version without timestamp
)
```

### Example 5: Three-Group Comparison
```python
# Compare multiple experimental conditions
sessions_dict = {
    'Control': control_sessions,
    'Test1': test1_sessions,
    'Test2': test2_sessions
}

plot_performance_by_angle(
    sessions_dict,
    plot_title='Multi-Condition Angular Performance',
    plot_type='bias_corrected',
    plot_mode='radial',
    show_circular_stats=True,
    plot_circular_means=True,
    cue_modes=['visual_trials'],  # Must be single mode for multi-group
    output_path=Path('./results')
)
```

## Interpreting Results

### Radial Plots
- **Distance from centre:** Performance metric value
- **Angle:** Spatial location of cue/response
- **Error bands:** Standard error across subjects
- **Arrows (if shown):** Circular mean direction and concentration

### Statistical Output
The function prints comprehensive statistics including:
1. **Exclusion Summary:** Number of trials excluded and reasons
2. **Individual Mouse Data:** Per-subject circular statistics
3. **Group Statistics:** Mean direction, concentration, and uniformity tests
4. **Between-Group Comparisons:** All pairwise comparisons with p-values

### Common Patterns
- **Uniform performance:** Flat circular profile suggests no angular preference
- **Peaked performance:** Better performance at specific angles
- **Bimodal patterns:** May indicate categorical processing
- **Shifted peaks in bias vs hit rate:** Reveals dissociation between preference and ability

## Troubleshooting

### Common Issues

**Issue:** "When providing a sessions dictionary, cue_modes must contain exactly one trial type"
- **Solution:** Use only one cue mode when comparing multiple groups

**Issue:** Low trial counts in certain bins
- **Solution:** Reduce number of bins or use 'tpb' mode with appropriate trials_per_bin

**Issue:** Circular statistics showing NaN
- **Solution:** Check for sufficient trials and non-zero performance values

**Issue:** Plot appears empty
- **Solution:** Verify session data format and check exclusion criteria aren't too strict

### Data Requirements

Sessions must contain trials with:
- `mouse_id`: Subject identifier
- `catch`: Boolean for catch trials
- `cue_start`: Trial start timestamp
- `next_sensor`: Response information
- `correct_port`: Expected response
- `turn_data`: Dictionary containing:
  - `cue_presentation_angle`
  - `port_touched_angle`
  - `left_ear_likelihood`
  - `right_ear_likelihood`

## Best Practices

1. **Start with hit rate** to understand basic performance patterns
2. **Compare bias and hit rate** to identify motor vs perceptual effects  
3. **Use bias correction** when spatial preferences are strong
4. **Apply d-prime** for tasks requiring discrimination
5. **Set appropriate filters** based on data quality
6. **Save both PNG and SVG** formats for publication
7. **Use draft mode** during exploration, disable for final figures
8. **Check circular statistics** to validate angular preferences
9. **Compare multiple groups** with identical parameters for fair comparison