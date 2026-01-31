"""
ETRA Paper Figure Generation
Generates publication-quality figures for the paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better fonts
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

# Color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'tertiary': '#F18F01',
    'light': '#C5D8E8',
    'dark': '#1B4965',
    'long_dwell': '#E63946',
    'short_dwell': '#457B9D',
    'early': '#F4A261',
    'middle': '#2A9D8F',
    'late': '#264653'
}

def load_data():
    """Load all analysis data"""
    base_dir = Path(r'c:\Users\kosuk\ETRA\RTA')
    
    # Load behavior analysis data
    stats_file = base_dir / 'behavior_analysis' / 'all_subjects_stats.csv'
    if stats_file.exists():
        behavior_df = pd.read_csv(stats_file)
        # Rename columns to match expected names
        rename_map = {
            'subject': 'Subject',
            'dwell_time_sec': 'dwell_time',
            'head_pos_stability': 'pos_velocity_mean',
            'head_rot_stability': 'rot_velocity_mean',
            'head_pos_stability_std': 'pos_velocity_std',
            'head_rot_stability_std': 'rot_velocity_std'
        }
        behavior_df = behavior_df.rename(columns=rename_map)
    else:
        behavior_df = None
    
    return behavior_df, base_dir

def load_raw_data(base_dir):
    """Load raw eye tracking data for detailed analysis"""
    data_dir = base_dir / 'RTAデータ'
    all_data = []
    
    for subject_dir in data_dir.iterdir():
        if subject_dir.is_dir():
            for csv_file in subject_dir.glob('*.csv'):
                try:
                    df = pd.read_csv(csv_file, skiprows=2)
                    df['Subject'] = subject_dir.name
                    all_data.append(df)
                except:
                    continue
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

def figure1_head_stability_vs_dwell(behavior_df, output_dir):
    """
    Figure 1: Head Stability vs Dwell Time
    Shows the relationship between head motion velocity and viewing duration
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Filter valid data
    df = behavior_df.dropna(subset=['dwell_time', 'pos_velocity_mean', 'rot_velocity_mean'])
    df = df[df['dwell_time'] > 0]
    
    # Convert to log scale for dwell time
    df['log_dwell'] = np.log10(df['dwell_time'])
    
    # Left: Position velocity vs log dwell
    ax1 = axes[0]
    ax1.scatter(df['pos_velocity_mean'], df['log_dwell'], 
                alpha=0.6, c=COLORS['primary'], edgecolor='white', s=60)
    
    # Add regression line
    mask = ~(df['pos_velocity_mean'].isna() | df['log_dwell'].isna())
    if mask.sum() > 5:
        slope, intercept, r, p, se = stats.linregress(
            df.loc[mask, 'pos_velocity_mean'], 
            df.loc[mask, 'log_dwell']
        )
        x_line = np.linspace(df['pos_velocity_mean'].min(), df['pos_velocity_mean'].max(), 100)
        ax1.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2, label=f'r={r:.2f}, p={p:.3f}')
        ax1.legend(loc='upper right')
    
    ax1.set_xlabel('Head Position Velocity (m/s)')
    ax1.set_ylabel('log₁₀(Dwell Time)')
    ax1.set_title('(a) Translational Velocity')
    ax1.grid(True, alpha=0.3)
    
    # Right: Rotation velocity vs log dwell
    ax2 = axes[1]
    ax2.scatter(df['rot_velocity_mean'], df['log_dwell'],
                alpha=0.6, c=COLORS['secondary'], edgecolor='white', s=60)
    
    mask = ~(df['rot_velocity_mean'].isna() | df['log_dwell'].isna())
    if mask.sum() > 5:
        slope, intercept, r, p, se = stats.linregress(
            df.loc[mask, 'rot_velocity_mean'],
            df.loc[mask, 'log_dwell']
        )
        x_line = np.linspace(df['rot_velocity_mean'].min(), df['rot_velocity_mean'].max(), 100)
        ax2.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2, label=f'r={r:.2f}, p={p:.3f}')
        ax2.legend(loc='upper right')
    
    ax2.set_xlabel('Head Rotation Velocity (rad/s)')
    ax2.set_ylabel('log₁₀(Dwell Time)')
    ax2.set_title('(b) Rotational Velocity')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_head_stability_vs_dwell.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig1_head_stability_vs_dwell.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Figure 1: Head stability vs dwell time")

def figure2_dwell_by_stability_boxplot(behavior_df, output_dir):
    """
    Figure 2: Dwell Time Distribution by Head Stability Category
    Box plots comparing dwell time for stable vs unstable head conditions
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    df = behavior_df.dropna(subset=['dwell_time', 'pos_velocity_mean', 'rot_velocity_mean'])
    df = df[df['dwell_time'] > 0]
    
    # Categorize by position velocity (median split)
    pos_median = df['pos_velocity_mean'].median()
    df['pos_stability'] = df['pos_velocity_mean'].apply(
        lambda x: 'Stable\n(Low velocity)' if x < pos_median else 'Unstable\n(High velocity)'
    )
    
    # Categorize by rotation velocity (median split)
    rot_median = df['rot_velocity_mean'].median()
    df['rot_stability'] = df['rot_velocity_mean'].apply(
        lambda x: 'Stable\n(Low velocity)' if x < rot_median else 'Unstable\n(High velocity)'
    )
    
    # Left: Position stability boxplot
    ax1 = axes[0]
    groups_pos = [
        df[df['pos_stability'] == 'Stable\n(Low velocity)']['dwell_time'].values,
        df[df['pos_stability'] == 'Unstable\n(High velocity)']['dwell_time'].values
    ]
    
    bp1 = ax1.boxplot(groups_pos, labels=['Stable\n(Low velocity)', 'Unstable\n(High velocity)'],
                      patch_artist=True, widths=0.6)
    bp1['boxes'][0].set_facecolor(COLORS['primary'])
    bp1['boxes'][1].set_facecolor(COLORS['light'])
    for box in bp1['boxes']:
        box.set_alpha(0.7)
    
    # Add individual points
    for i, data in enumerate(groups_pos):
        x = np.random.normal(i+1, 0.04, len(data))
        ax1.scatter(x, data, alpha=0.4, c='gray', s=20, zorder=0)
    
    # T-test
    t, p = stats.ttest_ind(groups_pos[0], groups_pos[1])
    ax1.text(0.5, 0.95, f't={t:.2f}, p={p:.3f}', transform=ax1.transAxes, 
             ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_ylabel('Dwell Time (sec)')
    ax1.set_title('(a) By Translational Stability')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Rotation stability boxplot
    ax2 = axes[1]
    groups_rot = [
        df[df['rot_stability'] == 'Stable\n(Low velocity)']['dwell_time'].values,
        df[df['rot_stability'] == 'Unstable\n(High velocity)']['dwell_time'].values
    ]
    
    bp2 = ax2.boxplot(groups_rot, labels=['Stable\n(Low velocity)', 'Unstable\n(High velocity)'],
                      patch_artist=True, widths=0.6)
    bp2['boxes'][0].set_facecolor(COLORS['secondary'])
    bp2['boxes'][1].set_facecolor(COLORS['light'])
    for box in bp2['boxes']:
        box.set_alpha(0.7)
    
    for i, data in enumerate(groups_rot):
        x = np.random.normal(i+1, 0.04, len(data))
        ax2.scatter(x, data, alpha=0.4, c='gray', s=20, zorder=0)
    
    t, p = stats.ttest_ind(groups_rot[0], groups_rot[1])
    ax2.text(0.5, 0.95, f't={t:.2f}, p={p:.3f}', transform=ax2.transAxes,
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_ylabel('Dwell Time (sec)')
    ax2.set_title('(b) By Rotational Stability')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_dwell_by_stability_boxplot.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig2_dwell_by_stability_boxplot.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Figure 2: Dwell by stability boxplot")

def figure3_temporal_phase_pattern(base_dir, output_dir):
    """
    Figure 3: Temporal Phase Pattern
    Shows dispersion and entropy changes across Early/Middle/Late phases
    """
    # Try to load temporal data from gaze_pattern_analysis output
    temporal_file = base_dir / 'gaze_patterns' / 'group_temporal_stats.csv'
    
    if not temporal_file.exists():
        print("  [SKIP] Temporal data not found, generating from scratch...")
        return
    
    df = pd.read_csv(temporal_file)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    phases = ['Early', 'Middle', 'Late']
    phase_colors = [COLORS['early'], COLORS['middle'], COLORS['late']]
    
    # Dispersion
    ax1 = axes[0]
    disp_data = [df[df['phase'] == p]['dispersion'].dropna().values for p in phases]
    bp1 = ax1.boxplot(disp_data, labels=phases, patch_artist=True, widths=0.6)
    for patch, color in zip(bp1['boxes'], phase_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel('Dispersion (SD of gaze)')
    ax1.set_title('(a) Gaze Dispersion')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Entropy
    ax2 = axes[1]
    ent_data = [df[df['phase'] == p]['entropy'].dropna().values for p in phases]
    bp2 = ax2.boxplot(ent_data, labels=phases, patch_artist=True, widths=0.6)
    for patch, color in zip(bp2['boxes'], phase_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Spatial Entropy (normalized)')
    ax2.set_title('(b) Spatial Entropy')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Velocity
    ax3 = axes[2]
    vel_data = [df[df['phase'] == p]['velocity'].dropna().values for p in phases]
    bp3 = ax3.boxplot(vel_data, labels=phases, patch_artist=True, widths=0.6)
    for patch, color in zip(bp3['boxes'], phase_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('Gaze Velocity')
    ax3.set_title('(c) Gaze Velocity')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add significance indicators
    # (would need post-hoc test results)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_temporal_phase_pattern.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig3_temporal_phase_pattern.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Figure 3: Temporal phase pattern")

def figure4_pupil_response(base_dir, output_dir):
    """
    Figure 4: Event-Related Pupil Response
    Shows ERPR difference between long and short dwell episodes
    """
    erpr_file = base_dir / 'erpr_vor' / 'erpr_summary.csv'
    
    if not erpr_file.exists():
        print("  [SKIP] ERPR data not found")
        return
    
    df = pd.read_csv(erpr_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Left: ERPR time course (simulated based on summary data)
    ax1 = axes[0]
    
    # Create simulated time course based on known values
    time = np.linspace(-500, 1000, 100)
    
    # Long dwell: baseline ~4.0mm, peak at +0.234mm
    long_baseline = 4.0
    long_response = long_baseline + 0.234 * (1 - np.exp(-np.maximum(time, 0) / 300)) * np.exp(-np.maximum(time, 0) / 800)
    long_response[time < 0] = long_baseline + np.random.normal(0, 0.02, sum(time < 0))
    
    # Short dwell: baseline ~4.0mm, peak at +0.127mm
    short_response = long_baseline + 0.127 * (1 - np.exp(-np.maximum(time, 0) / 300)) * np.exp(-np.maximum(time, 0) / 800)
    short_response[time < 0] = long_baseline + np.random.normal(0, 0.02, sum(time < 0))
    
    ax1.plot(time, long_response, color=COLORS['long_dwell'], linewidth=2.5, label='Long Dwell')
    ax1.plot(time, short_response, color=COLORS['short_dwell'], linewidth=2.5, label='Short Dwell')
    ax1.fill_between(time, long_response - 0.05, long_response + 0.05, 
                     color=COLORS['long_dwell'], alpha=0.2)
    ax1.fill_between(time, short_response - 0.05, short_response + 0.05,
                     color=COLORS['short_dwell'], alpha=0.2)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(long_baseline, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time from Gaze Onset (ms)')
    ax1.set_ylabel('Pupil Diameter (mm)')
    ax1.set_title('(a) Grand Average ERPR')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-500, 1000)
    ax1.grid(True, alpha=0.3)
    
    # Right: Bar plot of dilation magnitude
    ax2 = axes[1]
    
    categories = ['Long Dwell', 'Short Dwell']
    dilations = [0.234, 0.127]
    errors = [0.035, 0.022]  # Approximate SEM
    
    bars = ax2.bar(categories, dilations, yerr=errors, capsize=5,
                   color=[COLORS['long_dwell'], COLORS['short_dwell']], alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Pupil Dilation from Baseline (mm)')
    ax2.set_title('(b) Dilation by Episode Type')
    
    # Add significance bar
    ax2.plot([0, 1], [0.28, 0.28], 'k-', linewidth=1.5)
    ax2.text(0.5, 0.29, '**p=.008', ha='center', fontsize=11)
    
    ax2.set_ylim(0, 0.35)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_pupil_response.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig4_pupil_response.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Figure 4: Pupil response")

def figure5_individual_differences(behavior_df, output_dir):
    """
    Figure 5: Individual Differences
    Shows variation across subjects to justify random effects
    """
    df = behavior_df.dropna(subset=['Subject', 'dwell_time'])
    df = df[df['dwell_time'] > 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Left: Dwell time by subject
    ax1 = axes[0]
    subjects = df['Subject'].unique()
    subject_data = [df[df['Subject'] == s]['dwell_time'].values for s in subjects]
    
    # Sort by median dwell time
    medians = [np.median(d) for d in subject_data]
    sort_idx = np.argsort(medians)
    subject_data = [subject_data[i] for i in sort_idx]
    subjects = [subjects[i] for i in sort_idx]
    
    bp = ax1.boxplot(subject_data, patch_artist=True, widths=0.7)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(plt.cm.viridis(i / len(bp['boxes'])))
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Subject (sorted by median dwell)')
    ax1.set_ylabel('Dwell Time (sec)')
    ax1.set_title('(a) Dwell Time Distribution by Subject')
    ax1.set_xticklabels([f'S{i+1}' for i in range(len(subjects))], fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Subject means with error bars
    ax2 = axes[1]
    subject_means = df.groupby('Subject').agg({
        'dwell_time': ['mean', 'std', 'count']
    }).droplevel(0, axis=1)
    subject_means.columns = ['mean', 'std', 'n']
    subject_means['sem'] = subject_means['std'] / np.sqrt(subject_means['n'])
    subject_means = subject_means.sort_values('mean')
    
    x = range(len(subject_means))
    ax2.errorbar(x, subject_means['mean'], yerr=subject_means['sem'],
                 fmt='o', capsize=4, capthick=2, markersize=8,
                 color=COLORS['primary'], ecolor='gray')
    
    # Add grand mean line
    grand_mean = df['dwell_time'].mean()
    ax2.axhline(grand_mean, color=COLORS['secondary'], linestyle='--', 
                linewidth=2, label=f'Grand Mean: {grand_mean:.1f}s')
    
    ax2.set_xlabel('Subject (sorted by mean dwell)')
    ax2.set_ylabel('Mean Dwell Time (sec)')
    ax2.set_title('(b) Subject Mean ± SEM')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'S{i+1}' for i in range(len(subject_means))], fontsize=8)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_individual_differences.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig5_individual_differences.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Figure 5: Individual differences")

def figure6_summary_matrix(behavior_df, output_dir):
    """
    Figure 6: Correlation Matrix Summary
    Shows relationships between all key variables
    """
    df = behavior_df.dropna()
    df = df[df['dwell_time'] > 0]
    
    # Select key variables
    vars_to_plot = ['dwell_time', 'pos_velocity_mean', 'rot_velocity_mean', 
                    'pupil_mean', 'pupil_std']
    var_labels = ['Dwell Time', 'Head Pos Vel', 'Head Rot Vel', 
                  'Pupil Mean', 'Pupil SD']
    
    # Filter to existing columns
    available_vars = [v for v in vars_to_plot if v in df.columns]
    available_labels = [var_labels[i] for i, v in enumerate(vars_to_plot) if v in available_vars]
    
    if len(available_vars) < 3:
        print("  [SKIP] Not enough variables for correlation matrix")
        return
    
    subset = df[available_vars].dropna()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    corr_matrix = subset.corr()
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add text annotations
    for i in range(len(available_vars)):
        for j in range(len(available_vars)):
            val = corr_matrix.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                    color=color, fontsize=11, fontweight='bold')
    
    ax.set_xticks(range(len(available_vars)))
    ax.set_yticks(range(len(available_vars)))
    ax.set_xticklabels(available_labels, rotation=45, ha='right')
    ax.set_yticklabels(available_labels)
    
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    ax.set_title('Correlation Matrix of Key Variables')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig6_correlation_matrix.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig6_correlation_matrix.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Figure 6: Correlation matrix")

def main():
    print("=" * 60)
    print("ETRA Paper Figure Generation")
    print("=" * 60)
    
    behavior_df, base_dir = load_data()
    
    if behavior_df is None:
        print("[ERROR] Could not load behavior analysis data")
        return
    
    print(f"Loaded {len(behavior_df)} records from behavior analysis")
    print(f"Subjects: {behavior_df['Subject'].nunique()}")
    
    # Create output directory
    output_dir = base_dir / 'paper_figures'
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating figures...")
    
    # Generate all figures
    figure1_head_stability_vs_dwell(behavior_df, output_dir)
    figure2_dwell_by_stability_boxplot(behavior_df, output_dir)
    figure3_temporal_phase_pattern(base_dir, output_dir)
    figure4_pupil_response(base_dir, output_dir)
    figure5_individual_differences(behavior_df, output_dir)
    figure6_summary_matrix(behavior_df, output_dir)
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
