"""
Within-Object Gaze Pattern Analysis
====================================
オブジェクト内の視線パターン分析

分析内容:
- 視線分散（Dispersion）: 広く見回す vs 一点集中
- 視線エントロピー: 探索的 vs 集中的
- スキャンパス長: 視線移動総距離
- 時系列パターン: 注視の前半・後半での変化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'Meiryo'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


def load_data(file_path):
    """データ読み込み"""
    df = pd.read_csv(file_path, skiprows=2)
    numeric_cols = ['TS', 'Gaze_ID', 'WorldGazePoint_X', 'WorldGazePoint_Y', 'WorldGazePoint_Z',
                    'LP', 'RP', 'CamPos_X', 'CamPos_Y', 'CamPos_Z']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def compute_spatial_entropy(positions, n_bins=10):
    """空間的エントロピーを計算（視線位置の分布の均一性）"""
    if len(positions) < 5:
        return np.nan
    
    # 2D投影（X-Y平面）
    x, y = positions[:, 0], positions[:, 1]
    
    # ヒストグラムで離散化
    hist, _, _ = np.histogram2d(x, y, bins=n_bins)
    hist = hist.flatten()
    hist = hist[hist > 0]  # 0を除外
    
    # 正規化
    prob = hist / hist.sum()
    
    # エントロピー計算
    entropy = -np.sum(prob * np.log2(prob + 1e-10))
    
    # 最大エントロピーで正規化（0-1）
    max_entropy = np.log2(n_bins * n_bins)
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy


def compute_dispersion(positions):
    """視線分散を計算（標準偏差ベース）"""
    if len(positions) < 3:
        return np.nan
    
    # 各軸の標準偏差の平均
    dispersion = np.mean(np.std(positions, axis=0))
    return dispersion


def compute_scanpath_length(positions):
    """スキャンパス長（総移動距離）"""
    if len(positions) < 2:
        return 0
    
    diffs = np.diff(positions, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    return np.sum(distances)


def compute_scanpath_velocity(positions, timestamps):
    """スキャンパス速度"""
    if len(positions) < 2:
        return np.nan
    
    diffs = np.diff(positions, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    time_diffs = np.diff(timestamps)
    time_diffs[time_diffs == 0] = 0.001  # ゼロ除算防止
    
    velocities = distances / time_diffs
    return np.median(velocities)


def analyze_temporal_pattern(positions, timestamps, n_phases=3):
    """時系列パターン分析（前半・中間・後半）"""
    if len(positions) < n_phases * 3:
        return {}
    
    n = len(positions)
    phase_size = n // n_phases
    
    results = {}
    for i in range(n_phases):
        start = i * phase_size
        end = (i + 1) * phase_size if i < n_phases - 1 else n
        
        phase_pos = positions[start:end]
        phase_ts = timestamps[start:end]
        
        results[f'phase_{i+1}_dispersion'] = compute_dispersion(phase_pos)
        results[f'phase_{i+1}_velocity'] = compute_scanpath_velocity(phase_pos, phase_ts)
        results[f'phase_{i+1}_entropy'] = compute_spatial_entropy(phase_pos, n_bins=5)
    
    return results


def extract_viewing_episodes(df):
    """連続した注視エピソードを抽出"""
    episodes = []
    
    # Gaze_IDの変化点を検出
    df['gaze_change'] = df['Gaze_ID'].diff().fillna(1) != 0
    df['episode_id'] = df['gaze_change'].cumsum()
    
    for episode_id, group in df.groupby('episode_id'):
        gaze_id = group['Gaze_ID'].iloc[0]
        
        if pd.isna(gaze_id) or gaze_id <= 0:
            continue
        
        if len(group) < 10:  # 最低10フレーム
            continue
        
        positions = group[['WorldGazePoint_X', 'WorldGazePoint_Y', 'WorldGazePoint_Z']].values
        
        # NaNチェック
        if np.isnan(positions).any():
            continue
        
        timestamps = group['TS'].values
        
        episodes.append({
            'episode_id': episode_id,
            'gaze_id': int(gaze_id),
            'positions': positions,
            'timestamps': timestamps,
            'duration': timestamps[-1] - timestamps[0],
            'n_frames': len(group)
        })
    
    return episodes


def analyze_subject(file_path, subject_name):
    """被験者ごとの分析"""
    df = load_data(file_path)
    episodes = extract_viewing_episodes(df)
    
    results = []
    
    for ep in episodes:
        pos = ep['positions']
        ts = ep['timestamps']
        
        # 基本指標
        result = {
            'subject': subject_name,
            'episode_id': ep['episode_id'],
            'gaze_id': ep['gaze_id'],
            'duration': ep['duration'],
            'n_frames': ep['n_frames'],
            
            # 空間パターン
            'dispersion': compute_dispersion(pos),
            'entropy': compute_spatial_entropy(pos),
            'scanpath_length': compute_scanpath_length(pos),
            'scanpath_velocity': compute_scanpath_velocity(pos, ts),
        }
        
        # 時系列パターン（3フェーズ）
        temporal = analyze_temporal_pattern(pos, ts, n_phases=3)
        result.update(temporal)
        
        results.append(result)
    
    return pd.DataFrame(results)


def create_visualizations(all_results, output_dir):
    """可視化"""
    
    # ==========================================
    # Figure 1: エントロピー vs Dwell Time
    # ==========================================
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle('Gaze Pattern Metrics vs Viewing Duration', fontsize=14, fontweight='bold')
    
    # Entropy
    ax = axes[0]
    ax.scatter(all_results['duration'], all_results['entropy'], alpha=0.5, s=30, c='#9C27B0')
    slope, intercept, r, p, se = stats.linregress(
        all_results['duration'].dropna(), 
        all_results['entropy'].dropna()
    )
    x_line = np.linspace(all_results['duration'].min(), all_results['duration'].max(), 100)
    ax.plot(x_line, intercept + slope * x_line, 'r-', alpha=0.7)
    ax.set_xlabel('Duration (sec)')
    ax.set_ylabel('Spatial Entropy (normalized)')
    ax.set_title(f'Entropy vs Duration\nr={r:.3f}, p={p:.4f}')
    
    # Dispersion
    ax = axes[1]
    ax.scatter(all_results['duration'], all_results['dispersion'], alpha=0.5, s=30, c='#2196F3')
    r, p = stats.pearsonr(all_results['duration'].dropna(), all_results['dispersion'].dropna())
    ax.set_xlabel('Duration (sec)')
    ax.set_ylabel('Dispersion (m)')
    ax.set_title(f'Dispersion vs Duration\nr={r:.3f}, p={p:.4f}')
    
    # Scanpath Length
    ax = axes[2]
    ax.scatter(all_results['duration'], all_results['scanpath_length'], alpha=0.5, s=30, c='#4CAF50')
    r, p = stats.pearsonr(all_results['duration'].dropna(), all_results['scanpath_length'].dropna())
    ax.set_xlabel('Duration (sec)')
    ax.set_ylabel('Scanpath Length (m)')
    ax.set_title(f'Scanpath vs Duration\nr={r:.3f}, p={p:.4f}')
    
    plt.tight_layout()
    fig1.savefig(output_dir / 'gaze_pattern_vs_duration.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # ==========================================
    # Figure 2: 時系列パターン（フェーズ別）
    # ==========================================
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('Temporal Evolution of Gaze Patterns (3 Phases)', fontsize=14, fontweight='bold')
    
    # 長い注視のみ（3秒以上）
    long_episodes = all_results[all_results['duration'] >= 3]
    
    # Dispersion by phase
    ax = axes[0]
    phase_disp = [
        long_episodes['phase_1_dispersion'].dropna(),
        long_episodes['phase_2_dispersion'].dropna(),
        long_episodes['phase_3_dispersion'].dropna()
    ]
    bp = ax.boxplot(phase_disp, labels=['Early', 'Middle', 'Late'])
    ax.set_ylabel('Dispersion')
    ax.set_title('Dispersion Over Time')
    
    # Velocity by phase
    ax = axes[1]
    phase_vel = [
        long_episodes['phase_1_velocity'].dropna(),
        long_episodes['phase_2_velocity'].dropna(),
        long_episodes['phase_3_velocity'].dropna()
    ]
    bp = ax.boxplot(phase_vel, labels=['Early', 'Middle', 'Late'])
    ax.set_ylabel('Gaze Velocity (m/s)')
    ax.set_title('Gaze Velocity Over Time')
    
    # Entropy by phase
    ax = axes[2]
    phase_ent = [
        long_episodes['phase_1_entropy'].dropna(),
        long_episodes['phase_2_entropy'].dropna(),
        long_episodes['phase_3_entropy'].dropna()
    ]
    bp = ax.boxplot(phase_ent, labels=['Early', 'Middle', 'Late'])
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy Over Time')
    
    plt.tight_layout()
    fig2.savefig(output_dir / 'temporal_pattern.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # ==========================================
    # Figure 3: オブジェクト別パターン
    # ==========================================
    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle('Gaze Patterns by Object', fontsize=14, fontweight='bold')
    
    obj_stats = all_results.groupby('gaze_id').agg({
        'duration': 'mean',
        'entropy': 'mean',
        'dispersion': 'mean',
        'scanpath_velocity': 'mean'
    }).reset_index()
    
    ax = axes[0]
    ax.bar(obj_stats['gaze_id'], obj_stats['entropy'], color='#9C27B0', alpha=0.7)
    ax.set_xlabel('Object ID')
    ax.set_ylabel('Mean Entropy')
    ax.set_title('Entropy by Object')
    
    ax = axes[1]
    ax.bar(obj_stats['gaze_id'], obj_stats['dispersion'], color='#2196F3', alpha=0.7)
    ax.set_xlabel('Object ID')
    ax.set_ylabel('Mean Dispersion')
    ax.set_title('Dispersion by Object')
    
    ax = axes[2]
    ax.scatter(obj_stats['duration'], obj_stats['entropy'], s=100, c='#E91E63', alpha=0.8)
    for i, row in obj_stats.iterrows():
        ax.annotate(f'ID:{int(row["gaze_id"])}', (row['duration'], row['entropy']), fontsize=9)
    ax.set_xlabel('Mean Duration (sec)')
    ax.set_ylabel('Mean Entropy')
    ax.set_title('Duration vs Entropy by Object')
    
    plt.tight_layout()
    fig3.savefig(output_dir / 'object_pattern.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # ==========================================
    # Figure 4: 相関マトリックス
    # ==========================================
    fig4, ax = plt.subplots(figsize=(10, 8))
    
    corr_cols = ['duration', 'entropy', 'dispersion', 'scanpath_length', 'scanpath_velocity']
    corr = all_results[corr_cols].corr()
    
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(['Duration', 'Entropy', 'Dispersion', 'Scanpath', 'Velocity'], rotation=45, ha='right')
    ax.set_yticklabels(['Duration', 'Entropy', 'Dispersion', 'Scanpath', 'Velocity'])
    
    for i in range(len(corr)):
        for j in range(len(corr)):
            color = 'white' if abs(corr.iloc[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', color=color)
    
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('Gaze Pattern Feature Correlation', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig4.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)


def main():
    """メイン処理"""
    base_dir = Path(r'c:\Users\kosuk\ETRA\RTA\RTAデータ')
    output_dir = Path(r'c:\Users\kosuk\ETRA\RTA\gaze_pattern_analysis')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Within-Object Gaze Pattern Analysis")
    print("=" * 60)
    
    all_results = []
    
    for subject_folder in base_dir.iterdir():
        if subject_folder.is_dir():
            csv_files = list(subject_folder.glob('*.csv'))
            if csv_files:
                csv_file = csv_files[0]
                subject_name = subject_folder.name.replace(' ', '_').replace('（', '_').replace('）', '')
                
                print(f"\nProcessing: {subject_folder.name}")
                
                try:
                    results = analyze_subject(csv_file, subject_name)
                    all_results.append(results)
                    print(f"   Episodes: {len(results)}")
                except Exception as e:
                    print(f"   [ERROR] {str(e)[:80]}")
    
    # 統合
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(output_dir / 'all_gaze_patterns.csv', index=False)
        
        # 可視化
        print("\nGenerating visualizations...")
        create_visualizations(combined, output_dir)
        
        # サマリー
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total episodes: {len(combined)}")
        print(f"Subjects: {combined['subject'].nunique()}")
        print(f"\nCorrelations with Duration:")
        for col in ['entropy', 'dispersion', 'scanpath_length']:
            r, p = stats.pearsonr(combined['duration'].dropna(), combined[col].dropna())
            sig = "**" if p < 0.05 else ""
            print(f"   {col}: r={r:.3f}, p={p:.4f} {sig}")
        
        # 時系列変化の統計検定
        print("\nTemporal Pattern (Repeated Measures):")
        long_eps = combined[combined['duration'] >= 3]
        if len(long_eps) > 10:
            for metric in ['dispersion', 'velocity', 'entropy']:
                cols = [f'phase_1_{metric}', f'phase_2_{metric}', f'phase_3_{metric}']
                data = [long_eps[c].dropna() for c in cols]
                if all(len(d) > 5 for d in data):
                    f_stat, p_val = stats.f_oneway(*data)
                    sig = "**" if p_val < 0.05 else ""
                    print(f"   {metric}: F={f_stat:.2f}, p={p_val:.4f} {sig}")
    
    print("\n" + "=" * 60)
    print(f"Complete! Results saved to {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
