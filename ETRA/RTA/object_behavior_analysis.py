"""
Object Viewing Behavior Analysis
================================
オブジェクト注視時の特徴的な挙動を分析

分析対象:
- Gaze_ID別の注視パターン
- 瞳孔径変化（LP/RP）
- まばたき（BlinkCount）
- 頭部の安定性（CamPos/CamRot）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'Meiryo'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


def load_data(file_path):
    """データ読み込み"""
    df = pd.read_csv(file_path, skiprows=2)
    numeric_cols = ['TS', 'LP', 'RP', 'Gaze_ID', 'BlinkCount',
                    'CamPos_X', 'CamPos_Y', 'CamPos_Z',
                    'CamRot_X', 'CamRot_Y', 'CamRot_Z', 'CamRot_W']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def compute_head_stability(df):
    """頭部の安定性を計算（位置と回転の変動）"""
    # 位置の変動（フレーム間の移動量）
    pos_dx = np.diff(df['CamPos_X'].values)
    pos_dy = np.diff(df['CamPos_Y'].values)
    pos_dz = np.diff(df['CamPos_Z'].values)
    pos_velocity = np.sqrt(pos_dx**2 + pos_dy**2 + pos_dz**2)
    pos_velocity = np.insert(pos_velocity, 0, 0)
    
    # 回転の変動（クォータニオンの変化量）
    rot_dx = np.diff(df['CamRot_X'].values)
    rot_dy = np.diff(df['CamRot_Y'].values)
    rot_dz = np.diff(df['CamRot_Z'].values)
    rot_dw = np.diff(df['CamRot_W'].values)
    rot_velocity = np.sqrt(rot_dx**2 + rot_dy**2 + rot_dz**2 + rot_dw**2)
    rot_velocity = np.insert(rot_velocity, 0, 0)
    
    return pos_velocity, rot_velocity


def analyze_object_behavior(df, output_dir, subject_name):
    """オブジェクト別の行動パターンを分析"""
    
    # 派生変数
    pos_vel, rot_vel = compute_head_stability(df)
    df['pos_velocity'] = pos_vel
    df['rot_velocity'] = rot_vel
    df['pupil_mean'] = (df['LP'] + df['RP']) / 2
    
    # まばたき発生を検出（BlinkCountの増加）
    df['blink_event'] = df['BlinkCount'].diff().fillna(0).clip(lower=0)
    
    # Gaze_ID別の統計
    object_stats = {}
    
    for gaze_id in df['Gaze_ID'].unique():
        if pd.isna(gaze_id):
            continue
        gaze_id = int(gaze_id)
        obj_data = df[df['Gaze_ID'] == gaze_id]
        
        if len(obj_data) < 10:
            continue
        
        # サンプリングレート推定
        dt = np.mean(np.diff(df['TS'].values[:1000]))
        
        # 統計量計算
        stats_dict = {
            'gaze_id': gaze_id,
            'total_samples': len(obj_data),
            'dwell_time_sec': len(obj_data) * dt,
            
            # 瞳孔径
            'pupil_mean': obj_data['pupil_mean'].mean(),
            'pupil_std': obj_data['pupil_mean'].std(),
            'pupil_baseline_diff': obj_data['pupil_mean'].mean() - df['pupil_mean'].mean(),
            
            # まばたき
            'blink_count': obj_data['blink_event'].sum(),
            'blink_rate_per_min': obj_data['blink_event'].sum() / (len(obj_data) * dt / 60) if len(obj_data) * dt > 0 else 0,
            
            # 頭部安定性
            'head_pos_stability': obj_data['pos_velocity'].mean(),  # 低い = 安定
            'head_rot_stability': obj_data['rot_velocity'].mean(),  # 低い = 安定
            'head_pos_stability_std': obj_data['pos_velocity'].std(),
            'head_rot_stability_std': obj_data['rot_velocity'].std(),
        }
        object_stats[gaze_id] = stats_dict
    
    stats_df = pd.DataFrame(object_stats).T
    stats_df = stats_df.sort_values('dwell_time_sec', ascending=False)
    
    # ==========================================
    # 可視化
    # ==========================================
    
    # Figure 1: オブジェクト別の主要指標比較
    fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig1.suptitle(f'Object Viewing Behavior - {subject_name}', fontsize=14, fontweight='bold')
    
    objects = stats_df.index.astype(int).tolist()
    x = np.arange(len(objects))
    
    # 1. Dwell Time
    ax = axes[0, 0]
    colors = ['#4CAF50' if gid > 0 else '#9E9E9E' for gid in objects]
    ax.bar(x, stats_df['dwell_time_sec'], color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'ID:{g}' for g in objects])
    ax.set_ylabel('Dwell Time (sec)')
    ax.set_title('A. Total Viewing Time per Object')
    
    # 2. 瞳孔径（ベースラインからの差分）
    ax = axes[0, 1]
    colors = ['#F44336' if v > 0 else '#2196F3' for v in stats_df['pupil_baseline_diff']]
    ax.bar(x, stats_df['pupil_baseline_diff'], color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'ID:{g}' for g in objects])
    ax.set_ylabel('Pupil Δ from Baseline (mm)')
    ax.set_title('B. Pupil Dilation (vs Baseline)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 3. まばたき率
    ax = axes[0, 2]
    ax.bar(x, stats_df['blink_rate_per_min'], color='#9C27B0', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'ID:{g}' for g in objects])
    ax.set_ylabel('Blink Rate (per min)')
    ax.set_title('C. Blink Rate during Viewing')
    
    # 4. 頭部位置安定性
    ax = axes[1, 0]
    ax.bar(x, stats_df['head_pos_stability'], color='#00BCD4', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'ID:{g}' for g in objects])
    ax.set_ylabel('Position Velocity (m/frame)')
    ax.set_title('D. Head Position Stability (lower=stable)')
    
    # 5. 頭部回転安定性
    ax = axes[1, 1]
    ax.bar(x, stats_df['head_rot_stability'], color='#FF9800', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'ID:{g}' for g in objects])
    ax.set_ylabel('Rotation Velocity (rad/frame)')
    ax.set_title('E. Head Rotation Stability (lower=stable)')
    
    # 6. 瞳孔径の変動性
    ax = axes[1, 2]
    ax.bar(x, stats_df['pupil_std'], color='#E91E63', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'ID:{g}' for g in objects])
    ax.set_ylabel('Pupil SD (mm)')
    ax.set_title('F. Pupil Variability during Viewing')
    
    plt.tight_layout()
    fig1.savefig(output_dir / f'{subject_name}_object_behavior.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Figure 2: 興味指標の相関マトリックス
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    corr_cols = ['dwell_time_sec', 'pupil_baseline_diff', 'blink_rate_per_min', 
                 'head_pos_stability', 'head_rot_stability', 'pupil_std']
    corr_labels = ['Dwell Time', 'Pupil Δ', 'Blink Rate', 
                   'Head Pos Vel', 'Head Rot Vel', 'Pupil SD']
    
    if len(stats_df) > 2:
        corr_data = stats_df[corr_cols].dropna()
        if len(corr_data) > 2:
            corr = corr_data.corr()
            im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(corr_labels)))
            ax.set_yticks(np.arange(len(corr_labels)))
            ax.set_xticklabels(corr_labels, rotation=45, ha='right')
            ax.set_yticklabels(corr_labels)
            
            for i in range(len(corr)):
                for j in range(len(corr)):
                    color = 'white' if abs(corr.iloc[i, j]) > 0.5 else 'black'
                    ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', color=color)
            
            plt.colorbar(im, ax=ax, label='Correlation')
            ax.set_title(f'Feature Correlation - {subject_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig2.savefig(output_dir / f'{subject_name}_correlation.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # Figure 3: 時系列での行動パターン
    fig3, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig3.suptitle(f'Time Series Behavior - {subject_name}', fontsize=14, fontweight='bold')
    
    time = df['TS'].values
    
    # Gaze ID
    ax = axes[0]
    scatter = ax.scatter(time, df['Gaze_ID'], c=df['Gaze_ID'], cmap='tab10', s=1, alpha=0.5)
    ax.set_ylabel('Gaze ID')
    ax.set_title('A. Attended Object over Time')
    
    # 瞳孔径
    ax = axes[1]
    ax.plot(time, df['pupil_mean'], color='#2196F3', alpha=0.7, linewidth=0.5)
    ax.axhline(y=df['pupil_mean'].mean(), color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_ylabel('Pupil (mm)')
    ax.set_title('B. Pupil Diameter')
    ax.legend(loc='upper right')
    
    # 頭部安定性
    ax = axes[2]
    ax.plot(time, df['pos_velocity'], color='#00BCD4', alpha=0.7, linewidth=0.5, label='Position')
    ax.plot(time, df['rot_velocity'], color='#FF9800', alpha=0.7, linewidth=0.5, label='Rotation')
    ax.set_ylabel('Head Velocity')
    ax.set_title('C. Head Stability (lower=more stable)')
    ax.legend(loc='upper right')
    
    # まばたき
    ax = axes[3]
    blink_times = time[df['blink_event'] > 0]
    ax.scatter(blink_times, [1]*len(blink_times), color='#9C27B0', s=10, alpha=0.7)
    ax.set_ylabel('Blink')
    ax.set_xlabel('Time (sec)')
    ax.set_title('D. Blink Events')
    ax.set_ylim(0, 2)
    
    plt.tight_layout()
    fig3.savefig(output_dir / f'{subject_name}_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"  [OK] Analysis complete: {subject_name}")
    
    return stats_df


def main():
    """メイン処理"""
    base_dir = Path(r'c:\Users\kosuk\ETRA\RTA\RTAデータ')
    output_dir = Path(r'c:\Users\kosuk\ETRA\RTA\behavior_analysis')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Object Viewing Behavior Analysis")
    print("=" * 60)
    
    all_stats = []
    
    for subject_folder in base_dir.iterdir():
        if subject_folder.is_dir():
            csv_files = list(subject_folder.glob('*.csv'))
            if csv_files:
                csv_file = csv_files[0]
                subject_name = subject_folder.name.replace(' ', '_').replace('（', '_').replace('）', '')
                
                print(f"\nProcessing: {subject_folder.name}")
                
                try:
                    df = load_data(csv_file)
                    stats_df = analyze_object_behavior(df, output_dir, subject_name)
                    stats_df['subject'] = subject_name
                    all_stats.append(stats_df)
                    
                    # 個別CSV出力
                    stats_df.to_csv(output_dir / f'{subject_name}_stats.csv')
                    
                except Exception as e:
                    print(f"  [ERROR] {str(e)[:100]}")
    
    # 全被験者統合
    if all_stats:
        combined = pd.concat(all_stats, ignore_index=True)
        combined.to_csv(output_dir / 'all_subjects_stats.csv', index=False)
        
        # グループ平均の可視化
        create_group_summary(combined, output_dir)
    
    print("\n" + "=" * 60)
    print(f"Complete! Results saved to {output_dir}")
    print("=" * 60)


def create_group_summary(combined, output_dir):
    """全被験者のグループサマリー"""
    
    # Gaze_ID別の平均（0以外）
    group_stats = combined[combined['gaze_id'] > 0].groupby('gaze_id').agg({
        'dwell_time_sec': 'mean',
        'pupil_baseline_diff': 'mean',
        'blink_rate_per_min': 'mean',
        'head_pos_stability': 'mean',
        'head_rot_stability': 'mean',
        'pupil_std': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Group Mean by Object (N=all subjects)', fontsize=14, fontweight='bold')
    
    objects = group_stats['gaze_id'].tolist()
    x = np.arange(len(objects))
    
    metrics = [
        ('dwell_time_sec', 'Dwell Time (sec)', '#4CAF50'),
        ('pupil_baseline_diff', 'Pupil Δ (mm)', '#F44336'),
        ('blink_rate_per_min', 'Blink Rate (/min)', '#9C27B0'),
        ('head_pos_stability', 'Head Pos Velocity', '#00BCD4'),
        ('head_rot_stability', 'Head Rot Velocity', '#FF9800'),
        ('pupil_std', 'Pupil SD (mm)', '#E91E63')
    ]
    
    for ax, (metric, label, color) in zip(axes.flatten(), metrics):
        ax.bar(x, group_stats[metric], color=color, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'ID:{int(g)}' for g in objects])
        ax.set_ylabel(label)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'group_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print("\n  [OK] Group summary generated")


if __name__ == '__main__':
    main()
