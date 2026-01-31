"""
Event-Related Pupil Response (ERPR) & VOR Analysis
===================================================
瞳孔反応と頭部-眼球協調の分析

分析内容:
1. ERPR: オブジェクト注視開始時の瞳孔径変化
2. VOR: 頭部回転と視線移動の協調パターン
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from scipy import stats, signal
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'Meiryo'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


def load_data(file_path):
    """データ読み込み"""
    df = pd.read_csv(file_path, skiprows=2)
    numeric_cols = ['TS', 'Gaze_ID', 'LP', 'RP',
                    'WorldGazePoint_X', 'WorldGazePoint_Y', 'WorldGazePoint_Z',
                    'CamPos_X', 'CamPos_Y', 'CamPos_Z',
                    'CamRot_X', 'CamRot_Y', 'CamRot_Z', 'CamRot_W']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 瞳孔径の平均
    df['pupil'] = (df['LP'] + df['RP']) / 2
    df['pupil'] = df['pupil'].replace([np.inf, -np.inf], np.nan)
    df['pupil'] = df['pupil'].interpolate(method='linear', limit=5)
    
    return df


def quaternion_to_euler(qx, qy, qz, qw):
    """クォータニオンからオイラー角（度）に変換"""
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))
    
    return np.degrees(yaw), np.degrees(pitch)


def compute_gaze_angle(df):
    """視線方向の角度を計算"""
    # WorldGazePointから視線角度を推定
    gaze_x = df['WorldGazePoint_X'].values
    gaze_z = df['WorldGazePoint_Z'].values
    
    gaze_yaw = np.degrees(np.arctan2(gaze_x, gaze_z))
    return gaze_yaw


def find_gaze_onsets(df, min_duration=0.5):
    """オブジェクト注視開始点を検出"""
    onsets = []
    
    # Gaze_IDの変化点を検出
    gaze_changes = df['Gaze_ID'].diff().fillna(1) != 0
    change_indices = np.where(gaze_changes)[0]
    
    for i, start_idx in enumerate(change_indices):
        gaze_id = df.iloc[start_idx]['Gaze_ID']
        
        if pd.isna(gaze_id) or gaze_id <= 0:
            continue
        
        # 終了点を見つける
        end_idx = change_indices[i + 1] if i + 1 < len(change_indices) else len(df)
        duration = df.iloc[end_idx - 1]['TS'] - df.iloc[start_idx]['TS']
        
        if duration >= min_duration:
            onsets.append({
                'onset_idx': start_idx,
                'onset_time': df.iloc[start_idx]['TS'],
                'gaze_id': int(gaze_id),
                'duration': duration
            })
    
    return onsets


def extract_erpr(df, onsets, pre_ms=500, post_ms=2000):
    """Event-Related Pupil Response を抽出"""
    fs = 1 / np.mean(np.diff(df['TS'].values[:1000]))
    pre_samples = int(pre_ms / 1000 * fs)
    post_samples = int(post_ms / 1000 * fs)
    
    epochs = []
    
    for onset in onsets:
        start = onset['onset_idx'] - pre_samples
        end = onset['onset_idx'] + post_samples
        
        if start < 0 or end >= len(df):
            continue
        
        epoch_data = df.iloc[start:end]['pupil'].values
        
        if np.isnan(epoch_data).sum() > len(epoch_data) * 0.2:
            continue
        
        # ベースライン補正（onset前500msの平均で引く）
        baseline = np.nanmean(epoch_data[:pre_samples])
        if np.isnan(baseline):
            continue
        
        epoch_normalized = epoch_data - baseline
        
        epochs.append({
            'gaze_id': onset['gaze_id'],
            'duration': onset['duration'],
            'epoch': epoch_normalized,
            'baseline': baseline
        })
    
    return epochs, pre_samples, post_samples, fs


def analyze_vor(df):
    """VOR (前庭動眼反射) 分析"""
    # 頭部回転角度
    head_yaw, head_pitch = quaternion_to_euler(
        df['CamRot_X'].values, df['CamRot_Y'].values,
        df['CamRot_Z'].values, df['CamRot_W'].values
    )
    
    # 視線角度
    gaze_yaw = compute_gaze_angle(df)
    
    # 角速度
    dt = np.diff(df['TS'].values)
    dt[dt == 0] = 0.001
    
    head_yaw_vel = np.diff(head_yaw) / dt
    head_yaw_vel = np.insert(head_yaw_vel, 0, 0)
    
    gaze_yaw_vel = np.diff(gaze_yaw) / dt
    gaze_yaw_vel = np.insert(gaze_yaw_vel, 0, 0)
    
    # 平滑化
    head_yaw_vel = uniform_filter1d(head_yaw_vel, size=5)
    gaze_yaw_vel = uniform_filter1d(gaze_yaw_vel, size=5)
    
    # VOR gain = -眼球速度 / 頭部速度（完璧なVORは-1）
    # 閾値以上の頭部運動のみ
    threshold = 5  # deg/s
    active_head = np.abs(head_yaw_vel) > threshold
    
    if np.sum(active_head) > 100:
        vor_gain = -gaze_yaw_vel[active_head] / head_yaw_vel[active_head]
        vor_gain = np.clip(vor_gain, -5, 5)  # 外れ値除外
    else:
        vor_gain = np.array([])
    
    return {
        'head_yaw': head_yaw,
        'gaze_yaw': gaze_yaw,
        'head_yaw_vel': head_yaw_vel,
        'gaze_yaw_vel': gaze_yaw_vel,
        'vor_gain': vor_gain,
        'active_head_mask': active_head
    }


def analyze_vor_by_object(df, vor_data, onsets):
    """オブジェクト別のVOR分析"""
    object_vor = {}
    
    for onset in onsets:
        gaze_id = onset['gaze_id']
        start_idx = onset['onset_idx']
        
        # 次のonsetまでのデータ
        end_idx = start_idx + int(onset['duration'] * 60)  # 約60Hz
        end_idx = min(end_idx, len(df) - 1)
        
        if end_idx - start_idx < 30:
            continue
        
        segment_mask = vor_data['active_head_mask'][start_idx:end_idx]
        if np.sum(segment_mask) < 10:
            continue
        
        segment_head_vel = vor_data['head_yaw_vel'][start_idx:end_idx][segment_mask]
        segment_gaze_vel = vor_data['gaze_yaw_vel'][start_idx:end_idx][segment_mask]
        
        segment_vor = -segment_gaze_vel / segment_head_vel
        segment_vor = np.clip(segment_vor, -5, 5)
        
        if gaze_id not in object_vor:
            object_vor[gaze_id] = []
        object_vor[gaze_id].extend(segment_vor.tolist())
    
    return object_vor


def analyze_subject(file_path, subject_name):
    """被験者ごとの分析"""
    df = load_data(file_path)
    
    # ERPR分析
    onsets = find_gaze_onsets(df, min_duration=0.5)
    epochs, pre_samples, post_samples, fs = extract_erpr(df, onsets)
    
    # VOR分析
    vor_data = analyze_vor(df)
    object_vor = analyze_vor_by_object(df, vor_data, onsets)
    
    return {
        'subject': subject_name,
        'epochs': epochs,
        'pre_samples': pre_samples,
        'post_samples': post_samples,
        'fs': fs,
        'vor_data': vor_data,
        'object_vor': object_vor,
        'onsets': onsets
    }


def create_visualizations(all_results, output_dir):
    """可視化"""
    
    # ==========================================
    # Figure 1: Grand Average ERPR
    # ==========================================
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle('Event-Related Pupil Response (ERPR)', fontsize=14, fontweight='bold')
    
    # 全エポックを集める
    all_epochs = []
    epoch_lengths = []
    for result in all_results:
        for ep in result['epochs']:
            all_epochs.append(ep['epoch'])
            epoch_lengths.append(len(ep['epoch']))
    
    if all_epochs:
        # 最小長に揃える
        min_len = min(epoch_lengths)
        aligned_epochs = np.array([ep[:min_len] for ep in all_epochs])
        
        # Grand average
        pre_samples = all_results[0]['pre_samples']
        fs = all_results[0]['fs']
        
        time_axis = (np.arange(min_len) - pre_samples) / fs * 1000  # ms
        
        mean_erpr = np.nanmean(aligned_epochs, axis=0)
        sem_erpr = np.nanstd(aligned_epochs, axis=0) / np.sqrt(len(aligned_epochs))
        
        ax = axes[0]
        ax.plot(time_axis, mean_erpr, 'b-', linewidth=2, label='Grand Mean')
        ax.fill_between(time_axis, mean_erpr - sem_erpr, mean_erpr + sem_erpr, alpha=0.3)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Onset')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('Time from Onset (ms)')
        ax.set_ylabel('Pupil Change from Baseline (mm)')
        ax.set_title(f'Grand Average ERPR (N={len(aligned_epochs)} epochs)')
        ax.legend()
        ax.set_xlim([-500, 2000])
        
        # 統計検定：0-500msの平均 vs 500-1000msの平均
        pre_window = (time_axis >= 0) & (time_axis < 500)
        post_window = (time_axis >= 500) & (time_axis < 1000)
        
        pre_means = np.nanmean(aligned_epochs[:, pre_window], axis=1)
        post_means = np.nanmean(aligned_epochs[:, post_window], axis=1)
        
        t, p = stats.ttest_rel(pre_means, post_means)
        
        ax = axes[1]
        ax.boxplot([pre_means, post_means], labels=['0-500ms', '500-1000ms'])
        ax.set_ylabel('Mean Pupil Change (mm)')
        ax.set_title(f'Pupil Change by Time Window\nt={t:.2f}, p={p:.4f}')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig1.savefig(output_dir / 'erpr_grand_average.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # ==========================================
    # Figure 2: ERPR by Duration (High vs Low Interest)
    # ==========================================
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    if all_epochs:
        # Duration中央値で分割
        durations = [r['epochs'][i]['duration'] for r in all_results for i in range(len(r['epochs']))]
        median_dur = np.median(durations)
        
        long_epochs = []
        short_epochs = []
        for result in all_results:
            for ep in result['epochs']:
                if len(ep['epoch']) >= min_len:
                    if ep['duration'] >= median_dur:
                        long_epochs.append(ep['epoch'][:min_len])
                    else:
                        short_epochs.append(ep['epoch'][:min_len])
        
        if long_epochs and short_epochs:
            long_mean = np.nanmean(long_epochs, axis=0)
            short_mean = np.nanmean(short_epochs, axis=0)
            
            ax.plot(time_axis, long_mean, 'r-', linewidth=2, label=f'Long Dwell (N={len(long_epochs)})')
            ax.plot(time_axis, short_mean, 'b-', linewidth=2, label=f'Short Dwell (N={len(short_epochs)})')
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.set_xlabel('Time from Onset (ms)')
            ax.set_ylabel('Pupil Change from Baseline (mm)')
            ax.set_title('ERPR: Long vs Short Viewing Duration')
            ax.legend()
            ax.set_xlim([-500, 2000])
    
    plt.tight_layout()
    fig2.savefig(output_dir / 'erpr_by_duration.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # ==========================================
    # Figure 3: VOR Analysis
    # ==========================================
    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle('Vestibulo-Ocular Reflex (VOR) Analysis', fontsize=14, fontweight='bold')
    
    # 全VOR gainを集める
    all_vor = []
    for result in all_results:
        all_vor.extend(result['vor_data']['vor_gain'].tolist())
    
    if all_vor:
        all_vor = np.array(all_vor)
        all_vor = all_vor[np.isfinite(all_vor)]
        
        ax = axes[0]
        ax.hist(all_vor, bins=50, alpha=0.7, color='#2196F3', density=True)
        ax.axvline(x=1, color='red', linestyle='--', label='Perfect VOR (gain=1)')
        ax.axvline(x=np.median(all_vor), color='green', linestyle='--', label=f'Median={np.median(all_vor):.2f}')
        ax.set_xlabel('VOR Gain')
        ax.set_ylabel('Density')
        ax.set_title('VOR Gain Distribution')
        ax.legend()
        ax.set_xlim([-2, 3])
    
    # VOR by Object
    all_object_vor = {}
    for result in all_results:
        for gid, gains in result['object_vor'].items():
            if gid not in all_object_vor:
                all_object_vor[gid] = []
            all_object_vor[gid].extend(gains)
    
    if all_object_vor:
        ax = axes[1]
        object_ids = sorted(all_object_vor.keys())
        vor_means = [np.nanmean(all_object_vor[gid]) for gid in object_ids]
        vor_stds = [np.nanstd(all_object_vor[gid]) for gid in object_ids]
        
        ax.bar(range(len(object_ids)), vor_means, yerr=vor_stds, alpha=0.7, color='#4CAF50', capsize=5)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Perfect VOR')
        ax.set_xticks(range(len(object_ids)))
        ax.set_xticklabels([f'ID:{gid}' for gid in object_ids])
        ax.set_xlabel('Object ID')
        ax.set_ylabel('Mean VOR Gain')
        ax.set_title('VOR Gain by Object')
        ax.legend()
    
    # Duration vs VOR
    ax = axes[2]
    vor_by_duration = {'short': [], 'long': []}
    for result in all_results:
        for onset in result['onsets']:
            gid = onset['gaze_id']
            dur = onset['duration']
            if gid in result['object_vor']:
                gains = result['object_vor'][gid]
                if gains:
                    median_dur = np.median([o['duration'] for o in result['onsets']])
                    if dur >= median_dur:
                        vor_by_duration['long'].extend(gains)
                    else:
                        vor_by_duration['short'].extend(gains)
    
    if vor_by_duration['short'] and vor_by_duration['long']:
        ax.boxplot([vor_by_duration['short'], vor_by_duration['long']], 
                   labels=['Short Dwell', 'Long Dwell'])
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel('VOR Gain')
        ax.set_title('VOR Gain by Viewing Duration')
        
        t, p = stats.ttest_ind(vor_by_duration['short'], vor_by_duration['long'])
        ax.text(0.5, 0.95, f't={t:.2f}, p={p:.4f}', transform=ax.transAxes, 
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig3.savefig(output_dir / 'vor_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # ==========================================
    # Figure 4: Head-Eye Coordination Time Series
    # ==========================================
    # 最初の被験者の例を表示
    if all_results:
        result = all_results[0]
        vor_data = result['vor_data']
        
        fig4, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        fig4.suptitle(f'Head-Eye Coordination Example ({result["subject"]})', fontsize=14, fontweight='bold')
        
        # 最初の30秒のみ
        n_samples = min(len(vor_data['head_yaw']), 30 * 60)
        
        ax = axes[0]
        ax.plot(vor_data['head_yaw'][:n_samples], label='Head Yaw', alpha=0.8)
        ax.plot(vor_data['gaze_yaw'][:n_samples], label='Gaze Yaw', alpha=0.8)
        ax.set_ylabel('Angle (deg)')
        ax.set_title('Head and Gaze Direction')
        ax.legend(loc='upper right')
        
        ax = axes[1]
        ax.plot(vor_data['head_yaw_vel'][:n_samples], label='Head Velocity', alpha=0.8)
        ax.plot(vor_data['gaze_yaw_vel'][:n_samples], label='Gaze Velocity', alpha=0.8)
        ax.set_ylabel('Angular Velocity (deg/s)')
        ax.set_title('Head and Gaze Velocity')
        ax.legend(loc='upper right')
        
        ax = axes[2]
        vor_gain_ts = np.zeros(n_samples)
        active_mask = vor_data['active_head_mask'][:n_samples]
        vor_gain_ts[active_mask] = -vor_data['gaze_yaw_vel'][:n_samples][active_mask] / vor_data['head_yaw_vel'][:n_samples][active_mask]
        vor_gain_ts = np.clip(vor_gain_ts, -3, 3)
        ax.plot(vor_gain_ts, alpha=0.7, color='#9C27B0')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Perfect VOR')
        ax.set_ylabel('VOR Gain')
        ax.set_xlabel('Sample')
        ax.set_title('Instantaneous VOR Gain')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        fig4.savefig(output_dir / 'head_eye_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close(fig4)


def main():
    """メイン処理"""
    base_dir = Path(r'c:\Users\kosuk\ETRA\RTA\RTAデータ')
    output_dir = Path(r'c:\Users\kosuk\ETRA\RTA\erpr_vor_analysis')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("ERPR & VOR Analysis")
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
                    result = analyze_subject(csv_file, subject_name)
                    all_results.append(result)
                    print(f"   Epochs: {len(result['epochs'])}, VOR samples: {len(result['vor_data']['vor_gain'])}")
                except Exception as e:
                    print(f"   [ERROR] {str(e)[:80]}")
    
    # 可視化
    print("\nGenerating visualizations...")
    create_visualizations(all_results, output_dir)
    
    # サマリー
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_epochs = sum(len(r['epochs']) for r in all_results)
    print(f"Total ERPR epochs: {total_epochs}")
    
    all_vor = []
    for r in all_results:
        all_vor.extend(r['vor_data']['vor_gain'].tolist())
    all_vor = np.array([v for v in all_vor if np.isfinite(v)])
    
    print(f"Total VOR samples: {len(all_vor)}")
    print(f"Mean VOR Gain: {np.mean(all_vor):.3f} (SD={np.std(all_vor):.3f})")
    print(f"Median VOR Gain: {np.median(all_vor):.3f}")
    
    print("\n" + "=" * 60)
    print(f"Complete! Results saved to {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
