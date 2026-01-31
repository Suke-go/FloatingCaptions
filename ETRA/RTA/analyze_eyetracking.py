"""
Eye Tracking Data Analysis & Visualization for Mathematical Modeling
=====================================================================
RTAデータの構造分析と数理モデル構築のための可視化

データ構造:
- TS: タイムスタンプ（秒）
- N: フレーム番号
- EYE_OPENING_L/R: 左右目の開き具合
- LP/RP: 左右瞳孔径
- CamPos_X/Y/Z: カメラ位置（頭部位置）
- CamRot_X/Y/Z/W: カメラ回転（クォータニオン）
- WorldGazePoint_X/Y/Z: 視線交差点のワールド座標
- Gaze_ID: 注視対象オブジェクトID
- BlinkCount: まばたき累積回数
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
matplotlib.rcParams['font.family'] = 'Meiryo'
matplotlib.rcParams['axes.unicode_minus'] = False

# スタイル設定
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['grid.alpha'] = 0.3

def load_eyetracking_data(file_path):
    """CSVファイルからアイトラッキングデータを読み込む"""
    df = pd.read_csv(file_path, skiprows=2)  # ヘッダー行をスキップ
    # NaN/Inf値を処理
    numeric_cols = ['LP', 'RP', 'EYE_OPENING_L', 'EYE_OPENING_R', 
                    'WorldGazePoint_X', 'WorldGazePoint_Y', 'WorldGazePoint_Z',
                    'CamPos_X', 'CamPos_Y', 'CamPos_Z',
                    'CamRot_X', 'CamRot_Y', 'CamRot_Z', 'CamRot_W']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
    return df

def compute_gaze_velocity(df):
    """視線速度を計算（度/秒）"""
    dt = np.diff(df['TS'].values)
    dx = np.diff(df['WorldGazePoint_X'].values)
    dy = np.diff(df['WorldGazePoint_Y'].values)
    dz = np.diff(df['WorldGazePoint_Z'].values)
    
    # 3D距離
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    velocity = dist / dt
    velocity = np.insert(velocity, 0, 0)  # 最初のフレームは0
    return velocity

def compute_pupil_derivative(df):
    """瞳孔径の変化率を計算"""
    dt = np.diff(df['TS'].values)
    d_lp = np.diff(df['LP'].values) / dt
    d_rp = np.diff(df['RP'].values) / dt
    d_lp = np.insert(d_lp, 0, 0)
    d_rp = np.insert(d_rp, 0, 0)
    return d_lp, d_rp

def quaternion_to_euler(qx, qy, qz, qw):
    """クォータニオンからオイラー角（度）に変換"""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def detect_fixations_and_saccades(velocity, threshold=50):
    """固視とサッケードを検出"""
    is_saccade = velocity > threshold
    return is_saccade

def create_visualizations(df, output_dir, subject_name):
    """数理モデル構築のための包括的な可視化を作成"""
    
    # 派生変数の計算
    velocity = compute_gaze_velocity(df)
    d_lp, d_rp = compute_pupil_derivative(df)
    roll, pitch, yaw = quaternion_to_euler(
        df['CamRot_X'].values, df['CamRot_Y'].values,
        df['CamRot_Z'].values, df['CamRot_W'].values
    )
    
    time = df['TS'].values
    
    # ==========================================================
    # Figure 1: 時系列オーバービュー（横軸：時間）
    # ==========================================================
    fig1, axes1 = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    fig1.suptitle(f'時系列オーバービュー - {subject_name}', fontsize=14, fontweight='bold')
    
    # 1. 瞳孔径
    ax = axes1[0]
    ax.plot(time, df['LP'], label='左瞳孔(LP)', color='#2196F3', alpha=0.8, linewidth=0.8)
    ax.plot(time, df['RP'], label='右瞳孔(RP)', color='#F44336', alpha=0.8, linewidth=0.8)
    ax.set_ylabel('瞳孔径 (mm)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('A. 瞳孔径の時間変化', fontsize=11, loc='left')
    
    # 2. 目の開き具合
    ax = axes1[1]
    ax.plot(time, df['EYE_OPENING_L'], label='左目', color='#4CAF50', alpha=0.8, linewidth=0.8)
    ax.plot(time, df['EYE_OPENING_R'], label='右目', color='#FF9800', alpha=0.8, linewidth=0.8)
    ax.set_ylabel('目の開き')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('B. 目の開き具合（まばたき検出用）', fontsize=11, loc='left')
    
    # 3. 視線速度
    ax = axes1[2]
    ax.plot(time, np.clip(velocity, 0, 500), color='#9C27B0', alpha=0.8, linewidth=0.8)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='サッケード閾値')
    ax.set_ylabel('視線速度 (m/s)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('C. 視線速度（サッケード/固視分類用）', fontsize=11, loc='left')
    
    # 4. 頭部回転（オイラー角）
    ax = axes1[3]
    ax.plot(time, yaw, label='Yaw (左右)', color='#00BCD4', alpha=0.8, linewidth=0.8)
    ax.plot(time, pitch, label='Pitch (上下)', color='#E91E63', alpha=0.8, linewidth=0.8)
    ax.set_ylabel('頭部角度 (°)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('D. 頭部回転（VOR解析用）', fontsize=11, loc='left')
    
    # 5. 注視対象ID
    ax = axes1[4]
    ax.scatter(time, df['Gaze_ID'], c=df['Gaze_ID'], cmap='tab10', s=1, alpha=0.5)
    ax.set_ylabel('Gaze ID')
    ax.set_xlabel('時間 (秒)')
    ax.set_title('E. 注視対象オブジェクトID', fontsize=11, loc='left')
    
    plt.tight_layout()
    fig1.savefig(output_dir / f'{subject_name}_01_timeseries_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # ==========================================================
    # Figure 2: 瞳孔動態の詳細解析
    # ==========================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(f'瞳孔動態解析 - {subject_name}', fontsize=14, fontweight='bold')
    
    # 2a. 左右瞳孔の相関
    ax = axes2[0, 0]
    # NaN/Infを除外してプロットと相関計算
    lp_vals = df['LP'].values.astype(float)
    rp_vals = df['RP'].values.astype(float)
    valid_mask = np.isfinite(lp_vals) & np.isfinite(rp_vals)
    if valid_mask.sum() > 2:
        ax.scatter(lp_vals[valid_mask], rp_vals[valid_mask], alpha=0.2, s=5, c=time[valid_mask], cmap='viridis')
        r, p = pearsonr(lp_vals[valid_mask], rp_vals[valid_mask])
    else:
        r = np.nan
    ax.set_xlabel('左瞳孔径 (mm)')
    ax.set_ylabel('右瞳孔径 (mm)')
    ax.set_title(f'A. 左右瞳孔相関 (r={r:.3f})' if not np.isnan(r) else 'A. 左右瞳孔相関 (N/A)', fontsize=11, loc='left')
    ax.plot([df['LP'].min(), df['LP'].max()], [df['LP'].min(), df['LP'].max()], 
            'r--', alpha=0.5, label='y=x')
    ax.legend()
    
    # 2b. 瞳孔径のヒストグラム
    ax = axes2[0, 1]
    ax.hist(df['LP'], bins=50, alpha=0.6, label='左瞳孔', color='#2196F3', density=True)
    ax.hist(df['RP'], bins=50, alpha=0.6, label='右瞳孔', color='#F44336', density=True)
    ax.set_xlabel('瞳孔径 (mm)')
    ax.set_ylabel('確率密度')
    ax.set_title('B. 瞳孔径分布', fontsize=11, loc='left')
    ax.legend()
    ax.text(0.95, 0.95, f'LP: μ={df["LP"].mean():.2f}, σ={df["LP"].std():.2f}\nRP: μ={df["RP"].mean():.2f}, σ={df["RP"].std():.2f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2c. 瞳孔変化率
    ax = axes2[1, 0]
    ax.plot(time[1:100*60], d_lp[1:100*60], alpha=0.7, linewidth=0.5, label='左瞳孔変化率')
    ax.set_xlabel('時間 (秒) - 最初の100秒')
    ax.set_ylabel('瞳孔変化率 (mm/s)')
    ax.set_title('C. 瞳孔変化率（自律神経応答）', fontsize=11, loc='left')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # 2d. 瞳孔径のパワースペクトル密度
    ax = axes2[1, 1]
    fs = 1 / np.mean(np.diff(time))  # サンプリング周波数
    if len(df['LP']) > 256:
        lp_clean = df['LP'].values - df['LP'].mean()
        lp_clean = np.nan_to_num(lp_clean, nan=0.0, posinf=0.0, neginf=0.0)
        rp_clean = df['RP'].values - df['RP'].mean()
        rp_clean = np.nan_to_num(rp_clean, nan=0.0, posinf=0.0, neginf=0.0)
        f, Pxx = signal.welch(lp_clean, fs=fs, nperseg=256)
        ax.semilogy(f, Pxx, color='#2196F3', alpha=0.8, label='左瞳孔')
        f, Pxx = signal.welch(rp_clean, fs=fs, nperseg=256)
        ax.semilogy(f, Pxx, color='#F44336', alpha=0.8, label='右瞳孔')
    ax.set_xlabel('周波数 (Hz)')
    ax.set_ylabel('パワースペクトル密度')
    ax.set_title('D. 瞳孔振動スペクトル', fontsize=11, loc='left')
    ax.legend()
    ax.set_xlim([0, 10])
    
    plt.tight_layout()
    fig2.savefig(output_dir / f'{subject_name}_02_pupil_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # ==========================================================
    # Figure 3: 視線パターン解析
    # ==========================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle(f'視線パターン解析 - {subject_name}', fontsize=14, fontweight='bold')
    
    # 3a. 2D視線軌跡（X-Y平面）
    ax = axes3[0, 0]
    scatter = ax.scatter(df['WorldGazePoint_X'], df['WorldGazePoint_Y'], 
                         c=time, cmap='viridis', s=1, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='時間 (秒)')
    ax.set_xlabel('X座標 (m)')
    ax.set_ylabel('Y座標 (m)')
    ax.set_title('A. 視線軌跡 XY平面', fontsize=11, loc='left')
    
    # 3b. 視線速度のヒストグラム（対数スケール）
    ax = axes3[0, 1]
    velocity_clipped = np.clip(velocity, 0.01, 500)
    ax.hist(velocity_clipped, bins=100, alpha=0.7, color='#9C27B0', density=True)
    ax.axvline(x=50, color='red', linestyle='--', label='サッケード閾値 (50 m/s)')
    ax.set_xlabel('視線速度 (m/s)')
    ax.set_ylabel('確率密度')
    ax.set_title('B. 視線速度分布', fontsize=11, loc='left')
    ax.set_xscale('log')
    ax.legend()
    
    # サッケード/固視の割合を計算
    is_saccade = detect_fixations_and_saccades(velocity)
    saccade_ratio = np.sum(is_saccade) / len(is_saccade) * 100
    ax.text(0.95, 0.95, f'サッケード: {saccade_ratio:.1f}%\n固視: {100-saccade_ratio:.1f}%',
            transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3c. 注視対象別の注視時間
    ax = axes3[1, 0]
    gaze_counts = df['Gaze_ID'].value_counts().sort_index()
    gaze_durations = gaze_counts * np.mean(np.diff(time))  # 秒に変換
    ax.bar(gaze_durations.index, gaze_durations.values, color='#00BCD4', alpha=0.7)
    ax.set_xlabel('Gaze ID')
    ax.set_ylabel('累積注視時間 (秒)')
    ax.set_title('C. オブジェクト別累積注視時間', fontsize=11, loc='left')
    
    # 3d. 視線とヘッドの連動性
    ax = axes3[1, 1]
    # 頭部位置変化
    head_dx = np.diff(df['CamPos_X'].values)
    head_dy = np.diff(df['CamPos_Y'].values)
    head_movement = np.sqrt(head_dx**2 + head_dy**2)
    head_movement = np.insert(head_movement, 0, 0)
    
    ax.scatter(head_movement[::10], velocity[::10], alpha=0.2, s=5)
    ax.set_xlabel('頭部移動速度 (m/frame)')
    ax.set_ylabel('視線速度 (m/s)')
    ax.set_title('D. 頭部-視線速度連動', fontsize=11, loc='left')
    if len(head_movement) > 100:
        valid_mask = np.isfinite(head_movement) & np.isfinite(velocity)
        if valid_mask.sum() > 2:
            r, p = pearsonr(head_movement[valid_mask], velocity[valid_mask])
            ax.text(0.95, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig3.savefig(output_dir / f'{subject_name}_03_gaze_patterns.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # ==========================================================
    # Figure 4: まばたき解析
    # ==========================================================
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle(f'まばたき解析 - {subject_name}', fontsize=14, fontweight='bold')
    
    # まばたき検出（目の開きが閾値以下）
    blink_threshold = 0.5  # 調整可能
    is_blink_l = df['EYE_OPENING_L'] < blink_threshold
    is_blink_r = df['EYE_OPENING_R'] < blink_threshold
    is_blink = is_blink_l | is_blink_r
    
    # 4a. まばたき時の瞳孔挙動（最初の30秒）
    ax = axes4[0, 0]
    t_window = time < 30
    ax.plot(time[t_window], df['EYE_OPENING_L'].values[t_window], 
            label='左目開き', color='#4CAF50', alpha=0.8, linewidth=0.8)
    ax.fill_between(time[t_window], 0, 1, where=is_blink_l.values[t_window], 
                    alpha=0.3, color='red', label='まばたき検出')
    ax.set_xlabel('時間 (秒)')
    ax.set_ylabel('目の開き')
    ax.set_title('A. まばたき検出（最初の30秒）', fontsize=11, loc='left')
    ax.legend(loc='upper right')
    
    # 4b. まばたき間隔（IBI）分布
    ax = axes4[0, 1]
    blink_indices = np.where(np.diff(is_blink.astype(int)) == 1)[0]
    if len(blink_indices) > 1:
        blink_times = time[blink_indices]
        ibi = np.diff(blink_times)
        ax.hist(ibi, bins=30, alpha=0.7, color='#E91E63', density=True)
        ax.set_xlabel('まばたき間隔 (秒)')
        ax.set_ylabel('確率密度')
        ax.axvline(x=np.median(ibi), color='red', linestyle='--', 
                   label=f'中央値: {np.median(ibi):.2f}秒')
        ax.legend()
        ax.text(0.95, 0.95, f'まばたき回数: {len(blink_indices)}\n平均IBI: {np.mean(ibi):.2f}秒',
                transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title('B. まばたき間隔分布', fontsize=11, loc='left')
    
    # 4c. まばたき率の時間変化
    ax = axes4[1, 0]
    window_size = 60  # 60秒ウィンドウ
    blink_rate = []
    time_points = []
    for t_start in np.arange(0, time.max() - window_size, 10):
        mask = (time >= t_start) & (time < t_start + window_size)
        blink_count = np.sum(np.diff(is_blink[mask].astype(int)) == 1)
        blink_rate.append(blink_count / window_size * 60)  # blinks per minute
        time_points.append(t_start + window_size/2)
    ax.plot(time_points, blink_rate, color='#3F51B5', linewidth=1.5)
    ax.set_xlabel('時間 (秒)')
    ax.set_ylabel('まばたき率 (回/分)')
    ax.set_title('C. まばたき率の時間推移', fontsize=11, loc='left')
    ax.axhline(y=np.mean(blink_rate), color='red', linestyle='--', alpha=0.5,
               label=f'平均: {np.mean(blink_rate):.1f}回/分')
    ax.legend()
    
    # 4d. 左右目の開き相関
    ax = axes4[1, 1]
    ax.scatter(df['EYE_OPENING_L'], df['EYE_OPENING_R'], alpha=0.1, s=3)
    valid_mask = np.isfinite(df['EYE_OPENING_L']) & np.isfinite(df['EYE_OPENING_R'])
    if valid_mask.sum() > 2:
        r, p = pearsonr(df['EYE_OPENING_L'][valid_mask], df['EYE_OPENING_R'][valid_mask])
    else:
        r = np.nan
    ax.set_xlabel('左目の開き')
    ax.set_ylabel('右目の開き')
    ax.set_title(f'D. 左右目の開き相関 (r={r:.3f})', fontsize=11, loc='left')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
    ax.legend()
    
    plt.tight_layout()
    fig4.savefig(output_dir / f'{subject_name}_04_blink_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    # ==========================================================
    # Figure 5: 頭部運動解析
    # ==========================================================
    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
    fig5.suptitle(f'頭部運動解析 - {subject_name}', fontsize=14, fontweight='bold')
    
    # 5a. 頭部軌跡 3D
    ax = axes5[0, 0]
    ax.scatter(df['CamPos_X'], df['CamPos_Z'], c=time, cmap='viridis', s=1, alpha=0.5)
    ax.set_xlabel('X位置 (m)')
    ax.set_ylabel('Z位置 (m)')
    ax.set_title('A. 頭部位置軌跡 (XZ平面)', fontsize=11, loc='left')
    
    # 5b. 頭部回転の時間変化
    ax = axes5[0, 1]
    ax.plot(time, yaw, label='Yaw', alpha=0.7, linewidth=0.8)
    ax.plot(time, pitch, label='Pitch', alpha=0.7, linewidth=0.8)
    ax.plot(time, roll, label='Roll', alpha=0.7, linewidth=0.8)
    ax.set_xlabel('時間 (秒)')
    ax.set_ylabel('回転角度 (°)')
    ax.set_title('B. 頭部回転の時間変化', fontsize=11, loc='left')
    ax.legend()
    
    # 5c. 頭部回転速度
    ax = axes5[1, 0]
    dt = np.diff(time)
    yaw_vel = np.abs(np.diff(yaw)) / dt
    yaw_vel = np.insert(yaw_vel, 0, 0)
    ax.plot(time, np.clip(yaw_vel, 0, 100), alpha=0.7, linewidth=0.5)
    ax.set_xlabel('時間 (秒)')
    ax.set_ylabel('Yaw角速度 (°/s)')
    ax.set_title('C. 頭部Yaw角速度', fontsize=11, loc='left')
    
    # 5d. 頭部回転と視線の相関
    ax = axes5[1, 1]
    gaze_angle_x = np.arctan2(df['WorldGazePoint_X'].values, df['WorldGazePoint_Z'].values)
    gaze_angle_x = np.degrees(gaze_angle_x)
    ax.scatter(yaw[::10], gaze_angle_x[::10], alpha=0.2, s=5)
    if len(yaw) > 100:
        valid_mask = np.isfinite(yaw) & np.isfinite(gaze_angle_x)
        if valid_mask.sum() > 2:
            r, p = pearsonr(yaw[valid_mask], gaze_angle_x[valid_mask])
            ax.text(0.95, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('頭部Yaw角 (°)')
    ax.set_ylabel('視線X角度 (°)')
    ax.set_title('D. 頭部-視線角度相関（VOR解析）', fontsize=11, loc='left')
    
    plt.tight_layout()
    fig5.savefig(output_dir / f'{subject_name}_05_head_motion.png', dpi=150, bbox_inches='tight')
    plt.close(fig5)
    
    # ==========================================================
    # Figure 6: 統計サマリー
    # ==========================================================
    fig6, ax = plt.subplots(figsize=(12, 8))
    fig6.suptitle(f'データ統計サマリー - {subject_name}', fontsize=14, fontweight='bold')
    
    # 統計情報をテーブルとして表示
    stats_data = {
        '指標': [
            '記録時間 (秒)',
            'サンプル数',
            'サンプリングレート (Hz)',
            '',
            '左瞳孔径 平均 (mm)',
            '左瞳孔径 標準偏差 (mm)',
            '右瞳孔径 平均 (mm)',
            '右瞳孔径 標準偏差 (mm)',
            '左右瞳孔相関係数',
            '',
            'まばたき検出数',
            'まばたき率 (回/分)',
            '平均まばたき間隔 (秒)',
            '',
            'サッケード割合 (%)',
            '固視割合 (%)',
            '平均視線速度 (m/s)',
            '',
            '頭部Yaw範囲 (°)',
            '頭部Pitch範囲 (°)',
        ],
        '値': [
            f'{time.max():.1f}',
            f'{len(df)}',
            f'{1/np.mean(np.diff(time)):.1f}',
            '',
            f'{df["LP"].mean():.3f}',
            f'{df["LP"].std():.3f}',
            f'{df["RP"].mean():.3f}',
            f'{df["RP"].std():.3f}',
            f'{r:.3f}',  # 前で計算済みの相関係数を再利用
            '',
            f'{len(blink_indices)}',
            f'{len(blink_indices) / (time.max() / 60):.1f}',
            f'{np.mean(ibi):.2f}' if len(blink_indices) > 1 else 'N/A',
            '',
            f'{saccade_ratio:.1f}',
            f'{100-saccade_ratio:.1f}',
            f'{np.mean(velocity):.2f}',
            '',
            f'{yaw.max() - yaw.min():.1f}',
            f'{pitch.max() - pitch.min():.1f}',
        ]
    }
    
    # テーブルを非表示にしてテキストで表示
    ax.axis('off')
    table = ax.table(cellText=list(zip(stats_data['指標'], stats_data['値'])),
                     colLabels=['指標', '値'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # ヘッダーのスタイル
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#E3F2FD')
        elif stats_data['指標'][row-1] == '':
            cell.set_facecolor('#F5F5F5')
    
    plt.tight_layout()
    fig6.savefig(output_dir / f'{subject_name}_06_statistics.png', dpi=150, bbox_inches='tight')
    plt.close(fig6)
    
    print(f"  [OK] 6 visualizations generated: {subject_name}", flush=True)
    
    return {
        'duration': time.max(),
        'samples': len(df),
        'sampling_rate': 1/np.mean(np.diff(time)),
        'lp_mean': df['LP'].mean(),
        'lp_std': df['LP'].std(),
        'rp_mean': df['RP'].mean(),
        'rp_std': df['RP'].std(),
        'pupil_correlation': r,  # 前で計算済みの相関係数を再利用
        'blink_count': len(blink_indices),
        'blink_rate': len(blink_indices) / (time.max() / 60),
        'saccade_ratio': saccade_ratio,
        'velocity_mean': np.mean(velocity),
        'yaw_range': yaw.max() - yaw.min(),
        'pitch_range': pitch.max() - pitch.min(),
    }

def main():
    """メイン処理"""
    base_dir = Path(r'c:\Users\kosuk\ETRA\RTA\RTAデータ')
    output_dir = Path(r'c:\Users\kosuk\ETRA\RTA\visualizations')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("アイトラッキングデータ解析 - 数理モデル構築用可視化")
    print("=" * 60)
    
    all_stats = []
    
    # 各被験者のフォルダを処理
    for subject_folder in base_dir.iterdir():
        if subject_folder.is_dir():
            csv_files = list(subject_folder.glob('*.csv'))
            if csv_files:
                csv_file = csv_files[0]  # 最初のCSVを使用
                print(f"\n処理中: {subject_folder.name}")
                print(f"  ファイル: {csv_file.name}")
                
                try:
                    df = load_eyetracking_data(csv_file)
                    subject_name = subject_folder.name.replace(' ', '_').replace('（', '_').replace('）', '')
                    stats = create_visualizations(df, output_dir, subject_name)
                    stats['subject'] = subject_folder.name
                    all_stats.append(stats)
                except Exception as e:
                    print(f"  [ERROR] {str(e).encode('ascii', errors='replace').decode()}")
    
    # 全被験者の比較可視化
    if all_stats:
        create_group_comparison(all_stats, output_dir)
    
    print("\n" + "=" * 60)
    print(f"完了！可視化は {output_dir} に保存されました")
    print("=" * 60)

def create_group_comparison(all_stats, output_dir):
    """全被験者の比較可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('全被験者比較', fontsize=14, fontweight='bold')
    
    subjects = [s['subject'][:10] for s in all_stats]  # 名前を短縮
    
    # 瞳孔径比較
    ax = axes[0, 0]
    x = np.arange(len(subjects))
    width = 0.35
    ax.bar(x - width/2, [s['lp_mean'] for s in all_stats], width, label='左瞳孔', color='#2196F3', alpha=0.7)
    ax.bar(x + width/2, [s['rp_mean'] for s in all_stats], width, label='右瞳孔', color='#F44336', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.set_ylabel('平均瞳孔径 (mm)')
    ax.set_title('A. 被験者別瞳孔径', fontsize=11, loc='left')
    ax.legend()
    
    # まばたき率比較
    ax = axes[0, 1]
    ax.bar(subjects, [s['blink_rate'] for s in all_stats], color='#E91E63', alpha=0.7)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.set_ylabel('まばたき率 (回/分)')
    ax.set_title('B. 被験者別まばたき率', fontsize=11, loc='left')
    ax.axhline(y=np.mean([s['blink_rate'] for s in all_stats]), color='red', 
               linestyle='--', alpha=0.5, label='全体平均')
    ax.legend()
    
    # サッケード割合比較
    ax = axes[1, 0]
    ax.bar(subjects, [s['saccade_ratio'] for s in all_stats], color='#9C27B0', alpha=0.7)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.set_ylabel('サッケード割合 (%)')
    ax.set_title('C. 被験者別サッケード割合', fontsize=11, loc='left')
    
    # 頭部運動範囲比較
    ax = axes[1, 1]
    ax.bar(x - width/2, [s['yaw_range'] for s in all_stats], width, label='Yaw', color='#00BCD4', alpha=0.7)
    ax.bar(x + width/2, [s['pitch_range'] for s in all_stats], width, label='Pitch', color='#FF9800', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.set_ylabel('回転範囲 (°)')
    ax.set_title('D. 被験者別頭部運動範囲', fontsize=11, loc='left')
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / 'group_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\n  [OK] Group comparison chart generated", flush=True)

if __name__ == '__main__':
    main()
