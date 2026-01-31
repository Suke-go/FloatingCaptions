"""
Lightweight HMM Attention Detector
==================================
Unity C#移植可能な軽量HMM実装

特徴:
- 純粋なNumPy実装（hmmlearnなし）
- Forward algorithmのみ（リアルタイム推論用）
- JSON/CSVエクスポートでUnity連携

使用法:
    # 訓練
    detector = LightweightAttentionHMM()
    detector.fit(training_features, episode_lengths)
    detector.export_for_unity("hmm_params.json")
    
    # 推論
    state, confidence = detector.predict_step(current_features)
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AttentionState:
    """注意状態の定義"""
    GLANCING = 0
    OVERVIEW = 1
    DEEP_ENGAGEMENT = 2
    
    NAMES = ['Glancing', 'Overview', 'DeepEngagement']


class LightweightAttentionHMM:
    """
    軽量HMM実装（Unity移植可能）
    
    状態:
        0: Glancing (一瞥)
        1: Overview (概観)
        2: Deep Engagement (深い興味) → トリガー対象
    
    観測特徴量 (4次元):
        - head_rot_velocity: 頭部回転速度 (rad/s)
        - gaze_dispersion: 視線分散 (m)
        - spatial_entropy: 空間エントロピー (0-1)
        - pupil_delta: 瞳孔拡張 (mm)
    """
    
    def __init__(self, n_states: int = 3, n_features: int = 4):
        self.n_states = n_states
        self.n_features = n_features
        
        # 初期状態確率
        self.pi = np.array([0.7, 0.2, 0.1])
        
        # 状態遷移行列 (事前知識に基づく)
        self.A = np.array([
            [0.7, 0.25, 0.05],   # Glancing → ...
            [0.2, 0.5, 0.3],     # Overview → ...
            [0.1, 0.3, 0.6],     # DeepEngagement → ...
        ])
        
        # 各状態の観測モデル (ガウス分布)
        # means: [n_states, n_features]
        self.means = np.array([
            [0.5, 0.8, 0.3, 0.0],    # Glancing: 高速頭部動、広い分散、低エントロピー
            [0.2, 0.5, 0.4, 0.1],    # Overview: 減速開始、中程度
            [0.05, 0.2, 0.6, 0.23],  # DeepEngagement: 安定、高エントロピー、瞳孔拡張
        ])
        
        # 標準偏差: [n_states, n_features]
        self.stds = np.array([
            [0.2, 0.3, 0.1, 0.1],
            [0.1, 0.2, 0.1, 0.1],
            [0.05, 0.1, 0.1, 0.1],
        ])
        
        # 特徴量正規化パラメータ
        self.feature_means = np.zeros(n_features)
        self.feature_stds = np.ones(n_features)
        
        # リアルタイム推論用の状態
        self.alpha = self.pi.copy()  # Forward変数
        
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """特徴量を正規化"""
        return (X - self.feature_means) / (self.feature_stds + 1e-8)
    
    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
        """多変量ガウス確率密度（対角共分散）"""
        log_prob = -0.5 * np.sum(((x - mean) / (std + 1e-8)) ** 2)
        log_prob -= 0.5 * self.n_features * np.log(2 * np.pi)
        log_prob -= np.sum(np.log(std + 1e-8))
        return np.exp(np.clip(log_prob, -500, 0))
    
    def _emission_prob(self, x: np.ndarray) -> np.ndarray:
        """観測確率 P(x|state) を計算"""
        probs = np.zeros(self.n_states)
        for k in range(self.n_states):
            probs[k] = self._gaussian_pdf(x, self.means[k], self.stds[k])
        return probs + 1e-10  # ゼロ除算防止
    
    def fit(self, X: np.ndarray, lengths: List[int], n_iter: int = 50) -> 'LightweightAttentionHMM':
        """
        Baum-Welch (EM) アルゴリズムで訓練
        
        Args:
            X: 特徴量行列 [n_total_samples, n_features]
            lengths: 各エピソードのサンプル数
            n_iter: 反復回数
        """
        # 正規化パラメータを計算
        self.feature_means = np.nanmean(X, axis=0)
        self.feature_stds = np.nanstd(X, axis=0) + 1e-8
        
        X_norm = self._normalize_features(X)
        X_norm = np.nan_to_num(X_norm, nan=0.0)
        
        # エピソードに分割
        episodes = []
        start = 0
        for length in lengths:
            if length > 5:  # 短すぎるエピソードは除外
                episodes.append(X_norm[start:start+length])
            start += length
        
        print(f"Training on {len(episodes)} episodes...")
        
        for iteration in range(n_iter):
            # E-step: Forward-Backward
            gamma_sum = np.zeros((self.n_states, self.n_features))
            gamma_sqsum = np.zeros((self.n_states, self.n_features))
            gamma_count = np.zeros(self.n_states)
            xi_sum = np.zeros((self.n_states, self.n_states))
            
            total_log_likelihood = 0
            
            for obs in episodes:
                T = len(obs)
                
                # Forward
                alpha = np.zeros((T, self.n_states))
                alpha[0] = self.pi * self._emission_prob(obs[0])
                alpha[0] /= alpha[0].sum() + 1e-10
                
                for t in range(1, T):
                    for j in range(self.n_states):
                        alpha[t, j] = self._emission_prob(obs[t])[j] * np.sum(alpha[t-1] * self.A[:, j])
                    alpha[t] /= alpha[t].sum() + 1e-10
                
                # Backward
                beta = np.zeros((T, self.n_states))
                beta[-1] = 1.0
                
                for t in range(T-2, -1, -1):
                    for i in range(self.n_states):
                        beta[t, i] = np.sum(self.A[i] * self._emission_prob(obs[t+1]) * beta[t+1])
                    beta[t] /= beta[t].sum() + 1e-10
                
                # Gamma and Xi
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10
                
                for t in range(T):
                    for k in range(self.n_states):
                        gamma_sum[k] += gamma[t, k] * obs[t]
                        gamma_sqsum[k] += gamma[t, k] * (obs[t] ** 2)
                        gamma_count[k] += gamma[t, k]
                
                for t in range(T-1):
                    xi = np.outer(alpha[t], self._emission_prob(obs[t+1]) * beta[t+1])
                    xi *= self.A
                    xi /= xi.sum() + 1e-10
                    xi_sum += xi
                
                total_log_likelihood += np.log(alpha[-1].sum() + 1e-10)
            
            # M-step
            # 遷移行列更新
            self.A = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-10)
            
            # 観測モデル更新
            for k in range(self.n_states):
                if gamma_count[k] > 1:
                    self.means[k] = gamma_sum[k] / gamma_count[k]
                    variance = gamma_sqsum[k] / gamma_count[k] - self.means[k] ** 2
                    self.stds[k] = np.sqrt(np.maximum(variance, 0.01))
            
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: log-likelihood = {total_log_likelihood:.2f}")
        
        print("Training complete.")
        return self
    
    def reset_forward(self):
        """リアルタイム推論のリセット"""
        self.alpha = self.pi.copy()
    
    def predict_step(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        1フレーム分のリアルタイム推論（Forward step）
        
        Args:
            x: 特徴量ベクトル [n_features]
        
        Returns:
            (predicted_state, state_probabilities)
        """
        x_norm = self._normalize_features(x)
        x_norm = np.nan_to_num(x_norm, nan=0.0)
        
        # Forward更新
        emission = self._emission_prob(x_norm)
        alpha_new = np.zeros(self.n_states)
        
        for j in range(self.n_states):
            alpha_new[j] = emission[j] * np.sum(self.alpha * self.A[:, j])
        
        # 正規化
        alpha_new /= alpha_new.sum() + 1e-10
        self.alpha = alpha_new
        
        return np.argmax(self.alpha), self.alpha.copy()
    
    def should_trigger(self, threshold: float = 0.6) -> Tuple[bool, float]:
        """
        キャプション表示をトリガーすべきか判定
        
        Returns:
            (should_trigger, confidence)
        """
        de_prob = self.alpha[AttentionState.DEEP_ENGAGEMENT]
        return de_prob > threshold, de_prob
    
    def export_for_unity(self, output_path: str):
        """
        Unity C#用にパラメータをJSON出力
        """
        params = {
            'n_states': self.n_states,
            'n_features': self.n_features,
            'pi': self.pi.tolist(),
            'A': self.A.tolist(),
            'means': self.means.tolist(),
            'stds': self.stds.tolist(),
            'feature_means': self.feature_means.tolist(),
            'feature_stds': self.feature_stds.tolist(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"Exported HMM parameters to {output_path}")
    
    @classmethod
    def load_from_json(cls, path: str) -> 'LightweightAttentionHMM':
        """JSONからモデルを読み込み"""
        with open(path, 'r') as f:
            params = json.load(f)
        
        model = cls(params['n_states'], params['n_features'])
        model.pi = np.array(params['pi'])
        model.A = np.array(params['A'])
        model.means = np.array(params['means'])
        model.stds = np.array(params['stds'])
        model.feature_means = np.array(params['feature_means'])
        model.feature_stds = np.array(params['feature_stds'])
        
        return model


# ============================================
# 訓練用ヘルパー関数
# ============================================

def load_rta_data(data_dir: Path) -> Tuple[np.ndarray, List[int]]:
    """
    RTAデータから訓練用特徴量を抽出
    
    Returns:
        X: 特徴量行列 [n_total_frames, 4]
        lengths: 各エピソードのフレーム数
    """
    import pandas as pd
    
    all_features = []
    lengths = []
    
    for subject_dir in data_dir.iterdir():
        if not subject_dir.is_dir():
            continue
        
        csv_files = list(subject_dir.glob('*.csv'))
        if not csv_files:
            continue
        
        csv_file = csv_files[0]
        print(f"Loading: {subject_dir.name}")
        
        try:
            df = pd.read_csv(csv_file, skiprows=2)
            
            # 必要なカラムの存在確認
            required = ['TS', 'Gaze_ID', 'LP', 'RP', 
                       'CamRot_X', 'CamRot_Y', 'CamRot_Z', 'CamRot_W',
                       'WorldGazePoint_X', 'WorldGazePoint_Y', 'WorldGazePoint_Z']
            
            if not all(col in df.columns for col in required):
                print(f"  Skipping: missing columns")
                continue
            
            # 数値変換
            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # エピソード抽出
            df['gaze_change'] = df['Gaze_ID'].diff().fillna(1) != 0
            df['episode_id'] = df['gaze_change'].cumsum()
            
            for episode_id, group in df.groupby('episode_id'):
                if len(group) < 30:  # 0.5秒未満は除外
                    continue
                
                gaze_id = group['Gaze_ID'].iloc[0]
                if pd.isna(gaze_id) or gaze_id <= 0:
                    continue
                
                # 特徴量計算
                features = compute_frame_features(group)
                if features is not None and len(features) >= 10:
                    all_features.append(features)
                    lengths.append(len(features))
        
        except Exception as e:
            print(f"  Error: {str(e)[:50]}")
    
    if all_features:
        X = np.vstack(all_features)
        return X, lengths
    else:
        return np.array([]), []


def compute_frame_features(df) -> Optional[np.ndarray]:
    """
    フレームごとの特徴量を計算
    
    Returns:
        features: [n_frames, 4] or None
    """
    try:
        n = len(df)
        features = np.zeros((n, 4))
        
        # 1. 頭部回転速度
        qw = df['CamRot_W'].values
        dt = np.diff(df['TS'].values)
        dt[dt == 0] = 0.001
        
        # クォータニオンから角速度
        theta = 2 * np.arccos(np.clip(np.abs(qw[1:]), 0, 1))
        omega = theta / dt
        omega = np.insert(omega, 0, omega[0] if len(omega) > 0 else 0.1)
        features[:, 0] = omega
        
        # 2. 視線分散（スライディングウィンドウ）
        gaze_x = df['WorldGazePoint_X'].values
        gaze_y = df['WorldGazePoint_Y'].values
        window = 30  # 0.5秒
        
        for i in range(n):
            start = max(0, i - window)
            disp_x = np.nanstd(gaze_x[start:i+1])
            disp_y = np.nanstd(gaze_y[start:i+1])
            features[i, 1] = (disp_x + disp_y) / 2
        
        # 3. 空間エントロピー（スライディングウィンドウ）
        for i in range(n):
            start = max(0, i - window)
            x_seg = gaze_x[start:i+1]
            y_seg = gaze_y[start:i+1]
            
            if len(x_seg) >= 5:
                hist, _, _ = np.histogram2d(x_seg, y_seg, bins=5)
                hist = hist.flatten()
                hist = hist[hist > 0]
                if len(hist) > 0:
                    prob = hist / hist.sum()
                    entropy = -np.sum(prob * np.log2(prob + 1e-10))
                    features[i, 2] = entropy / np.log2(25)  # 正規化
        
        # 4. 瞳孔拡張
        pupil = (df['LP'].values + df['RP'].values) / 2
        baseline = np.nanmean(pupil[:30])  # 最初の0.5秒
        features[:, 3] = pupil - baseline
        
        # NaN処理
        features = np.nan_to_num(features, nan=0.0)
        
        return features
    
    except Exception as e:
        print(f"  Feature computation error: {e}")
        return None


# ============================================
# メイン処理
# ============================================

def main():
    """訓練とエクスポート"""
    data_dir = Path(r'c:\Users\kosuk\ETRA\RTA\RTAデータ')
    output_dir = Path(r'c:\Users\kosuk\ETRA\RTA')
    
    print("=" * 60)
    print("Lightweight HMM Training for Unity")
    print("=" * 60)
    
    # データ読み込み
    print("\n1. Loading RTA data...")
    X, lengths = load_rta_data(data_dir)
    
    if len(X) == 0:
        print("No data loaded!")
        return
    
    print(f"   Total samples: {len(X)}")
    print(f"   Episodes: {len(lengths)}")
    print(f"   Feature shape: {X.shape}")
    
    # 訓練
    print("\n2. Training HMM...")
    model = LightweightAttentionHMM()
    model.fit(X, lengths, n_iter=50)
    
    # 結果表示
    print("\n3. Learned parameters:")
    print("   Transition matrix A:")
    print(model.A)
    print("\n   State means:")
    for i, name in enumerate(AttentionState.NAMES):
        print(f"     {name}: {model.means[i]}")
    
    # エクスポート
    print("\n4. Exporting for Unity...")
    model.export_for_unity(output_dir / 'hmm_params.json')
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
