"""
Interest Estimation Engine for VR Eye Tracking Data
====================================================
対象物への興味度を推定するエンジン

興味推定に使用する特徴量:
1. Dwell Time (注視時間) - 長い = 興味あり
2. Revisit Count (再訪問回数) - 多い = 興味あり  
3. Pupil Dilation (瞳孔拡大) - 拡大 = 興奮・興味
4. First Fixation Rank (初回注視順位) - 早い = 注意を引く
5. Fixation Count (固視回数) - 多い = 興味あり
6. Transition Probability (遷移確率) - 他から来やすい = 魅力的

出力: 各オブジェクトの Interest Score (0-100)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
matplotlib.rcParams['font.family'] = 'Meiryo'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


class InterestEstimator:
    """対象物への興味度を推定するエンジン"""
    
    def __init__(self, 
                 dwell_weight=0.25,
                 revisit_weight=0.20,
                 pupil_weight=0.20,
                 first_fix_weight=0.15,
                 fixation_count_weight=0.10,
                 transition_weight=0.10):
        """
        重み付けパラメータ（合計=1.0）
        """
        self.weights = {
            'dwell_time': dwell_weight,
            'revisit_count': revisit_weight,
            'pupil_dilation': pupil_weight,
            'first_fixation_rank': first_fix_weight,
            'fixation_count': fixation_count_weight,
            'transition_prob': transition_weight
        }
        self.scaler = MinMaxScaler(feature_range=(0, 100))
        self.features_df = None
        self.interest_scores = None
        
    def load_data(self, file_path):
        """データ読み込みと前処理"""
        df = pd.read_csv(file_path, skiprows=2)
        
        # 数値変換とNaN処理
        numeric_cols = ['LP', 'RP', 'TS', 'Gaze_ID', 'EYE_OPENING_L', 'EYE_OPENING_R']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 瞳孔径の平均
        df['pupil_mean'] = (df['LP'] + df['RP']) / 2
        df['pupil_mean'] = df['pupil_mean'].replace([np.inf, -np.inf], np.nan)
        df['pupil_mean'] = df['pupil_mean'].fillna(df['pupil_mean'].median())
        
        self.df = df
        self.sampling_rate = 1 / np.mean(np.diff(df['TS'].dropna().values[:1000]))
        return df
    
    def extract_object_features(self):
        """各オブジェクト（Gaze_ID）ごとの特徴量を抽出"""
        df = self.df
        
        # 有効なGaze_IDのみ
        valid_gaze = df[df['Gaze_ID'] >= 0].copy()
        unique_objects = valid_gaze['Gaze_ID'].unique()
        
        features = {}
        
        for obj_id in unique_objects:
            obj_data = valid_gaze[valid_gaze['Gaze_ID'] == obj_id]
            
            if len(obj_data) < 5:  # 最低5サンプル
                continue
                
            # 1. Dwell Time（累積注視時間）
            dwell_time = len(obj_data) / self.sampling_rate
            
            # 2. Revisit Count（再訪問回数）
            # Gaze_IDの変化点を検出
            gaze_changes = valid_gaze['Gaze_ID'].diff().fillna(0) != 0
            visits = []
            current_visit = False
            for idx, (gaze, changed) in enumerate(zip(valid_gaze['Gaze_ID'], gaze_changes)):
                if gaze == obj_id:
                    if not current_visit:
                        visits.append(idx)
                        current_visit = True
                else:
                    current_visit = False
            revisit_count = len(visits)
            
            # 3. Pupil Dilation（瞳孔拡大）
            # このオブジェクト注視中の瞳孔径 vs 全体平均
            obj_pupil = obj_data['pupil_mean'].mean()
            baseline_pupil = df['pupil_mean'].mean()
            pupil_dilation = (obj_pupil - baseline_pupil) / baseline_pupil * 100  # %変化
            
            # 4. First Fixation Rank（初回注視順位）
            first_appearance = obj_data.index[0]
            first_fixation_rank = (valid_gaze.index < first_appearance).sum()
            
            # 5. Fixation Count（固視回数）
            # 視線速度が低い期間をカウント
            if len(obj_data) > 10:
                # 簡易的に連続注視区間をカウント
                fixation_count = revisit_count  # 訪問回数≒固視回数として近似
            else:
                fixation_count = 1
            
            # 6. Transition Probability（遷移確率）
            # 他のオブジェクトからこのオブジェクトへの遷移確率
            transitions_to = 0
            prev_gaze = None
            for gaze in valid_gaze['Gaze_ID']:
                if gaze == obj_id and prev_gaze is not None and prev_gaze != obj_id:
                    transitions_to += 1
                prev_gaze = gaze
            total_transitions = gaze_changes.sum()
            transition_prob = transitions_to / total_transitions if total_transitions > 0 else 0
            
            features[obj_id] = {
                'dwell_time': dwell_time,
                'revisit_count': revisit_count,
                'pupil_dilation': pupil_dilation,
                'first_fixation_rank': first_fixation_rank,
                'fixation_count': fixation_count,
                'transition_prob': transition_prob * 100  # %に変換
            }
        
        self.features_df = pd.DataFrame(features).T
        self.features_df.index.name = 'Gaze_ID'
        return self.features_df
    
    def compute_interest_scores(self):
        """興味スコアを計算（0-100）"""
        if self.features_df is None:
            self.extract_object_features()
        
        df = self.features_df.copy()
        
        # 各特徴量を正規化（0-100）
        normalized = pd.DataFrame(index=df.index)
        
        # Dwell Time: 長いほど高スコア
        normalized['dwell_time'] = self._normalize_feature(df['dwell_time'], higher_is_better=True)
        
        # Revisit Count: 多いほど高スコア
        normalized['revisit_count'] = self._normalize_feature(df['revisit_count'], higher_is_better=True)
        
        # Pupil Dilation: 拡大しているほど高スコア
        normalized['pupil_dilation'] = self._normalize_feature(df['pupil_dilation'], higher_is_better=True)
        
        # First Fixation Rank: 早いほど高スコア（順位が低いほど良い）
        normalized['first_fixation_rank'] = self._normalize_feature(df['first_fixation_rank'], higher_is_better=False)
        
        # Fixation Count: 多いほど高スコア
        normalized['fixation_count'] = self._normalize_feature(df['fixation_count'], higher_is_better=True)
        
        # Transition Prob: 高いほど高スコア
        normalized['transition_prob'] = self._normalize_feature(df['transition_prob'], higher_is_better=True)
        
        # 重み付け合計
        interest_score = sum(
            normalized[feat] * weight 
            for feat, weight in self.weights.items()
        )
        
        self.interest_scores = pd.DataFrame({
            'interest_score': interest_score,
            **{f'norm_{k}': v for k, v in normalized.items()},
            **{f'raw_{k}': df[k] for k in df.columns}
        })
        
        return self.interest_scores.sort_values('interest_score', ascending=False)
    
    def _normalize_feature(self, series, higher_is_better=True):
        """特徴量を0-100に正規化"""
        if series.max() == series.min():
            return pd.Series(50, index=series.index)
        
        normalized = (series - series.min()) / (series.max() - series.min()) * 100
        
        if not higher_is_better:
            normalized = 100 - normalized
        
        return normalized
    
    def classify_interest(self, threshold_high=70, threshold_low=30):
        """興味レベルを分類"""
        if self.interest_scores is None:
            self.compute_interest_scores()
        
        scores = self.interest_scores['interest_score']
        
        def classify(score):
            if score >= threshold_high:
                return 'HIGH'
            elif score >= threshold_low:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        self.interest_scores['interest_level'] = scores.apply(classify)
        return self.interest_scores
    
    def visualize_results(self, output_dir, subject_name='subject'):
        """結果の可視化"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if self.interest_scores is None:
            self.classify_interest()
        
        scores = self.interest_scores
        
        # ===========================================
        # Figure 1: 興味スコアランキング
        # ===========================================
        fig1, ax = plt.subplots(figsize=(12, 8))
        
        sorted_scores = scores.sort_values('interest_score', ascending=True)
        colors = {'HIGH': '#4CAF50', 'MEDIUM': '#FFC107', 'LOW': '#F44336'}
        bar_colors = [colors[level] for level in sorted_scores['interest_level']]
        
        y_pos = np.arange(len(sorted_scores))
        ax.barh(y_pos, sorted_scores['interest_score'], color=bar_colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Object {int(idx)}' for idx in sorted_scores.index])
        ax.set_xlabel('Interest Score (0-100)')
        ax.set_title(f'Object Interest Ranking - {subject_name}', fontsize=14, fontweight='bold')
        ax.axvline(x=70, color='green', linestyle='--', alpha=0.5, label='High threshold')
        ax.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='Low threshold')
        ax.legend()
        
        # スコア値を表示
        for i, (idx, row) in enumerate(sorted_scores.iterrows()):
            ax.text(row['interest_score'] + 1, i, f'{row["interest_score"]:.1f}', va='center')
        
        plt.tight_layout()
        fig1.savefig(output_dir / f'{subject_name}_interest_ranking.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        # ===========================================
        # Figure 2: 特徴量レーダーチャート（上位3オブジェクト）
        # ===========================================
        top_objects = scores.nlargest(3, 'interest_score')
        
        if len(top_objects) >= 3:
            fig2, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='polar'))
            fig2.suptitle(f'Feature Profile - Top 3 Objects ({subject_name})', fontsize=14, fontweight='bold')
            
            features = ['dwell_time', 'revisit_count', 'pupil_dilation', 
                       'first_fixation_rank', 'fixation_count', 'transition_prob']
            angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
            angles += angles[:1]  # 閉じる
            
            for ax, (obj_id, row) in zip(axes, top_objects.iterrows()):
                values = [row[f'norm_{f}'] for f in features]
                values += values[:1]  # 閉じる
                
                ax.plot(angles, values, 'o-', linewidth=2, label=f'Object {int(obj_id)}')
                ax.fill(angles, values, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(['Dwell', 'Revisit', 'Pupil', 'First', 'Fixation', 'Trans'], size=9)
                ax.set_ylim(0, 100)
                ax.set_title(f'ID:{int(obj_id)} (Score:{row["interest_score"]:.1f})', size=11)
            
            plt.tight_layout()
            fig2.savefig(output_dir / f'{subject_name}_feature_radar.png', dpi=150, bbox_inches='tight')
            plt.close(fig2)
        
        # ===========================================
        # Figure 3: 時系列での興味オブジェクト遷移
        # ===========================================
        fig3, ax = plt.subplots(figsize=(16, 6))
        
        # Gaze_IDの時系列プロット
        time = self.df['TS'].values
        gaze_ids = self.df['Gaze_ID'].values
        
        # 興味レベルで色分け
        level_map = scores['interest_level'].to_dict()
        colors_timeline = []
        for gid in gaze_ids:
            if gid in level_map:
                colors_timeline.append(colors[level_map[gid]])
            else:
                colors_timeline.append('#CCCCCC')
        
        ax.scatter(time, gaze_ids, c=colors_timeline, s=1, alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Gaze ID (Object)')
        ax.set_title(f'Gaze Timeline with Interest Levels - {subject_name}', fontsize=14, fontweight='bold')
        
        # 凡例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4CAF50', label='HIGH Interest'),
            Patch(facecolor='#FFC107', label='MEDIUM Interest'),
            Patch(facecolor='#F44336', label='LOW Interest'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        fig3.savefig(output_dir / f'{subject_name}_gaze_timeline.png', dpi=150, bbox_inches='tight')
        plt.close(fig3)
        
        # ===========================================
        # Figure 4: 特徴量相関マトリックス
        # ===========================================
        fig4, ax = plt.subplots(figsize=(10, 8))
        
        raw_features = scores[[f'raw_{f}' for f in ['dwell_time', 'revisit_count', 'pupil_dilation', 
                                                     'first_fixation_rank', 'fixation_count', 'transition_prob']]]
        raw_features.columns = ['Dwell Time', 'Revisit', 'Pupil Δ', 'First Rank', 'Fixation', 'Transition']
        
        corr = raw_features.corr()
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)
        
        # 相関値を表示
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', 
                       color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_title(f'Feature Correlation Matrix - {subject_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fig4.savefig(output_dir / f'{subject_name}_feature_correlation.png', dpi=150, bbox_inches='tight')
        plt.close(fig4)
        
        print(f"  [OK] 4 visualizations saved to {output_dir}", flush=True)
        
        return scores
    
    def export_results(self, output_path):
        """結果をCSVにエクスポート"""
        if self.interest_scores is None:
            self.classify_interest()
        
        self.interest_scores.to_csv(output_path)
        print(f"  [OK] Results exported to {output_path}", flush=True)
        return output_path


def main():
    """メイン処理"""
    base_dir = Path(r'c:\Users\kosuk\ETRA\RTA\RTAデータ')
    output_dir = Path(r'c:\Users\kosuk\ETRA\RTA\interest_analysis')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Interest Estimation Engine - VR Eye Tracking")
    print("=" * 60)
    
    all_results = []
    
    for subject_folder in base_dir.iterdir():
        if subject_folder.is_dir():
            csv_files = list(subject_folder.glob('*.csv'))
            if csv_files:
                csv_file = csv_files[0]
                subject_name = subject_folder.name.replace(' ', '_').replace('（', '_').replace('）', '')
                
                print(f"\nProcessing: {subject_folder.name}")
                print(f"  File: {csv_file.name}")
                
                try:
                    # エンジン初期化
                    estimator = InterestEstimator()
                    
                    # データ読み込み
                    estimator.load_data(csv_file)
                    
                    # 特徴量抽出
                    features = estimator.extract_object_features()
                    print(f"  Detected Objects: {len(features)}")
                    
                    # 興味スコア計算
                    scores = estimator.classify_interest()
                    
                    # 可視化
                    estimator.visualize_results(output_dir, subject_name)
                    
                    # CSV出力
                    estimator.export_results(output_dir / f'{subject_name}_interest_scores.csv')
                    
                    # 結果サマリー
                    high_interest = (scores['interest_level'] == 'HIGH').sum()
                    medium_interest = (scores['interest_level'] == 'MEDIUM').sum()
                    low_interest = (scores['interest_level'] == 'LOW').sum()
                    
                    print(f"  Interest Distribution: HIGH={high_interest}, MEDIUM={medium_interest}, LOW={low_interest}")
                    
                    all_results.append({
                        'subject': subject_folder.name,
                        'n_objects': len(scores),
                        'high_count': high_interest,
                        'medium_count': medium_interest,
                        'low_count': low_interest,
                        'top_object': scores['interest_score'].idxmax(),
                        'top_score': scores['interest_score'].max()
                    })
                    
                except Exception as e:
                    print(f"  [ERROR] {str(e).encode('ascii', errors='replace').decode()}")
    
    # 全被験者サマリー
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(output_dir / 'all_subjects_summary.csv', index=False)
        print(f"\n[OK] Summary saved: {output_dir / 'all_subjects_summary.csv'}")
    
    print("\n" + "=" * 60)
    print(f"Complete! Results saved to {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
