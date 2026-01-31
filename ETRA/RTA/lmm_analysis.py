"""
Linear Mixed Model (LMM) Analysis for Interest Estimation
==========================================================
線形混合モデルによる興味推定

モデル構造:
  Dwell ~ HeadPosStability + HeadRotStability + PupilSD + (1|Subject) + (1|Object)

固定効果: 頭部安定性、瞳孔変動 → 興味の生理指標
ランダム効果: 被験者間・オブジェクト間の変動を吸収
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'Meiryo'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


def load_and_prepare_data(stats_file):
    """データ読み込みと前処理"""
    df = pd.read_csv(stats_file)
    
    # Gaze_ID > 0 のみ（0は「見ていない」状態）
    df = df[df['gaze_id'] > 0].copy()
    
    # 欠損値処理
    df = df.dropna(subset=['dwell_time_sec', 'head_pos_stability', 
                           'head_rot_stability', 'pupil_std'])
    
    # 変数の標準化（解釈のため）
    for col in ['head_pos_stability', 'head_rot_stability', 'pupil_std']:
        df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()
    
    # Dwell Timeの対数変換（正規性改善）
    df['log_dwell'] = np.log(df['dwell_time_sec'] + 0.1)
    
    # カテゴリ変数
    df['subject'] = df['subject'].astype('category')
    df['gaze_id'] = df['gaze_id'].astype(int).astype('category')
    
    return df


def fit_lmm(df):
    """線形混合モデルをフィット"""
    
    # モデル式
    # Dwell Time ~ 頭部位置安定性 + 頭部回転安定性 + 瞳孔変動 + (1|Subject)
    formula = "log_dwell ~ head_pos_stability_z + head_rot_stability_z + pupil_std_z"
    
    # LMMフィット（被験者をランダム効果）
    model = mixedlm(formula, df, groups=df['subject'])
    result = model.fit(method='powell')
    
    return result


def analyze_and_visualize(df, result, output_dir):
    """結果の分析と可視化"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # ==========================================
    # Figure 1: モデル係数の可視化
    # ==========================================
    fig1, ax = plt.subplots(figsize=(10, 6))
    
    # 固定効果の抽出
    params = result.fe_params
    conf_int = result.conf_int()
    
    # Interceptを除く
    coef_names = ['Head Pos Stability', 'Head Rot Stability', 'Pupil SD']
    coef_vals = [params['head_pos_stability_z'], 
                 params['head_rot_stability_z'], 
                 params['pupil_std_z']]
    ci_low = [conf_int.loc['head_pos_stability_z', 0],
              conf_int.loc['head_rot_stability_z', 0],
              conf_int.loc['pupil_std_z', 0]]
    ci_high = [conf_int.loc['head_pos_stability_z', 1],
               conf_int.loc['head_rot_stability_z', 1],
               conf_int.loc['pupil_std_z', 1]]
    
    y_pos = np.arange(len(coef_names))
    colors = ['#F44336' if v < 0 else '#4CAF50' for v in coef_vals]
    
    ax.barh(y_pos, coef_vals, color=colors, alpha=0.7)
    ax.errorbar(coef_vals, y_pos, xerr=[np.array(coef_vals) - np.array(ci_low), 
                                         np.array(ci_high) - np.array(coef_vals)],
                fmt='none', color='black', capsize=5)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_names)
    ax.set_xlabel('Coefficient (Standardized)')
    ax.set_title('LMM Fixed Effects: Predictors of Dwell Time', fontsize=14, fontweight='bold')
    
    # 係数値をラベル
    for i, (val, name) in enumerate(zip(coef_vals, coef_names)):
        ax.text(val + 0.05 if val > 0 else val - 0.05, i, f'{val:.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontsize=10)
    
    plt.tight_layout()
    fig1.savefig(output_dir / 'lmm_coefficients.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # ==========================================
    # Figure 2: ランダム効果（被験者間変動）
    # ==========================================
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    random_effects = result.random_effects
    subjects = list(random_effects.keys())
    re_values = [random_effects[s]['Group'] for s in subjects]
    
    # ソート
    sorted_idx = np.argsort(re_values)
    subjects = [subjects[i] for i in sorted_idx]
    re_values = [re_values[i] for i in sorted_idx]
    
    colors = ['#F44336' if v < 0 else '#4CAF50' for v in re_values]
    ax.barh(range(len(subjects)), re_values, color=colors, alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([s[:15] for s in subjects], fontsize=8)
    ax.set_xlabel('Random Effect (Subject Deviation)')
    ax.set_title('Subject-level Random Effects', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig2.savefig(output_dir / 'lmm_random_effects.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # ==========================================
    # Figure 3: 予測値 vs 実測値
    # ==========================================
    fig3, ax = plt.subplots(figsize=(8, 8))
    
    df['predicted'] = result.fittedvalues
    
    ax.scatter(df['log_dwell'], df['predicted'], alpha=0.5, s=30)
    
    # 対角線
    min_val = min(df['log_dwell'].min(), df['predicted'].min())
    max_val = max(df['log_dwell'].max(), df['predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect fit')
    
    # R²計算
    ss_res = np.sum((df['log_dwell'] - df['predicted'])**2)
    ss_tot = np.sum((df['log_dwell'] - df['log_dwell'].mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    
    ax.set_xlabel('Observed log(Dwell Time)')
    ax.set_ylabel('Predicted log(Dwell Time)')
    ax.set_title(f'Model Fit: R² = {r2:.3f}', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    fig3.savefig(output_dir / 'lmm_fit.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # ==========================================
    # Figure 4: 変数ごとの関係
    # ==========================================
    fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    vars_to_plot = [
        ('head_pos_stability', 'Head Position Velocity', '#00BCD4'),
        ('head_rot_stability', 'Head Rotation Velocity', '#FF9800'),
        ('pupil_std', 'Pupil SD', '#E91E63')
    ]
    
    for ax, (var, label, color) in zip(axes, vars_to_plot):
        ax.scatter(df[var], df['dwell_time_sec'], alpha=0.5, s=30, color=color)
        
        # 回帰線
        slope, intercept, r, p, se = stats.linregress(df[var], df['dwell_time_sec'])
        x_line = np.linspace(df[var].min(), df[var].max(), 100)
        ax.plot(x_line, intercept + slope * x_line, 'r-', alpha=0.7)
        
        ax.set_xlabel(label)
        ax.set_ylabel('Dwell Time (sec)')
        ax.set_title(f'r = {r:.3f}, p = {p:.4f}')
    
    fig4.suptitle('Bivariate Relationships with Dwell Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig4.savefig(output_dir / 'lmm_bivariate.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    return r2


def derive_adaptive_threshold(df, result):
    """適応的閾値の導出"""
    
    # 予測値を計算
    df['predicted_dwell'] = np.exp(result.fittedvalues) - 0.1
    
    # 閾値候補: 予測値の75パーセンタイル
    threshold_75 = df['predicted_dwell'].quantile(0.75)
    
    # 被験者別の閾値（個人差考慮）
    subject_thresholds = {}
    for subj in df['subject'].unique():
        subj_data = df[df['subject'] == subj]
        subject_thresholds[subj] = subj_data['predicted_dwell'].quantile(0.75)
    
    return threshold_75, subject_thresholds


def main():
    """メイン処理"""
    stats_file = Path(r'c:\Users\kosuk\ETRA\RTA\behavior_analysis\all_subjects_stats.csv')
    output_dir = Path(r'c:\Users\kosuk\ETRA\RTA\lmm_analysis')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Linear Mixed Model Analysis")
    print("=" * 60)
    
    # データ準備
    print("\n1. Loading data...")
    df = load_and_prepare_data(stats_file)
    print(f"   Samples: {len(df)}, Subjects: {df['subject'].nunique()}, Objects: {df['gaze_id'].nunique()}")
    
    # LMMフィット
    print("\n2. Fitting LMM...")
    result = fit_lmm(df)
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(result.summary())
    
    # 可視化
    print("\n3. Generating visualizations...")
    r2 = analyze_and_visualize(df, result, output_dir)
    
    # 適応的閾値
    print("\n4. Deriving adaptive thresholds...")
    threshold, subj_thresholds = derive_adaptive_threshold(df, result)
    print(f"   Global threshold (75th percentile): {threshold:.2f} sec")
    print(f"   Subject-specific thresholds computed")
    
    # 結果をCSV出力
    results_df = pd.DataFrame({
        'Variable': ['Intercept', 'HeadPosStability', 'HeadRotStability', 'PupilSD'],
        'Coefficient': [result.fe_params['Intercept'], 
                       result.fe_params['head_pos_stability_z'],
                       result.fe_params['head_rot_stability_z'],
                       result.fe_params['pupil_std_z']],
        'Std_Error': [result.bse['Intercept'],
                     result.bse['head_pos_stability_z'],
                     result.bse['head_rot_stability_z'],
                     result.bse['pupil_std_z']],
        'p_value': [result.pvalues['Intercept'],
                   result.pvalues['head_pos_stability_z'],
                   result.pvalues['head_rot_stability_z'],
                   result.pvalues['pupil_std_z']]
    })
    results_df.to_csv(output_dir / 'lmm_coefficients.csv', index=False)
    
    # 閾値をCSV出力
    threshold_df = pd.DataFrame({
        'Subject': list(subj_thresholds.keys()),
        'Threshold_sec': list(subj_thresholds.values())
    })
    threshold_df.to_csv(output_dir / 'adaptive_thresholds.csv', index=False)
    
    print("\n" + "=" * 60)
    print(f"Complete! Results saved to {output_dir}")
    print("=" * 60)
    
    # 解釈のサマリー
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    coefs = {
        'head_pos_stability_z': result.fe_params['head_pos_stability_z'],
        'head_rot_stability_z': result.fe_params['head_rot_stability_z'],
        'pupil_std_z': result.fe_params['pupil_std_z']
    }
    
    labels = {
        'head_pos_stability_z': 'HeadPosStability',
        'head_rot_stability_z': 'HeadRotStability',
        'pupil_std_z': 'PupilSD'
    }
    
    for var, coef in coefs.items():
        label = labels[var]
        direction = "LONGER" if coef < 0 else "SHORTER"
        stability = "more stable" if coef < 0 else "less stable"
        if 'pupil' in var:
            stability = "MORE variable" if coef > 0 else "LESS variable"
            direction = "LONGER" if coef > 0 else "SHORTER"
        
        sig = "**" if abs(result.pvalues[var]) < 0.05 else ""
        print(f"   {label}: {direction} dwell when {stability} (beta={coef:.3f}) {sig}")
    
    print(f"\n   Model R²: {r2:.3f}")
    print(f"   Random Effect Variance (Subject): {result.cov_re.iloc[0, 0]:.4f}")


if __name__ == '__main__':
    main()
