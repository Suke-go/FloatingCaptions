"""
Temporal Pattern Statistical Tests
===================================
時系列パターンの統計検定（事後検定付き）
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv(r'c:\Users\kosuk\ETRA\RTA\gaze_pattern_analysis\all_gaze_patterns.csv')
long_eps = df[df['duration'] >= 3].copy()

print('=' * 60)
print('STATISTICAL TESTS: Temporal Pattern')
print(f'N = {len(long_eps)} episodes (duration >= 3 sec)')
print('=' * 60)

results = []

for metric in ['dispersion', 'velocity', 'entropy']:
    print(f'\n{"="*60}')
    print(f'{metric.upper()}')
    print('='*60)
    
    # データ準備
    p1 = long_eps[f'phase_1_{metric}'].dropna().values
    p2 = long_eps[f'phase_2_{metric}'].dropna().values
    p3 = long_eps[f'phase_3_{metric}'].dropna().values
    
    print(f'\nDescriptive Statistics:')
    print(f'  Early:  M={np.mean(p1):.4f}, SD={np.std(p1):.4f}')
    print(f'  Middle: M={np.mean(p2):.4f}, SD={np.std(p2):.4f}')
    print(f'  Late:   M={np.mean(p3):.4f}, SD={np.std(p3):.4f}')
    
    # 1. 全体のANOVA
    f_stat, p_anova = stats.f_oneway(p1, p2, p3)
    sig_anova = '**' if p_anova < 0.05 else ''
    print(f'\nOne-way ANOVA:')
    print(f'  F({2}, {len(p1)+len(p2)+len(p3)-3}) = {f_stat:.3f}, p = {p_anova:.4f} {sig_anova}')
    
    # 2. 事後検定（対応ありt検定 + Bonferroni補正）
    print(f'\nPost-hoc Paired t-tests (Bonferroni corrected, alpha=0.017):')
    
    comparisons = [
        ('Early', 'Middle', p1, p2),
        ('Middle', 'Late', p2, p3),
        ('Early', 'Late', p1, p3)
    ]
    
    for name1, name2, data1, data2 in comparisons:
        min_len = min(len(data1), len(data2))
        t, p = stats.ttest_rel(data1[:min_len], data2[:min_len])
        p_corrected = min(p * 3, 1.0)  # Bonferroni
        
        # 効果量（Cohen's d for paired samples）
        diff = data1[:min_len] - data2[:min_len]
        d = np.mean(diff) / np.std(diff)
        
        sig = '**' if p_corrected < 0.05 else ''
        effect_size = 'large' if abs(d) > 0.8 else ('medium' if abs(d) > 0.5 else 'small')
        
        print(f'  {name1} vs {name2}:')
        print(f'    t({min_len-1}) = {t:.3f}, p_corrected = {p_corrected:.4f} {sig}')
        print(f'    Cohen\'s d = {d:.3f} ({effect_size})')
        
        results.append({
            'metric': metric,
            'comparison': f'{name1} vs {name2}',
            't': t,
            'p_raw': p,
            'p_corrected': p_corrected,
            'cohens_d': d,
            'significant': p_corrected < 0.05
        })

# サマリー表
print('\n' + '='*60)
print('SUMMARY TABLE')
print('='*60)
results_df = pd.DataFrame(results)
for metric in ['dispersion', 'velocity', 'entropy']:
    metric_results = results_df[results_df['metric'] == metric]
    sig_pairs = metric_results[metric_results['significant']]['comparison'].tolist()
    print(f'\n{metric.upper()}:')
    if sig_pairs:
        print(f'  Significant pairs: {", ".join(sig_pairs)}')
    else:
        print(f'  No significant differences')

results_df.to_csv(r'c:\Users\kosuk\ETRA\RTA\gaze_pattern_analysis\posthoc_results.csv', index=False)
print(f'\nResults saved to posthoc_results.csv')
