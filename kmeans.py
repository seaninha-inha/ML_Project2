"""
K-Means Clustering Analysis (v2)
--------------------------------------------
v1 대비 변경사항:
  [Fix 1] Davies-Bouldin Index 추가 (유지)
  [Fix 2] Composite score 기반 K 추천 (유지)
  [Fix 3] BEST_K 명시적 오버라이드 블록 (유지)
  [Fix 4] Auto cluster naming (유지)
  [Fix 5] High-severity (Severity 3+4) 비율 출력 (유지)
  [Fix 6] BEST_K 기본값 변경: elbow_k -> 8 (해석 가능성 우선)
          K 민감도 비교 테스트 블록 추가 (K=7,8,10,12)
          season 피처 반영된 cluster naming 업데이트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
print("Loading data...")
X = pd.read_csv('kmeans_input.csv')
df_full = pd.read_csv('preprocessed_full.csv')
df_full = df_full.iloc[:len(X)].reset_index(drop=True)
print(f"Input shape: {X.shape}")


# ─────────────────────────────────────────────
# 2. PCA Preprocessing (remove one-hot noise)
# ─────────────────────────────────────────────
print("\nApplying PCA for dimensionality reduction...")
pca_full = PCA(random_state=42)
pca_full.fit(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cumvar >= 0.80) + 1
print(f"Components needed to explain 80% variance: {n_components}")

pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X)
print(f"Reduced shape: {X_pca.shape}")
print(f"Total explained variance: {cumvar[n_components-1]*100:.1f}%")


# ─────────────────────────────────────────────
# 3. Optimal K Search (Elbow + Silhouette + Davies-Bouldin)
# ─────────────────────────────────────────────
def find_elbow(k_list, inertia_list):
    """Kneedle algorithm: point farthest from diagonal = elbow."""
    k_arr = np.array(k_list, dtype=float)
    i_arr = np.array(inertia_list, dtype=float)
    k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min())
    i_norm = (i_arr - i_arr.min()) / (i_arr.max() - i_arr.min())
    a = i_norm[-1] - i_norm[0]
    b = k_norm[0]  - k_norm[-1]
    c = k_norm[-1] * i_norm[0] - k_norm[0] * i_norm[-1]
    distances = np.abs(a * k_norm + b * i_norm + c) / np.sqrt(a**2 + b**2)
    return int(k_arr[np.argmax(distances)])

K_MAX = 20
print(f"\nSearching for optimal K (K=2~{K_MAX})...")
K_RANGE    = range(2, K_MAX + 1)
inertias   = []
sil_scores = []
db_scores  = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)

    sample_size = min(10000, len(X_pca))
    idx = np.random.choice(len(X_pca), sample_size, replace=False)
    sil_scores.append(silhouette_score(X_pca[idx], labels[idx]))
    db_scores.append(davies_bouldin_score(X_pca[idx], labels[idx]))

    print(f"  K={k:2d}  inertia={km.inertia_:.0f}  "
          f"silhouette={sil_scores[-1]:.4f}  db={db_scores[-1]:.4f}")

elbow_k    = find_elbow(list(K_RANGE), inertias)
best_k_sil = list(K_RANGE)[sil_scores.index(max(sil_scores))]
best_k_db  = list(K_RANGE)[db_scores.index(min(db_scores))]

print(f"\n-> Elbow point     : K = {elbow_k}")
print(f"-> Best Silhouette : K = {best_k_sil}  ({max(sil_scores):.4f})")
print(f"-> Best DB Score   : K = {best_k_db}  ({min(db_scores):.4f})")


# ─────────────────────────────────────────────
# [Fix 2] Auto K recommendation via composite score
# ─────────────────────────────────────────────
votes = {}
for k in [elbow_k, best_k_sil, best_k_db]:
    votes[k] = votes.get(k, 0) + 1

recommended_k = max(votes, key=lambda k: (votes[k], -abs(k - 5)))

sil_arr = np.array(sil_scores)
is_sil_monotone = np.all(np.diff(sil_arr[-5:]) >= 0)

print(f"\n[Fix 2] K recommendation vote: {votes}")
if is_sil_monotone:
    print("  WARNING: Monotone silhouette -> No strong natural clusters")
print(f"  -> Recommended K: {recommended_k}")


# ─────────────────────────────────────────────
# Output 1: 4-panel K selection plot
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 4))

ax1 = axes[0]
ax1.plot(list(K_RANGE), inertias, 'o-', color='steelblue', lw=2, ms=5)
ax1.axvline(elbow_k, color='red', ls='--', lw=1.5, label=f'Elbow K={elbow_k}')
ax1.set_title('Elbow Method (K=2~20)', fontsize=12)
ax1.set_xlabel('K'); ax1.set_ylabel('Inertia (SSE)')
ax1.legend(); ax1.grid(axis='y', alpha=0.3)

ax2 = axes[1]
k_zoom = list(K_RANGE)[:9]
ax2.plot(k_zoom, inertias[:9], 'o-', color='steelblue', lw=2, ms=7)
ax2.axvline(elbow_k, color='red', ls='--', lw=1.5, label=f'Elbow K={elbow_k}')
ax2.set_title('Elbow Zoom (K=2~10)', fontsize=12)
ax2.set_xlabel('K'); ax2.set_ylabel('Inertia (SSE)')
ax2.legend(); ax2.grid(axis='y', alpha=0.3)

ax3 = axes[2]
ax3.plot(list(K_RANGE), sil_scores, 's-', color='coral', lw=2, ms=5)
ax3.axvline(best_k_sil, color='gray', ls='--', lw=1.5, label=f'Best Sil K={best_k_sil}')
ax3.axhline(0, color='black', ls=':', alpha=0.4)
if is_sil_monotone:
    ax3.text(0.05, 0.95, 'WARNING: Monotone\n-> Weak cluster structure',
             transform=ax3.transAxes, fontsize=9, va='top', color='firebrick')
ax3.set_title('Silhouette Score (K=2~20)', fontsize=12)
ax3.set_xlabel('K'); ax3.set_ylabel('Silhouette Score')
ax3.legend(); ax3.grid(axis='y', alpha=0.3)

ax4 = axes[3]
ax4.plot(list(K_RANGE), db_scores, '^-', color='mediumseagreen', lw=2, ms=5)
ax4.axvline(best_k_db, color='darkgreen', ls='--', lw=1.5, label=f'Best DB K={best_k_db}')
ax4.set_title('Davies-Bouldin Score\n(lower is better)', fontsize=12)
ax4.set_xlabel('K'); ax4.set_ylabel('DB Score')
ax4.legend(); ax4.grid(axis='y', alpha=0.3)

plt.suptitle(
    f'Optimal K Analysis  |  Recommended K={recommended_k}'
    f'  (Elbow={elbow_k}, Sil={best_k_sil}, DB={best_k_db})',
    fontsize=12, y=1.02
)
plt.tight_layout()
plt.savefig('output_1_optimal_k_k8.png', dpi=150, bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────
# [Fix 3 / Fix 6] BEST_K Decision
# ─────────────────────────────────────────────
# ┌──────────────────────────────────────────────────────────────────┐
# │  K Selection Rationale                                           │
# │                                                                  │
# │  v2 기본값: BEST_K = 8  (해석 가능성 우선)                        │
# │                                                                  │
# │  변경 이유 (v1 K=12 결과 분석):                                   │
# │    1) 실루엣이 K=2~20 단조 증가 -> 자연 군집 구조 약함             │
# │       K=12 는 알고리즘이 강제 분해한 것이지 데이터 고유 구조 아님    │
# │    2) C11 Precipitation 평균 9.95in -> 이상치 클러스터 흡수 의심   │
# │       Fix 4(캡핑) 적용 후 C11 이 흡수하는 이상치 양 감소 기대       │
# │    3) 도메인 지식: 날씨(맑음/비/눈/안개) × 시간(낮/밤) = 4~8개     │
# │                                                                  │
# │  K=12 로 되돌리려면:                                              │
# │    BEST_K = elbow_k   또는   BEST_K = 12                        │
# └──────────────────────────────────────────────────────────────────┘

BEST_K = 8   # [Fix 6] was: elbow_k  |  해석 가능성 우선 오버라이드
K_REASON = (
    f"[Fix 6] Manual K={BEST_K} (interpretability priority). "   # elbow_k -> BEST_K 로 수정
    f"Elbow={elbow_k}, DB_best={best_k_db}, Sil monotone (best at K={best_k_sil} invalid). "
    f"Domain: weather x time-of-day -> 4~8 clusters natural."
)

# ────────────────────────────────────────────────────────────────
# 대안 오버라이드 (주석 해제해서 사용):
# BEST_K = elbow_k      # 알고리즘 자동 탐지 (Kneedle)
# BEST_K = best_k_db    # DB Score 최소 -> 클러스터 응집도 우선
# BEST_K = recommended_k # 3개 지표 다수결
# BEST_K = 12           # v1 결과와 직접 비교 시
# ────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"[Fix 3+6] Final selected K = {BEST_K}")
print(f"  Reason: {K_REASON}")
print(f"{'='*60}")


# ─────────────────────────────────────────────
# [Fix 6] K 민감도 비교 테스트 (K=7, 8, 10, 12)
#
# 목적: K 값에 따라 최고위험 클러스터 비율이 어떻게 변하는지 확인
#       BEST_K 선택의 근거 보완
# ─────────────────────────────────────────────
SENSITIVITY_K_LIST = [7, 8, 10, 12]
print(f"\n[Fix 6] K sensitivity test: {SENSITIVITY_K_LIST}")
print(f"  {'K':>4}  {'max_high_sev':>13}  {'min_high_sev':>13}  {'spread':>8}  {'db_score':>10}")
print(f"  {'─'*55}")

sensitivity_results = []
for test_k in SENSITIVITY_K_LIST:
    km_test = KMeans(n_clusters=test_k, random_state=42, n_init=10)
    labels_test = km_test.fit_predict(X_pca)
    df_full['_test_cluster'] = labels_test

    # DB score
    idx_s = np.random.choice(len(X_pca), min(10000, len(X_pca)), replace=False)
    db_s  = davies_bouldin_score(X_pca[idx_s], labels_test[idx_s])

    high_sev_rates = []
    if 'Severity' in df_full.columns:
        for c in range(test_k):
            mask = df_full['_test_cluster'] == c
            rate = (df_full.loc[mask, 'Severity'] >= 3).mean() * 100
            high_sev_rates.append(rate)

    max_hs  = max(high_sev_rates) if high_sev_rates else float('nan')
    min_hs  = min(high_sev_rates) if high_sev_rates else float('nan')
    spread  = max_hs - min_hs
    marker  = ' <-- BEST_K' if test_k == BEST_K else ''
    print(f"  K={test_k:2d}  {max_hs:>12.1f}%  {min_hs:>12.1f}%  {spread:>7.1f}%  {db_s:>10.4f}{marker}")

    sensitivity_results.append({
        'K': test_k, 'max_high_sev': max_hs, 'min_high_sev': min_hs,
        'spread': spread, 'db_score': db_s
    })
    df_full.drop(columns='_test_cluster', inplace=True)

pd.DataFrame(sensitivity_results).to_csv('output_k_sensitivity_k8.csv', index=False)
print("  -> Saved: output_k_sensitivity.csv")

# K 민감도 비교 차트
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ks     = [r['K']        for r in sensitivity_results]
max_hs = [r['max_high_sev'] for r in sensitivity_results]
spread = [r['spread']   for r in sensitivity_results]
dbs    = [r['db_score'] for r in sensitivity_results]

colors_bar = ['crimson' if k == BEST_K else 'steelblue' for k in ks]

axes[0].bar([str(k) for k in ks], max_hs, color=colors_bar, edgecolor='white')
axes[0].set_title('Max High-Severity Rate per K (%)\n(lower = more balanced)', fontsize=11)
axes[0].set_xlabel('K'); axes[0].set_ylabel('Max High Severity Rate (%)')
axes[0].axhline(np.mean(max_hs), color='gray', ls='--', lw=1.2, label='Average')
axes[0].legend()
for i, (k, v) in enumerate(zip(ks, max_hs)):
    axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=10)

ax_db = axes[1].twinx()
l1, = axes[1].plot([str(k) for k in ks], spread, 'o-', color='coral', lw=2, ms=8, label='Spread')
l2, = ax_db.plot([str(k) for k in ks], dbs, 's--', color='mediumseagreen', lw=2, ms=8, label='DB Score')
axes[1].set_title('High-Severity Spread & DB Score\nSpread = max - min per K', fontsize=11)
axes[1].set_xlabel('K')
axes[1].set_ylabel('Severity Spread (%)', color='coral')
ax_db.set_ylabel('DB Score (lower=better)', color='mediumseagreen')
axes[1].legend(handles=[l1, l2], loc='upper right')

plt.suptitle(f'K Sensitivity Comparison  |  Selected BEST_K={BEST_K}', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('output_0_k_sensitivity_k8.png', dpi=150, bbox_inches='tight')
plt.show()
print("-> Saved: output_0_k_sensitivity.png")


# ─────────────────────────────────────────────
# 4. Final K-Means Run
# ─────────────────────────────────────────────
print(f"\nRunning K-Means with K={BEST_K}...")
km_final = KMeans(n_clusters=BEST_K, random_state=42, n_init=15)
df_full['cluster'] = km_final.fit_predict(X_pca)

print("Sample count per cluster:")
print(df_full['cluster'].value_counts().sort_index())

df_full[['cluster']].to_csv('kmeans_cluster_labels.csv')
print("-> Saved: kmeans_cluster_labels.csv")

# ── 소형 클러스터 체크 ──────────────────────────────────
MIN_CLUSTER_RATIO = 0.005  # 전체의 0.5% 미만이면 경고
total_n = len(df_full)
small_clusters = [
    (c, int((df_full['cluster'] == c).sum()))
    for c in range(BEST_K)
    if (df_full['cluster'] == c).sum() / total_n < MIN_CLUSTER_RATIO
]
if small_clusters:
    print("\n*** WARNING: 소형 클러스터 감지 (< 0.5%) — K를 줄이거나 아웃라이어 처리 필요 ***")
    for c, sz in small_clusters:
        print(f"  Cluster {c}: n={sz}  ({sz/total_n*100:.2f}%)")
    print(f"  권장: K={BEST_K - len(small_clusters)} 재실험 또는 해당 클러스터 제외 후 분석")
else:
    print(f"  OK: 모든 클러스터 >= {MIN_CLUSTER_RATIO*100:.1f}% (n >= {int(total_n*MIN_CLUSTER_RATIO):,})")


# ─────────────────────────────────────────────
# 5. Cluster Summary Table
# ─────────────────────────────────────────────
SUMMARY_COLS = [
    'Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
    'Wind_Speed(mph)', 'Precipitation(in)',
    'hour', 'is_night', 'is_weekend',
    'is_low_visibility', 'is_heavy_precip',
    'season',                               # [Fix 6] season 추가
    'Junction', 'Traffic_Signal', 'Crossing',
    'Severity', 'cluster'
]
available = [c for c in SUMMARY_COLS if c in df_full.columns]
cluster_summary = df_full[available].groupby('cluster').mean().round(2)
cluster_summary['count'] = df_full['cluster'].value_counts().sort_index()

print("\n[Cluster Summary]")
print(cluster_summary.to_string())
cluster_summary.to_csv('output_kmeans_summary_k8.csv')


# ─────────────────────────────────────────────
# [Fix 4] Auto Cluster Naming
#   season 피처 반영 업데이트 [Fix 6]
# ─────────────────────────────────────────────
SEASON_LABELS = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Autumn'}

def name_cluster(row):
    traits = []

    # 온도
    temp = row.get('Temperature(F)', 60)
    if temp < 35:
        traits.append("Freezing")
    elif temp < 50:
        traits.append("Cool")
    elif temp >= 70:
        traits.append("Hot")

    # 계절 [Fix 6] - season 평균이 뚜렷한 경우만 표시
    # 클러스터 평균 season: 0=Winter, 1=Spring, 2=Summer, 3=Autumn
    # 0~0.5 -> Winter 우세, 2.5~3 -> Autumn 우세
    season_val = row.get('season', 1.5)
    if season_val <= 0.5 and 'Freezing' not in traits and 'Cool' not in traits:
        traits.append("Winter")
    elif season_val >= 2.5:
        traits.append("Summer/Autumn")

    # 저시정
    if row.get('is_low_visibility', 0) > 0.10:
        traits.append("Low Vis")

    # 강수
    if row.get('is_heavy_precip', 0) > 0.05 or row.get('Precipitation(in)', 0) > 0.03:
        traits.append("Precipitation")

    # 습도 (강수/저시정 신호 없을 때)
    humidity = row.get('Humidity(%)', 60)
    if 'Precipitation' not in traits and 'Low Vis' not in traits:
        if humidity > 75:
            traits.append("High Humidity")
        elif humidity < 52:
            traits.append("Dry")

    # 시간대
    hour = row.get('hour', 12)
    is_night_val = row.get('is_night', 0)
    if hour < 6:
        traits.append("Pre-Dawn")          # 새벽 0~5시
    elif 6 <= hour < 10:
        traits.append("Morning Rush")       # 아침 출근
    elif 16 <= hour < 20:
        traits.append("Evening Rush")       # 저녁 퇴근
    elif is_night_val > 0.45:
        traits.append("Night")             # 야간 (시간대 불명확)
    elif is_night_val < 0.2:
        traits.append("Daytime")           # 낮 시간

    # 주말
    if row.get('is_weekend', 0) > 0.17:
        traits.append("Weekend")

    # 도로 인프라
    if row.get('Traffic_Signal', 0) > 0.18:
        traits.append("Signal Zone")
    if row.get('Junction', 0) > 0.1:
        traits.append("Junction")

    return " + ".join(traits) if traits else "Normal"

cluster_names = {
    c: name_cluster(cluster_summary.loc[c])
    for c in range(BEST_K)
}
print("\n[Fix 4+6] Auto cluster naming:")
for c, name in cluster_names.items():
    count = int(cluster_summary.loc[c, 'count'])
    sev   = cluster_summary.loc[c].get('Severity', float('nan'))
    print(f"  Cluster {c}: {name:<45}  (n={count:,}  avg_severity={sev:.2f})")


# ─────────────────────────────────────────────
# 6. Output 2: Cluster Feature Heatmap
# ─────────────────────────────────────────────
heatmap_cols = [
    'Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
    'Wind_Speed(mph)', 'Precipitation(in)',
    'is_low_visibility', 'is_heavy_precip',
    'season',                                # [Fix 6]
    'is_night', 'is_weekend', 'Junction', 'Traffic_Signal'
]
heatmap_cols = [c for c in heatmap_cols if c in cluster_summary.columns]
heatmap_data = cluster_summary[heatmap_cols].T
heatmap_norm = heatmap_data.apply(
    lambda row: (row - row.mean()) / (row.std() + 1e-8), axis=1
)

heatmap_data.columns = [f"C{c}\n{cluster_names[c]}" for c in range(BEST_K)]
heatmap_norm.columns = heatmap_data.columns

fig, ax = plt.subplots(figsize=(max(10, BEST_K * 2.5), 7))
sns.heatmap(
    heatmap_norm,
    annot=heatmap_data.round(2), fmt='.2f',
    cmap='RdYlBu_r', linewidths=0.5, ax=ax, annot_kws={'size': 9}
)
ax.set_title('Cluster Feature Heatmap (color: relative intensity)', fontsize=13)
ax.set_xlabel('Cluster'); ax.set_ylabel('Feature')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('output_2_cluster_heatmap_k8.png', dpi=150, bbox_inches='tight')
plt.show()
print("-> Saved: output_2_cluster_heatmap.png")


# ─────────────────────────────────────────────
# 7. Output 3: Severity Distribution + High-Severity Rate
# ─────────────────────────────────────────────
if 'Severity' in df_full.columns:
    severity_dist = df_full.groupby(['cluster', 'Severity']).size().unstack(fill_value=0)
    severity_pct  = severity_dist.div(severity_dist.sum(axis=1), axis=0) * 100

    high_sev_cols = [c for c in [3, 4] if c in severity_dist.columns]
    severity_pct['high_severity_pct'] = severity_pct[high_sev_cols].sum(axis=1)

    print("\n[Fix 5] High Severity (Severity 3+4) Rate per Cluster:")
    for c in range(BEST_K):
        pct = severity_pct.loc[c, 'high_severity_pct']
        print(f"  Cluster {c} [{cluster_names[c]}]: {pct:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    severity_dist.plot(kind='bar', ax=axes[0], colormap='Set2', edgecolor='white')
    axes[0].set_title('Accident Count by Cluster & Severity', fontsize=12)
    axes[0].set_xlabel('Cluster'); axes[0].set_ylabel('Accident Count')
    axes[0].set_xticklabels(
        [f"C{c}\n{cluster_names[c]}" for c in range(BEST_K)],
        rotation=20, ha='right', fontsize=8
    )
    axes[0].legend(title='Severity', bbox_to_anchor=(1.01, 1))

    severity_pct.drop(columns='high_severity_pct').plot(
        kind='bar', stacked=True, ax=axes[1], colormap='Set2', edgecolor='white'
    )
    axes[1].set_title('Severity Ratio by Cluster (%)', fontsize=12)
    axes[1].set_xlabel('Cluster'); axes[1].set_ylabel('Ratio (%)')
    axes[1].set_xticklabels([f"C{c}" for c in range(BEST_K)], rotation=0)
    axes[1].legend(title='Severity', bbox_to_anchor=(1.01, 1))

    mean_high_sev = severity_pct['high_severity_pct'].mean()
    colors = ['crimson' if p > mean_high_sev else 'steelblue'
              for p in severity_pct['high_severity_pct']]
    axes[2].bar(range(BEST_K), severity_pct['high_severity_pct'],
                color=colors, edgecolor='white')
    axes[2].axhline(mean_high_sev, color='gray', ls='--', lw=1.5, label='Overall Average')
    axes[2].set_title('High Severity (3+4) Rate per Cluster (%)', fontsize=12)
    axes[2].set_xlabel('Cluster'); axes[2].set_ylabel('High Severity Rate (%)')
    axes[2].set_xticks(range(BEST_K))
    axes[2].set_xticklabels(
        [f"C{c}\n{cluster_names[c]}" for c in range(BEST_K)],
        rotation=20, ha='right', fontsize=8
    )
    axes[2].legend()
    plt.tight_layout()
    plt.savefig('output_3_severity_by_cluster_k8.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("-> Saved: output_3_severity_by_cluster.png")


# ─────────────────────────────────────────────
# 8. Output 4: PCA 2D Visualization
# ─────────────────────────────────────────────
print("\nGenerating PCA 2D visualization...")
SAMPLE_VIS = min(30000, len(X_pca))
idx_sample = np.random.choice(len(X_pca), SAMPLE_VIS, replace=False)

pca_vis = PCA(n_components=2, random_state=42)
X_vis   = pca_vis.fit_transform(X)
explained_vis = pca_vis.explained_variance_ratio_ * 100
labels_sample = df_full['cluster'].values[idx_sample]

colors = plt.cm.tab10(np.linspace(0, 0.9, BEST_K))
fig, ax = plt.subplots(figsize=(9, 7))
for k in range(BEST_K):
    mask = labels_sample == k
    ax.scatter(
        X_vis[idx_sample][mask, 0],
        X_vis[idx_sample][mask, 1],
        s=3, alpha=0.4, color=colors[k],
        label=f'C{k}: {cluster_names[k]}'
    )
ax.set_title(
    f'PCA 2D Visualization  (Explained Variance: {sum(explained_vis):.1f}%)\n'
    f'Note: {100 - sum(explained_vis):.1f}% of total variance is not captured in this 2D view',
    fontsize=12
)
ax.set_xlabel(f'PC1 ({explained_vis[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({explained_vis[1]:.1f}%)')
ax.legend(markerscale=4, loc='best', fontsize=8)
plt.tight_layout()
plt.savefig('output_4_pca_clusters_k8.png', dpi=150, bbox_inches='tight')
plt.show()
print("-> Saved: output_4_pca_clusters.png")


# ─────────────────────────────────────────────
# 9. Final Interpretation Output
# ─────────────────────────────────────────────
print("\n" + "="*65)
print(f"[Final Analysis Summary]  K={BEST_K}  |  Reason: {K_REASON}")
print("="*65)

for c in range(BEST_K):
    row  = cluster_summary.loc[c]
    name = cluster_names[c]
    n    = int(row['count'])
    sev  = row.get('Severity', float('nan'))

    high_sev = 0.0
    if 'Severity' in df_full.columns:
        c_df = df_full[df_full['cluster'] == c]
        high_sev = (c_df['Severity'] >= 3).mean() * 100

    low_vis_rate = row.get('is_low_visibility', 0) * 100
    heavy_p_rate = row.get('is_heavy_precip', 0) * 100
    season_val   = row.get('season', float('nan'))
    season_name  = SEASON_LABELS.get(round(season_val), f'{season_val:.1f}') if not np.isnan(season_val) else 'N/A'

    print(f"\n  Cluster {c}: {name}")
    print(f"    Sample count       : {n:,}  ({n / len(df_full) * 100:.1f}%)")
    print(f"    Avg severity       : {sev:.2f}")
    print(f"    High severity rate : {high_sev:.1f}%")
    print(f"    Low visibility rate: {low_vis_rate:.1f}%")
    print(f"    Heavy precip rate  : {heavy_p_rate:.1f}%")
    print(f"    Avg season         : {season_name}  (mean={season_val:.2f})")
