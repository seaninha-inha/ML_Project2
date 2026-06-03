"""
K-Means × DBSCAN 교차 분석
========================================
목적: DBSCAN 지리적 핫스팟에 K-Means 조건 클러스터를 연결
     → "어디서" + "어떤 조건에서" 사고가 밀집하는가

출력:
  output_cross_heatmap.png  : 핫스팟 × 조건 클러스터 분포 히트맵
  output_cross_map.png      : 지배 클러스터 색으로 표시한 핫스팟 지도
  cross_analysis_summary.csv: 핫스팟별 통계 + 지배 클러스터
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 설정 — DBSCAN 코드에서 찾은 값 그대로 입력
# ─────────────────────────────────────────────
EPS           = 0.2521   # k-distance graph elbow 값
MIN_SAMPLES   = 10
DBSCAN_SAMPLE = 50_000
RANDOM_STATE  = 42

CLUSTER_NAMES = {
    0: 'Daytime',
    1: 'High Humidity + Pre-Dawn',
    2: 'Dry + Weekend',
    3: 'Cool + Low Vis',
    4: 'Low Vis + Morning Rush',
    5: 'Low Vis + Precipitation',   # 가장 위험 (중증률 25.1%)
    6: 'Signal Zone',               # 가장 안전 (중증률 7.4%)
    7: 'Freezing + Low Vis',
}


# ─────────────────────────────────────────────
# 1. 데이터 로드 — K-Means 레이블을 df_full에 병합
#    핵심: DBSCAN을 같은 데이터에서 다시 샘플링해서
#          K-Means 레이블과 인덱스 기준 정합 유지
# ─────────────────────────────────────────────
print("데이터 로드 중...")
df_full   = pd.read_csv('preprocessed_full.csv')
km_labels = pd.read_csv('kmeans_cluster_labels.csv', index_col=0)

df_full = df_full.iloc[:len(km_labels)].copy().reset_index(drop=True)
df_full['kmeans_cluster'] = km_labels['cluster'].values
print(f"전체 데이터: {df_full.shape}  K-Means 레이블 병합 완료")


# ─────────────────────────────────────────────
# 2. DBSCAN 샘플링 (K-Means 레이블 포함 상태로)
# ─────────────────────────────────────────────
COLS = ['Start_Lat', 'Start_Lng', 'Severity', 'kmeans_cluster']
df_sample = (
    df_full[COLS]
    .sample(n=min(DBSCAN_SAMPLE, len(df_full)), random_state=RANDOM_STATE)
    .reset_index(drop=True)
)
print(f"DBSCAN 샘플: {df_sample.shape}")

coords = df_sample[['Start_Lat', 'Start_Lng']].values


# ─────────────────────────────────────────────
# 3. DBSCAN 실행 (이전과 동일한 파라미터)
# ─────────────────────────────────────────────
print(f"\nDBSCAN 실행 중 (eps={EPS}, min_samples={MIN_SAMPLES})...")
db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='euclidean')
df_sample['dbscan_cluster'] = db.fit_predict(coords)

n_hotspots = int(df_sample['dbscan_cluster'].nunique()) - 1
noise_rate  = (df_sample['dbscan_cluster'] == -1).mean() * 100
print(f"핫스팟 수: {n_hotspots},  노이즈: {noise_rate:.1f}%")


# ─────────────────────────────────────────────
# 4. 교차 분석 — 핫스팟별 K-Means 클러스터 분포
# ─────────────────────────────────────────────
hotspot_df = df_sample[df_sample['dbscan_cluster'] >= 0].copy()

# 핫스팟 기본 정보
hotspot_info = (
    hotspot_df
    .groupby('dbscan_cluster')
    .agg(
        n_accidents  = ('Severity', 'size'),
        avg_severity = ('Severity', 'mean'),
        center_lat   = ('Start_Lat', 'mean'),
        center_lng   = ('Start_Lng', 'mean'),
    )
    .reset_index()
)

# 핫스팟별 K-Means 클러스터 비율 (%)
cluster_dist = (
    hotspot_df
    .groupby(['dbscan_cluster', 'kmeans_cluster'])
    .size()
    .unstack(fill_value=0)
)
cluster_pct = cluster_dist.div(cluster_dist.sum(axis=1), axis=0) * 100

# 지배 클러스터 (비율 최대)
hotspot_info = hotspot_info.set_index('dbscan_cluster')
hotspot_info['dominant_kmeans'] = cluster_pct.idxmax(axis=1)
hotspot_info['dominant_pct']    = cluster_pct.max(axis=1).round(1)
hotspot_info['dominant_name']   = hotspot_info['dominant_kmeans'].map(CLUSTER_NAMES)

# 가장 위험한 조건 클러스터(C5) 비율
hotspot_info['c5_precip_pct'] = cluster_pct.get(5, pd.Series(0, index=cluster_pct.index)).round(1)

hotspot_info.reset_index(inplace=True)

print("\n[교차 분석 — 상위 10 핫스팟 (사고 건수 기준)]")
top10 = hotspot_info.sort_values('n_accidents', ascending=False).head(10)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
print(top10[['dbscan_cluster', 'n_accidents', 'avg_severity',
             'center_lat', 'center_lng',
             'dominant_name', 'dominant_pct', 'c5_precip_pct']].to_string(index=False))

hotspot_info.to_csv('cross_analysis_summary.csv', index=False)
print("\n저장: cross_analysis_summary.csv")


# ─────────────────────────────────────────────
# 5. 시각화 1 — 상위 15 핫스팟 × K-Means 클러스터 히트맵
# ─────────────────────────────────────────────
top15_ids = (
    hotspot_info.sort_values('n_accidents', ascending=False)
    .head(15)['dbscan_cluster']
    .tolist()
)
heatmap_data = cluster_pct.loc[top15_ids].copy()
heatmap_data.columns = [CLUSTER_NAMES.get(int(c), f'C{c}') for c in heatmap_data.columns]
heatmap_data.index   = [f'HS{i+1}' for i in range(len(heatmap_data))]

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(
    heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd',
    linewidths=0.5, ax=ax, annot_kws={'size': 8}
)
ax.set_title('Top 15 Hotspots × K-Means Cluster Distribution (%)\n'
             '(Each row = one geographic hotspot, columns = condition patterns)', fontsize=12)
ax.set_xlabel('K-Means Condition Cluster')
ax.set_ylabel('DBSCAN Hotspot (sorted by size)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('output_cross_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("저장: output_cross_heatmap.png")


# ─────────────────────────────────────────────
# 6. 시각화 2 — 지도: 핫스팟을 지배 클러스터 색으로 표시
# ─────────────────────────────────────────────
COLOR_MAP = {
    0: '#4E79A7',   # Daytime — 파랑
    1: '#F28E2B',   # Pre-Dawn — 주황
    2: '#59A14F',   # Dry+Weekend — 초록
    3: '#E15759',   # Cool+Low Vis — 빨강(연)
    4: '#76B7B2',   # Low Vis+Morning — 청록
    5: '#CC0000',   # Low Vis+Precip — 진빨강 (가장 위험)
    6: '#9467BD',   # Signal Zone — 보라
    7: '#8C564B',   # Freezing — 갈색
}

fig, ax = plt.subplots(figsize=(14, 8))

noise_mask = df_sample['dbscan_cluster'] == -1
ax.scatter(df_sample.loc[noise_mask, 'Start_Lng'],
           df_sample.loc[noise_mask, 'Start_Lat'],
           s=1, c='lightgray', alpha=0.2, label='Noise')

for _, row in hotspot_info.iterrows():
    hid = row['dbscan_cluster']
    dom = int(row['dominant_kmeans']) if not pd.isna(row['dominant_kmeans']) else -1
    color = COLOR_MAP.get(dom, 'gray')
    mask  = df_sample['dbscan_cluster'] == hid
    ax.scatter(df_sample.loc[mask, 'Start_Lng'],
               df_sample.loc[mask, 'Start_Lat'],
               s=4, color=color, alpha=0.7)

from matplotlib.patches import Patch
patches = [Patch(color=COLOR_MAP[k], label=f'C{k}: {v}')
           for k, v in CLUSTER_NAMES.items()]
patches.append(Patch(color='lightgray', label='Noise'))
ax.legend(handles=patches, loc='lower left', fontsize=7, title='Dominant Condition Cluster')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title('DBSCAN Hotspots — Colored by Dominant K-Means Condition Cluster\n'
             '(Red = Low Vis + Precipitation: highest risk)', fontsize=12)
plt.tight_layout()
plt.savefig('output_cross_map.png', dpi=150, bbox_inches='tight')
plt.show()
print("저장: output_cross_map.png")


# ─────────────────────────────────────────────
# 7. 핵심 인사이트 출력 (보고서 문장 근거용)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("[교차 분석 핵심 인사이트]")
print("="*60)

print(f"\n전체 핫스팟 {n_hotspots}개의 지배 조건 클러스터 분포:")
dominant_counts = hotspot_info['dominant_name'].value_counts()
for name, cnt in dominant_counts.items():
    pct = cnt / n_hotspots * 100
    print(f"  {name}: {cnt}개 ({pct:.1f}%)")

# C5 비율 높은 위험 핫스팟
high_risk = hotspot_info[hotspot_info['c5_precip_pct'] > 20].sort_values(
    'c5_precip_pct', ascending=False
)
print(f"\nC5(저시정+강수) 비율 > 20% 핫스팟: {len(high_risk)}개")
if len(high_risk) > 0:
    print(f"  해당 핫스팟 평균 심각도: {high_risk['avg_severity'].mean():.3f}")
    print(f"  전체 평균 심각도 대비: {high_risk['avg_severity'].mean() - hotspot_info['avg_severity'].mean():+.3f}")

# 밀집 ≠ 위험 검증 (보고서 한계 항목)
corr = hotspot_info['n_accidents'].corr(hotspot_info['avg_severity'])
print(f"\n사고 밀집도 vs 평균 심각도 상관계수: {corr:.3f}")
if abs(corr) < 0.3:
    print("  → '밀집 지역 ≠ 위험 지역' 확인 (DBSCAN 한계 근거)")
else:
    print("  → 어느 정도 상관 있음")

print("\n완료. 출력 파일:")
print("  output_cross_heatmap.png")
print("  output_cross_map.png")
print("  cross_analysis_summary.png")
