"""
US Accidents - DBSCAN 지리적 핫스팟 탐지
==========================================
지리적으로 사고가 비정상적으로 밀집된 '핫스팟'을 자동으로 탐지한다.

핵심 설계 (강의 내용 + Project#1 피드백 반영):
  1) 위경도(Start_Lat, Start_Lng)로만 군집화한다. Severity는 군집화에 넣지 않고
     '해석' 단계에서만 사용한다. (단위가 다른 변수를 거리 계산에 섞지 않기 위함)
  2) eps는 임의로 정하지 않고 k-distance graph의 elbow로 근거 있게 선택한다.
  3) min_samples는 "핫스팟으로 인정할 최소 사고 건수"라는 도메인 논리로 정한다.
  4) 평가: 군집 개수 / 노이즈 비율 / Silhouette score / 지도 시각화.
  5) DBSCAN의 한계(밀도 차이, '밀집≠위험')를 결과로 함께 확인한다.

실행 전 설치:
    pip install pandas numpy scikit-learn matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# matplotlib 한글 폰트 (윈도우). 없으면 라벨이 깨지므로 영어로 두어도 무방.
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# 0. 설정
# ─────────────────────────────────────────────
INPUT_CSV       = 'dbscan_input.csv'   # 전처리 결과: Start_Lat, Start_Lng, Severity
DBSCAN_SAMPLE   = 50_000   # DBSCAN은 O(n log n)이라 너무 크면 느림. 필요시 조절.
MIN_SAMPLES     = 10       # 핫스팟으로 인정할 최소 사고 건수 (도메인 근거)
K_FOR_KDIST     = MIN_SAMPLES  # k-distance graph의 k는 보통 min_samples와 맞춘다
RANDOM_STATE    = 42


# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
print("데이터 로드 중...")
df = pd.read_csv(INPUT_CSV)
print(f"로드 크기: {df.shape}")

# DBSCAN은 표본 크기에 민감하므로 필요하면 추가 샘플링
if len(df) > DBSCAN_SAMPLE:
    df = df.sample(n=DBSCAN_SAMPLE, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"샘플링 후: {df.shape}")

# 위경도만 군집화에 사용 (Severity는 해석용으로 따로 보관)
coords = df[['Start_Lat', 'Start_Lng']].values
severity = df['Severity'].values


# ─────────────────────────────────────────────
# 2. eps 선택을 위한 k-distance graph
#    각 점에서 k번째 가까운 이웃까지의 거리를 정렬 → elbow가 좋은 eps
# ─────────────────────────────────────────────
print(f"\nk-distance graph 계산 중 (k={K_FOR_KDIST})...")
nbrs = NearestNeighbors(n_neighbors=K_FOR_KDIST).fit(coords)
distances, _ = nbrs.kneighbors(coords)

# k번째 이웃까지 거리만 추출 후 오름차순 정렬
k_dist = np.sort(distances[:, -1])

plt.figure(figsize=(8, 5))
plt.plot(k_dist, color='steelblue')
plt.xlabel(f'Points sorted by distance')
plt.ylabel(f'{K_FOR_KDIST}-th nearest neighbor distance (degrees)')
plt.title('k-distance Graph (elbow = good eps)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()   # 화면에 바로 표시 (이 그래프의 꺾이는 지점을 eps로 사용)

# elbow 자동 추정 (Kneedle 방식: 양 끝점을 잇는 직선에서 가장 멀리 떨어진 점).
# 곡선이 급히 솟기 직전의 '무릎'을 찾는다. 단순 최대 곡률보다 노이즈에 강건.
# 위도/경도 1도 ≈ 111km. eps=0.01 ≈ 약 1.1km 반경.
x = np.arange(len(k_dist), dtype=float)
y = k_dist.astype(float)
# 양 끝을 잇는 직선 (x0,y0)-(x1,y1)에서 각 점까지의 수직 거리
x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
line_len = np.hypot(x1 - x0, y1 - y0)
# 점-직선 거리 = |cross| / line_len
dist_to_line = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / line_len
elbow_idx = int(np.argmax(dist_to_line))
eps_auto = float(k_dist[elbow_idx])
print(f"자동 추정 eps ≈ {eps_auto:.4f} (약 {eps_auto * 111:.1f} km)")

# 실제 사용할 eps. 자동값을 기본으로 하되 그래프 보고 수정 가능.
EPS = round(eps_auto, 4)
print(f"사용할 eps = {EPS}  (약 {EPS * 111:.1f} km),  min_samples = {MIN_SAMPLES}")


# ─────────────────────────────────────────────
# 3. DBSCAN 실행
# ─────────────────────────────────────────────
print("\nDBSCAN 군집화 중...")
db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='euclidean')
labels = db.fit_predict(coords)

df['cluster'] = labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # -1은 노이즈
n_noise = int(np.sum(labels == -1))
noise_ratio = n_noise / len(labels) * 100

print(f"발견된 핫스팟(군집) 수: {n_clusters}")
print(f"노이즈 점: {n_noise:,}개 ({noise_ratio:.1f}%)")


# ─────────────────────────────────────────────
# 4. 평가: Silhouette score (노이즈 제외)
# ─────────────────────────────────────────────
mask = labels != -1
if n_clusters >= 2 and mask.sum() > n_clusters:
    sil = silhouette_score(coords[mask], labels[mask])
    print(f"Silhouette score (노이즈 제외): {sil:.3f}  (0.5 이상이면 양호)")
else:
    print("Silhouette score 계산 불가 (군집이 2개 미만)")


# ─────────────────────────────────────────────
# 5. 시각화: 지도 위 핫스팟
# ─────────────────────────────────────────────
plt.figure(figsize=(12, 7))

# 노이즈는 연회색 점으로
noise_mask = labels == -1
plt.scatter(coords[noise_mask, 1], coords[noise_mask, 0],
            s=1, c='lightgray', alpha=0.3, label='Noise')

# 각 군집은 색을 다르게
unique_clusters = sorted(set(labels) - {-1})
cmap = plt.cm.get_cmap('tab20', max(len(unique_clusters), 1))
for i, c in enumerate(unique_clusters):
    cm = labels == c
    plt.scatter(coords[cm, 1], coords[cm, 0],
                s=3, color=cmap(i), alpha=0.6)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'DBSCAN Accident Hotspots  (eps={EPS}, min_samples={MIN_SAMPLES})\n'
          f'{n_clusters} hotspots, {noise_ratio:.1f}% noise')
plt.tight_layout()
plt.show()   # 화면에 바로 표시


# ─────────────────────────────────────────────
# 6. 핫스팟 해석: 각 군집의 평균 심각도 & 규모
#    (여기서 비로소 Severity 사용 — 군집화가 아닌 '해석'에)
# ─────────────────────────────────────────────
print("\n[상위 핫스팟 요약 — 사고 건수 기준 Top 10]")
summary = (
    df[df['cluster'] != -1]
    .groupby('cluster')
    .agg(
        n_accidents=('cluster', 'size'),
        avg_severity=('Severity', 'mean'),
        center_lat=('Start_Lat', 'mean'),
        center_lng=('Start_Lng', 'mean'),
    )
    .sort_values('n_accidents', ascending=False)
    .reset_index()
)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
print(summary.head(10).to_string(index=False))

summary.to_csv('hotspot_summary.csv', index=False)
print("\n저장: hotspot_summary.csv")


# ─────────────────────────────────────────────
# 7. 한계 점검 (보고서용): 밀집 != 위험
#    사고 건수가 많은 핫스팟이 꼭 심각도가 높은 것은 아님을 수치로 확인
# ─────────────────────────────────────────────
if len(summary) >= 2:
    corr = summary['n_accidents'].corr(summary['avg_severity'])
    print(f"\n[한계 점검] 핫스팟의 (사고 건수) vs (평균 심각도) 상관계수: {corr:.3f}")
    print("  → 0에 가깝거나 음수면, '사고가 많이 나는 곳 ≠ 더 심각한 곳'이라는 의미.")
    print("    즉 DBSCAN이 찾은 핫스팟은 '밀집 지역'이지 '위험 지역'이라 단정 못 함.")

print("\n완료! 두 그래프(k-distance, 핫스팟 지도)는 화면에 표시되었습니다.")
print("핫스팟별 통계는 hotspot_summary.csv 로도 저장했습니다.")