"""
US Accidents Dataset - 데이터 전처리
------------------------------------
K-Means (조건 패턴 클러스터링) + DBSCAN (지리적 핫스팟 탐지) 공용 전처리 스크립트

실행 전 설치:
    pip install pandas numpy scikit-learn matplotlib seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. 데이터 로드 (필요한 컬럼만 선택해서 메모리 절약)
# ─────────────────────────────────────────────
USED_COLS = [
    'Severity',
    'Start_Time',
    'Start_Lat', 'Start_Lng',
    'Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
    'Wind_Speed(mph)', 'Precipitation(in)',
    'Weather_Condition', 'Sunrise_Sunset',
    'Junction', 'Traffic_Signal', 'Crossing',
]

print("데이터 로드 중...")
df = pd.read_csv('US_Accidents_March23.csv', usecols=USED_COLS)
print(f"원본 크기: {df.shape}  ({df.shape[0]:,}행)")


# ─────────────────────────────────────────────
# 2. 샘플링 (전체 280만 행 → 분석 가능한 크기로 축소)
#    K-Means: 50만, DBSCAN: 20만 이하 권장
# ─────────────────────────────────────────────
SAMPLE_SIZE = 300_000   # 필요에 따라 조절
df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
print(f"샘플링 후 크기: {df.shape}")


# ─────────────────────────────────────────────
# 3. 결측치 처리
# ─────────────────────────────────────────────
print("\n[결측치 현황]")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 위경도 결측치는 제거 (DBSCAN 핵심 변수)
df.dropna(subset=['Start_Lat', 'Start_Lng'], inplace=True)

# 수치형 → 중앙값으로 대체
numeric_cols = [
    'Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
    'Wind_Speed(mph)', 'Precipitation(in)'
]
for col in numeric_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# 범주형 → 최빈값으로 대체
df['Weather_Condition'].fillna(df['Weather_Condition'].mode()[0], inplace=True)
df['Sunrise_Sunset'].fillna('Day', inplace=True)

# Boolean 결측치 → False로 대체
bool_cols = ['Junction', 'Traffic_Signal', 'Crossing']
for col in bool_cols:
    df[col].fillna(False, inplace=True)

print(f"\n결측치 처리 후 잔여 결측치: {df.isnull().sum().sum()}개")


# ─────────────────────────────────────────────
# 4. 이상치 처리 (IQR × 3.0 기준 — 너무 좁히면 실제 극단 사고 조건이 사라짐)
# ─────────────────────────────────────────────
def remove_outliers_iqr(dataframe, col, factor=3.0):
    Q1 = dataframe[col].quantile(0.25)
    Q3 = dataframe[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return dataframe[(dataframe[col] >= lower) & (dataframe[col] <= upper)]

outlier_targets = ['Temperature(F)', 'Wind_Speed(mph)']
before = len(df)
for col in outlier_targets:
    df = remove_outliers_iqr(df, col)
print(f"\n이상치 제거: {before - len(df):,}행 제거 → {len(df):,}행 남음")


# ─────────────────────────────────────────────
# 5. 시간 변수 추출
# ─────────────────────────────────────────────
df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')

df['hour']        = df['Start_Time'].dt.hour
df['day_of_week'] = df['Start_Time'].dt.dayofweek   # 0=월요일
df['month']       = df['Start_Time'].dt.month
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

# 시간대 구분 (출퇴근/낮/밤 등 의미있는 구간으로 나눔)
def time_period(h):
    if   0 <= h <  6: return 0   # 새벽
    elif 6 <= h < 10: return 1   # 아침 출근
    elif 10 <= h < 16: return 2  # 낮
    elif 16 <= h < 20: return 3  # 저녁 퇴근
    else:              return 4  # 밤

df['time_period'] = df['hour'].apply(time_period)
df['is_night']    = (df['Sunrise_Sunset'] == 'Night').astype(int)


# ─────────────────────────────────────────────
# 6. 날씨 조건 단순화 (100+ 종류 → 7개 카테고리)
# ─────────────────────────────────────────────
def simplify_weather(w):
    w = str(w).lower()
    if any(k in w for k in ['rain', 'drizzle', 'shower']):
        return 'Rain'
    elif any(k in w for k in ['snow', 'sleet', 'ice', 'wintry', 'freezing']):
        return 'Snow_Ice'
    elif any(k in w for k in ['fog', 'haze', 'mist', 'smoke']):
        return 'Fog'
    elif any(k in w for k in ['thunder', 'storm', 'squall']):
        return 'Thunder'
    elif any(k in w for k in ['clear', 'fair']):
        return 'Clear'
    elif any(k in w for k in ['cloud', 'overcast', 'partly']):
        return 'Cloudy'
    else:
        return 'Other'

df['weather_simple'] = df['Weather_Condition'].apply(simplify_weather)

# One-hot encoding
weather_dummies = pd.get_dummies(df['weather_simple'], prefix='weather')
df = pd.concat([df, weather_dummies], axis=1)

# Boolean → int 변환
for col in bool_cols:
    df[col] = df[col].astype(int)


# ─────────────────────────────────────────────
# 7. K-Means용 feature 선택 & 정규화
# ─────────────────────────────────────────────
weather_dummy_cols = [c for c in df.columns if c.startswith('weather_')
                      if c.startswith('weather_') and c != 'weather_simple']

KMEANS_FEATURES = [
    'Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
    'Wind_Speed(mph)', 'Precipitation(in)',
    'hour', 'time_period', 'is_weekend', 'is_night',
    'Junction', 'Traffic_Signal', 'Crossing',
] + weather_dummy_cols

df_kmeans = df[KMEANS_FEATURES].copy()

scaler = StandardScaler()
df_kmeans_scaled = pd.DataFrame(
    scaler.fit_transform(df_kmeans),
    columns=KMEANS_FEATURES
)

print(f"\n[K-Means 입력] shape: {df_kmeans_scaled.shape}")
print("Feature 목록:", KMEANS_FEATURES)


# ─────────────────────────────────────────────
# 8. DBSCAN용 feature 선택 (위경도 + 심각도)
#    위경도는 정규화 없이 그대로 사용
# ─────────────────────────────────────────────
DBSCAN_SAMPLE = 100_000   # DBSCAN은 샘플 크기에 민감하므로 추가 조절

df_dbscan = df[['Start_Lat', 'Start_Lng', 'Severity']].sample(
    n=min(DBSCAN_SAMPLE, len(df)), random_state=42
).reset_index(drop=True)

print(f"\n[DBSCAN 입력] shape: {df_dbscan.shape}")


# ─────────────────────────────────────────────
# 9. EDA 요약 시각화 (선택)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('US Accidents - EDA 요약', fontsize=14)

df['hour'].hist(bins=24, ax=axes[0,0], color='steelblue')
axes[0,0].set_title('시간대별 사고 건수')
axes[0,0].set_xlabel('시간')

df['Severity'].value_counts().sort_index().plot(kind='bar', ax=axes[0,1], color='coral')
axes[0,1].set_title('사고 심각도 분포')
axes[0,1].set_xlabel('Severity')

df['weather_simple'].value_counts().plot(kind='bar', ax=axes[0,2], color='mediumseagreen')
axes[0,2].set_title('날씨 조건별 사고 건수')
axes[0,2].tick_params(axis='x', rotation=30)

df['Temperature(F)'].hist(bins=40, ax=axes[1,0], color='tomato')
axes[1,0].set_title('기온 분포 (°F)')

df['Visibility(mi)'].hist(bins=40, ax=axes[1,1], color='mediumpurple')
axes[1,1].set_title('시정 분포 (miles)')

axes[1,2].scatter(df['Start_Lng'].sample(5000), df['Start_Lat'].sample(5000),
                  s=0.3, alpha=0.3, color='navy')
axes[1,2].set_title('사고 위치 분포')
axes[1,2].set_xlabel('경도')
axes[1,2].set_ylabel('위도')

plt.tight_layout()
plt.savefig('eda_summary.png', dpi=120, bbox_inches='tight')
plt.show()
print("\nEDA 시각화 저장: eda_summary.png")


# ─────────────────────────────────────────────
# 10. 전처리된 파일 저장
# ─────────────────────────────────────────────
df_kmeans_scaled.to_csv('kmeans_input.csv', index=False)
df_dbscan.to_csv('dbscan_input.csv', index=False)
df.to_csv('preprocessed_full.csv', index=False)

print("\n전처리 완료! 저장된 파일:")
print("  - kmeans_input.csv   → K-Means 담당자에게 전달")
print("  - dbscan_input.csv   → DBSCAN 담당자에게 전달")
print("  - preprocessed_full.csv → 교차 분석용 전체 데이터")