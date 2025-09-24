# 1. 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2. 데이터 준비
# --- 데이터 파일이 'data' 폴더 안에 있다고 가정 ---
df = pd.read_csv('data/ch2_scores_em.csv', index_col='student number')

# 10명의 영어 점수 데이터 생성
scores = np.array(df['english'])[:10]
scores_df = pd.DataFrame({'score': scores},
                         index=pd.Index(['A', 'B', 'C', 'D', 'E',
                                       'F', 'G', 'H', 'I', 'J'],
                                      name='student'))
print("--- 10명의 영어 점수 ---")
print(scores_df)
print("\n" + "="*30 + "\n")


# 3. 데이터 중심 지표 계산
print("--- 데이터 중심 지표 ---")
mean_val = np.mean(scores)
median_val = np.median(scores)
mode_val = pd.Series([1, 1, 1, 2, 2, 3]).mode()
print(f"평균값: {mean_val:.3f}")
print(f"중앙값: {median_val:.3f}")
print("최빈값 (예시):")
print(mode_val)
print("\n" + "="*30 + "\n")


# 4. 데이터 산포도 지표 계산
print("--- 데이터 산포도 지표 ---")
deviation = scores - mean_val
variance = np.var(scores)
std_dev = np.std(scores)
data_range = np.max(scores) - np.min(scores)
scores_Q1 = np.percentile(scores, 25)
scores_Q3 = np.percentile(scores, 75)
scores_IQR = scores_Q3 - scores_Q1

print("편차:")
print(deviation)
print(f"\n분산: {variance:.3f}")
print(f"표준편차: {std_dev:.3f}")
print(f"범위: {data_range}")
print(f"사분위 범위 (IQR): {scores_IQR:.3f}")
print("\n--- 데이터 요약 ---")
print(pd.Series(scores).describe())
print("\n" + "="*30 + "\n")


# 5. 데이터 정규화
print("--- 데이터 정규화 ---")
z_scores = (scores - np.mean(scores)) / np.std(scores)
t_scores = 50 + 10 * z_scores
print("Z-점수 (표준화):")
print(z_scores)
print("\n편찻값 (T-점수):")
print(t_scores)
print("\n" + "="*30 + "\n")


# 6. 1차원 데이터 시각화
# 50명 전체 영어 점수 데이터
english_scores = np.array(df['english'])

# 도수분포표 생성
print("--- 50명 영어 점수 도수분포표 ---")
freq, _ = np.histogram(english_scores, bins=10, range=(0, 100))
freq_class = [f'{i}~{i+10}' for i in range(0, 100, 10)]
freq_dist_df = pd.DataFrame({'frequency': freq}, index=pd.Index(freq_class, name='class'))
print(freq_dist_df)
print("\n" + "="*30 + "\n")

# 히스토그램 생성
print("--- 히스토그램 생성 중... ---")
fig_hist = plt.figure(figsize=(10, 6))
ax_hist = fig_hist.add_subplot(111)
ax_hist.hist(english_scores, bins=10, range=(0, 100))
ax_hist.set_xlabel('Score')
ax_hist.set_ylabel('Person Number')
ax_hist.set_title('Histogram of English Scores')
# plt.show() # 로컬에서 실행 시 주석 해제하여 그래프 확인

# 상자 그림 생성
print("--- 상자 그림 생성 중... ---")
fig_box = plt.figure(figsize=(5, 6))
ax_box = fig_box.add_subplot(111)
ax_box.boxplot(english_scores, tick_labels=['english'])
ax_box.set_title('Box Plot of English Scores')

# 모든 그래프를 화면에 표시
plt.show()

print("\n--- 모든 코드 실행 완료 ---")