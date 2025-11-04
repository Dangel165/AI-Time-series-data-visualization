import matplotlib.pyplot as plt
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
import matplotlib
import os

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")

# 한글 폰트를 DejaVu Sans로 설정 
plt.rcParams['font.family'] = 'DejaVu Sans'

# 백엔드 설정을 Agg로 변경 
matplotlib.use('Agg')

# 저장할 경로 설정
save_path = '/kaggle/working/'

# 기존 파일 삭제 
files_to_delete = [
    'line_2_station_passenger.png', 
    'daily_total_passenger.png', 
    'forecast_passenger.png', 
    'forecast_result.csv'
]

for file_name in files_to_delete:
    file_path = os.path.join(save_path, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)

# CSV 파일 읽기
df = pd.read_csv("/kaggle/input/seoul-subway/Seoul subway.csv", encoding='cp949')
df.columns = ['날짜', '호선명', '승차총승객수', '하차총승객수', '총이용객']

# 컬럼명 목록 출력 및 데이터 확인
print("컬럼명 목록:", df.columns.tolist())
print(df.head())

# '날짜' 컬럼을 datetime 형식으로 변환
df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
df = df.dropna()

# 2호선 승하차 인원 합계 시각화
if set(['호선명', '역명', '승차총승객수', '하차총승객수']).issubset(df.columns):
    line_2 = df[df['호선명'] == '2호선']
    line_2_grouped = line_2.groupby('역명')[['승차총승객수', '하차총승객수']].sum().sort_values(by='승차총승객수', ascending=False)
    line_2_grouped.plot(kind='bar', figsize=(12, 6), title='2호선 역별 승하차 인원 합계')
    plt.xlabel("역명")
    plt.ylabel("인원 수")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'line_2_station_passenger.png'), bbox_inches='tight')
    plt.close()

# 날짜별 총 이용객수 집계 및 시각화
daily = df.groupby('날짜')['총이용객'].sum()
daily.plot(figsize=(12, 5), title="날짜별 총 이용객 수")
plt.xlabel("날짜")
plt.ylabel("총 이용객")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'daily_total_passenger.png'), bbox_inches='tight')
plt.close()

# ARIMA 모델을 위한 데이터 준비
daily = daily.sort_index()
daily = daily.asfreq('D')
daily = daily.fillna(method='ffill')

# ARIMA 모델 훈련 및 예측
model = ARIMA(daily, order=(5, 1, 0))
model_fit = model.fit()

# 향후 30일 예측
forecast = model_fit.forecast(steps=30)

# 예측 결과 시각화
plt.plot(daily.index[-30:], daily[-30:], label='실제 값')
plt.plot(pd.date_range(daily.index[-1], periods=31, freq='D')[1:], forecast, label='예측 값', color='red', linestyle='dashed')
plt.legend()
plt.title("시계열 예측: 총 이용객 수")
plt.xlabel("날짜")
plt.ylabel("총 이용객 수")
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'forecast_passenger.png'), bbox_inches='tight')
plt.close()

# 예측 결과를 CSV 파일로 저장
forecast_df = pd.DataFrame({
    '날짜': pd.date_range(daily.index[-1], periods=31, freq='D')[1:],
    '예측 총이용객수': forecast
})
forecast_df.to_csv(os.path.join(save_path, 'forecast_result.csv'), index=False)