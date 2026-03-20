import pandas as pd
import numpy as np
import semopy

# 재현 가능한 난수 시드 설정
np.random.seed(42)
n = 200

# 데이터 생성 (urban disparity -> urban sprawl -> carbon emissions)
urban_disparity = np.random.normal(0, 1, n)
urban_sprawl = 0.6 * urban_disparity + np.random.normal(0, 0.8, n)
carbon_emissions = 0.5 * urban_sprawl + 0.2 * urban_disparity + np.random.normal(0, 0.7, n)

df = pd.DataFrame({
    'urban_disparity': urban_disparity,
    'urban_sprawl': urban_sprawl,
    'carbon_emissions': carbon_emissions
})

# SEM 모델 정의 (경로분석)
model_desc = """
    urban_sprawl ~ urban_disparity
    carbon_emissions ~ urban_sprawl + urban_disparity
"""

# 모델 생성 및 추정
model = semopy.Model(model_desc)
result = model.fit(df)

print("=" * 50)
print("경로분석 결과 (Path Analysis Results)")
print("=" * 50)
print(model.inspect())

# 모델 적합도 지수
print("\n" + "=" * 50)
print("모델 적합도 (Model Fit Indices)")
print("=" * 50)
stats = semopy.calc_stats(model)
print(stats.T)

# 간접효과 계산 (urban_disparity -> urban_sprawl -> carbon_emissions)
params = model.inspect()
a = params.loc[(params['lval'] == 'urban_sprawl') & (params['rval'] == 'urban_disparity'), 'Estimate'].values[0]
b = params.loc[(params['lval'] == 'carbon_emissions') & (params['rval'] == 'urban_sprawl'), 'Estimate'].values[0]
c_prime = params.loc[(params['lval'] == 'carbon_emissions') & (params['rval'] == 'urban_disparity'), 'Estimate'].values[0]

indirect = a * b
total = indirect + c_prime

print("\n" + "=" * 50)
print("효과 분해 (Effect Decomposition)")
print("=" * 50)
print(f"직접효과 (urban_disparity -> carbon_emissions): {c_prime:.4f}")
print(f"간접효과 (urban_disparity -> urban_sprawl -> carbon_emissions): {a:.4f} × {b:.4f} = {indirect:.4f}")
print(f"총효과: {total:.4f}")
