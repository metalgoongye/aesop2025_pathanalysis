import pandas as pd
import numpy as np
import semopy
from scipy import stats
from sklearn.decomposition import PCA

# ── 데이터 로드 & 전처리 ──────────────────────────────────────
df = pd.read_excel('E:/data/aesop2025/aesop2025.xlsx', sheet_name='시군구(행정구제외)')
df = df[df['basic_gov'].notna()].copy()

vars_disparity = ['lit_pc', 'grdp_pc', 'bnbl_rate', 'hale', 'subs', 'oldb']
vars_uci       = ['DD', 'LUM', 'AC', 'SA']
vars_carbon    = ['build_elec_e_pc', 'transport_e_pc', 'absor_pc']
all_vars       = vars_disparity + vars_uci + vars_carbon

df_model = df[all_vars].dropna().copy()
print(f"분석 관측치: {len(df_model)}개\n")

# 표준화
df_z = df_model.copy()
for col in all_vars:
    df_z[col] = stats.zscore(df_model[col], nan_policy='omit')


# ════════════════════════════════════════════════════════════════
# 방법 1: Non-recursive SEM (상호영향 잠재변수 모델)
# ════════════════════════════════════════════════════════════════
# 식별 전략: 각 잠재변수의 지표가 상대방 방정식의 도구변수 역할
#   - UCI 방정식 식별: disparity 지표(lit_pc 등)가 UCI 오차와 무관
#   - disparity 방정식 식별: UCI 지표(DD 등)가 disparity 오차와 무관
# 비재귀모델의 교란항 공분산은 기본으로 자유추정

model_nr = semopy.Model("""
    # 측정 모델
    urban_disparity =~ lit_pc + grdp_pc + bnbl_rate + hale + subs + oldb
    UCI             =~ DD + LUM + AC + SA
    carbon          =~ build_elec_e_pc + transport_e_pc + absor_pc

    # 구조 모델 (상호영향: non-recursive)
    UCI             ~ urban_disparity
    urban_disparity ~ UCI
    carbon          ~ UCI + urban_disparity

    # 비재귀 내생변수 교란항 공분산 (unmeasured common causes)
    UCI             ~~ urban_disparity
""")

print("=" * 60)
print("방법 1: Non-recursive SEM")
print("=" * 60)
try:
    res_nr = model_nr.fit(df_z)
    params_nr = model_nr.inspect()
    # 구조경로만 출력
    structural = params_nr[params_nr['op'].isin(['~'])]
    print("\n[구조 경로]\n", structural[['lval','op','rval','Estimate','Std. Err','z-value','p-value']].to_string(index=False))

    print("\n[모델 적합도]")
    fit_nr = semopy.calc_stats(model_nr)
    idx = ['CFI', 'RMSEA', 'GFI', 'chi2', 'chi2 p-value', 'AIC', 'BIC']
    for i in idx:
        if i in fit_nr.index:
            print(f"  {i:15s}: {fit_nr.loc[i,'Value']:.4f}")

    # 효과 분해
    est = params_nr.set_index(['lval','op','rval'])['Estimate']
    a      = est.get(('UCI',             '~', 'urban_disparity'), np.nan)
    b      = est.get(('carbon',          '~', 'UCI'),             np.nan)
    c      = est.get(('carbon',          '~', 'urban_disparity'), np.nan)
    d      = est.get(('urban_disparity', '~', 'UCI'),             np.nan)

    print(f"\n[효과 분해: urban_disparity -> carbon]")
    print(f"  직접효과                        : {c:.4f}")
    print(f"  간접효과 (-> UCI ->)             : {a:.4f} x {b:.4f} = {a*b:.4f}")
    print(f"  총효과                          : {c + a*b:.4f}")
    print(f"\n[역방향: UCI -> urban_disparity  : {d:.4f}]")

except Exception as e:
    print(f"  Non-recursive SEM 오류: {e}")


# ════════════════════════════════════════════════════════════════
# 방법 2: 2SLS (IV 추정) — factor score 기반
# 각 탄소변수에 대해 개별 2SLS 모델 실행
# 식별 전략:
#   disparity_score 방정식 → UCI 지표(DD, LUM, AC, SA)를 IV로 사용
#   UCI_score 방정식       → disparity 지표(lit_pc, grdp_pc 등)를 IV로 사용
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("방법 2: 2SLS (Factor Score 기반 IV 추정)")
print("=" * 60)

try:
    from linearmodels.iv import IV2SLS

    # PCA 1요인으로 합성점수 추출
    def get_factor_score(data, cols):
        pca = PCA(n_components=1)
        score = pca.fit_transform(stats.zscore(data[cols], axis=0))
        return score.flatten()

    df_z2 = df_z.copy().reset_index(drop=True)
    df_z2['disparity_score'] = get_factor_score(df_z2, vars_disparity)
    df_z2['uci_score']       = get_factor_score(df_z2, vars_uci)

    print("\n[2SLS 모델 구조]")
    print("  내생변수: disparity_score, uci_score (상호내생)")
    print("  IV for uci_score      <- DD, LUM, AC, SA")
    print("  IV for disparity_score <- lit_pc, grdp_pc, bnbl_rate, hale, subs, oldb\n")

    for carbon_var in vars_carbon:
        print(f"--- 종속변수: {carbon_var} ---")
        # 2SLS: carbon ~ [uci_score + disparity_score] | [uci 지표 + disparity 지표]
        # endog: uci_score, disparity_score
        # instruments: 모든 지표 (교차 도구변수)
        iv_model = IV2SLS(
            dependent  = df_z2[carbon_var],
            exog       = None,
            endog      = df_z2[['uci_score', 'disparity_score']],
            instruments= df_z2[vars_uci + vars_disparity]
        )
        res_iv = iv_model.fit(cov_type='robust')
        print(res_iv.summary.tables[1])
        print()

except ImportError:
    print("  linearmodels 패키지 설치 필요: pip install linearmodels")
except Exception as e:
    print(f"  2SLS 오류: {e}")
