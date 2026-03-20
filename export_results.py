import pandas as pd
import numpy as np
import semopy
from scipy import stats
from sklearn.decomposition import PCA
from linearmodels.iv import IV2SLS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, numbers
from openpyxl.utils import get_column_letter
import zipfile, os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
OUT = 'results_output'
os.makedirs(OUT, exist_ok=True)

# ── 데이터 ────────────────────────────────────────────────────
df_raw = pd.read_excel('E:/data/aesop2025/aesop2025.xlsx', sheet_name='시군구(행정구제외)')
df_raw = df_raw[df_raw['basic_gov'].notna()].copy()

vars_disparity = ['lit_pc', 'grdp_pc', 'bnbl_rate', 'hale', 'subs', 'oldb']
vars_uci       = ['DD', 'LUM', 'AC', 'SA']
vars_carbon    = ['build_elec_e_pc', 'transport_e_pc', 'absor_pc']
all_vars       = vars_disparity + vars_uci + vars_carbon

df_model = df_raw[all_vars].dropna().copy()
df_z = df_model.copy()
for col in all_vars:
    df_z[col] = stats.zscore(df_model[col], nan_policy='omit')

N = len(df_model)
print(f"분석 관측치: {N}개")


# ════════════════════════════════════════════════════════════════
# 1. Non-recursive SEM
# ════════════════════════════════════════════════════════════════
model_nr = semopy.Model("""
    urban_disparity =~ lit_pc + grdp_pc + bnbl_rate + hale + subs + oldb
    UCI             =~ DD + LUM + AC + SA
    carbon          =~ build_elec_e_pc + transport_e_pc + absor_pc

    UCI             ~ urban_disparity
    urban_disparity ~ UCI
    carbon          ~ UCI + urban_disparity
    UCI             ~~ urban_disparity
""")
model_nr.fit(df_z)
params_nr = model_nr.inspect()
fit_nr    = semopy.calc_stats(model_nr)

est = params_nr.set_index(['lval','op','rval'])['Estimate']
se  = params_nr.set_index(['lval','op','rval'])['Std. Err']
pv  = params_nr.set_index(['lval','op','rval'])['p-value']

a   = est[('UCI',             '~', 'urban_disparity')]
b   = est[('carbon',          '~', 'UCI')]
c   = est[('carbon',          '~', 'urban_disparity')]
d   = est[('urban_disparity', '~', 'UCI')]

indirect = a * b
total    = c + indirect


# ════════════════════════════════════════════════════════════════
# 2. 2SLS
# ════════════════════════════════════════════════════════════════
df_z2 = df_z.copy().reset_index(drop=True)

def get_pca_score(data, cols):
    pca = PCA(n_components=1)
    return pca.fit_transform(stats.zscore(data[cols].values, axis=0)).flatten()

df_z2['disparity_score'] = get_pca_score(df_z2, vars_disparity)
df_z2['uci_score']       = get_pca_score(df_z2, vars_uci)

iv_results = {}
for cv in vars_carbon:
    res = IV2SLS(
        dependent   = df_z2[cv],
        exog        = None,
        endog       = df_z2[['uci_score', 'disparity_score']],
        instruments = df_z2[vars_uci + vars_disparity]
    ).fit(cov_type='robust')
    iv_results[cv] = res


# ════════════════════════════════════════════════════════════════
# 엑셀 저장
# ════════════════════════════════════════════════════════════════
def style_header(ws, row, cols, color='1F4E79'):
    fill = PatternFill('solid', fgColor=color)
    font = Font(color='FFFFFF', bold=True, size=10)
    for c in range(1, cols+1):
        cell = ws.cell(row=row, column=c)
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(horizontal='center')

def border_range(ws, r1, c1, r2, c2):
    thin = Side(style='thin')
    for r in range(r1, r2+1):
        for c in range(c1, c2+1):
            ws.cell(r,c).border = Border(left=thin, right=thin, top=thin, bottom=thin)

def sig_star(p):
    if pd.isna(p): return ''
    try:
        p = float(p)
    except (ValueError, TypeError):
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

def safe_round(val, n=4):
    try:
        v = float(val)
        return round(v, n)
    except (ValueError, TypeError):
        return ''

# semopy는 측정모델을 'indicator ~ latent' (~) 로 저장함 — =~ 아님
latent_vars = ['urban_disparity', 'UCI', 'carbon']
structural_mask   = (params_nr['op'] == '~') & \
                    params_nr['lval'].isin(latent_vars) & \
                    params_nr['rval'].isin(latent_vars)
measurement_mask  = (params_nr['op'] == '~') & \
                    params_nr['rval'].isin(latent_vars) & \
                    ~params_nr['lval'].isin(latent_vars)

wb = Workbook()

# ── Sheet 1: SEM 구조 경로 ───────────────────────────────────
ws1 = wb.active
ws1.title = 'SEM_Structural'
ws1.column_dimensions['A'].width = 22
ws1.column_dimensions['B'].width = 6
ws1.column_dimensions['C'].width = 22
for col in ['D','E','F','G','H']:
    ws1.column_dimensions[col].width = 13

ws1['A1'] = 'Non-recursive SEM — Structural Paths'
ws1['A1'].font = Font(bold=True, size=12)
ws1.merge_cells('A1:H1')

headers = ['LHS','Op','RHS','Estimate','Std. Err','z-value','p-value','Sig.']
for i, h in enumerate(headers, 1):
    ws1.cell(3, i, h)
style_header(ws1, 3, len(headers))

structural_rows = params_nr[structural_mask].reset_index(drop=True)
for r_i, row in structural_rows.iterrows():
    p = row['p-value']
    data = [row['lval'], row['op'], row['rval'],
            safe_round(row['Estimate']), safe_round(row['Std. Err']),
            safe_round(row['z-value'], 3), safe_round(p), sig_star(p)]
    for c_i, val in enumerate(data, 1):
        cell = ws1.cell(r_i+4, c_i, val)
        cell.alignment = Alignment(horizontal='center')
border_range(ws1, 3, 1, len(structural_rows)+4, len(headers))

# ── Sheet 2: SEM 측정 모델 ───────────────────────────────────
ws2 = wb.create_sheet('SEM_Measurement')
for col, w in zip(['A','B','C','D','E','F','G','H'], [22,6,22,13,13,13,13,8]):
    ws2.column_dimensions[col].width = w

ws2['A1'] = 'Non-recursive SEM — Measurement Model'
ws2['A1'].font = Font(bold=True, size=12)
ws2.merge_cells('A1:H1')
for i, h in enumerate(headers, 1):
    ws2.cell(3, i, h)
style_header(ws2, 3, len(headers), color='2E7539')

measurement_rows = params_nr[measurement_mask].reset_index(drop=True)
for r_i, row in measurement_rows.iterrows():
    p = row['p-value']
    data = [row['lval'], row['op'], row['rval'],
            safe_round(row['Estimate']), safe_round(row['Std. Err']),
            safe_round(row['z-value'], 3), safe_round(p), sig_star(p)]
    for c_i, val in enumerate(data, 1):
        ws2.cell(r_i+4, c_i, val).alignment = Alignment(horizontal='center')
border_range(ws2, 3, 1, len(measurement_rows)+4, len(headers))

# ── Sheet 3: 모델 적합도 ─────────────────────────────────────
ws3 = wb.create_sheet('Model_Fit')
ws3.column_dimensions['A'].width = 20
ws3.column_dimensions['B'].width = 15
ws3.column_dimensions['C'].width = 20

ws3['A1'] = 'Model Fit Indices'
ws3['A1'].font = Font(bold=True, size=12)
ws3.merge_cells('A1:C1')
for i, h in enumerate(['Index','Value','Threshold'], 1):
    ws3.cell(3, i, h)
style_header(ws3, 3, 3, color='7B3F00')

thresholds = {
    'DoF': '', 'chi2': '', 'chi2 p-value': '> .05 (good fit)',
    'CFI': '> .95 (good)', 'GFI': '> .90 (good)', 'AGFI': '> .85 (good)',
    'NFI': '> .90 (good)', 'TLI': '> .95 (good)',
    'RMSEA': '< .05 (good), < .08 (acceptable)',
    'AIC': 'lower = better', 'BIC': 'lower = better'
}
for r_i, (idx_name, thresh) in enumerate(thresholds.items(), 4):
    if idx_name in fit_nr.index:
        val = fit_nr.loc[idx_name, 'Value']
        ws3.cell(r_i, 1, idx_name).alignment = Alignment(horizontal='left')
        ws3.cell(r_i, 2, round(float(val), 4)).alignment = Alignment(horizontal='center')
        ws3.cell(r_i, 3, thresh).alignment = Alignment(horizontal='left')
border_range(ws3, 3, 1, 3+len(thresholds), 3)

# ── Sheet 4: 효과 분해 ───────────────────────────────────────
ws4 = wb.create_sheet('Effect_Decomp')
ws4.column_dimensions['A'].width = 45
ws4.column_dimensions['B'].width = 15

ws4['A1'] = 'Effect Decomposition: urban_disparity → carbon'
ws4['A1'].font = Font(bold=True, size=12)
ws4.merge_cells('A1:B1')
for i, h in enumerate(['Effect Type','Value'], 1):
    ws4.cell(3, i, h)
style_header(ws4, 3, 2, color='4B0082')

effect_data = [
    ('Direct Effect  (urban_disparity → carbon)', round(c, 4)),
    ('Path a  (urban_disparity → UCI)',            round(a, 4)),
    ('Path b  (UCI → carbon)',                     round(b, 4)),
    ('Indirect Effect  (a × b)',                   round(indirect, 4)),
    ('Total Effect  (direct + indirect)',           round(total, 4)),
    ('Reverse path  (UCI → urban_disparity)',       round(d, 4)),
]
for r_i, (label, val) in enumerate(effect_data, 4):
    ws4.cell(r_i, 1, label)
    ws4.cell(r_i, 2, val).alignment = Alignment(horizontal='center')
border_range(ws4, 3, 1, 3+len(effect_data), 2)

# ── Sheet 5: 2SLS 결과 ───────────────────────────────────────
ws5 = wb.create_sheet('2SLS_Results')
for col, w in zip(['A','B','C','D','E','F','G'], [22,13,13,13,13,13,8]):
    ws5.column_dimensions[col].width = w

ws5['A1'] = '2SLS Results (IV Estimation, Robust SE)'
ws5['A1'].font = Font(bold=True, size=12)
ws5.merge_cells('A1:G1')
ws5['A2'] = 'Endogenous: uci_score, disparity_score  |  IV: UCI indicators + Disparity indicators'
ws5['A2'].font = Font(italic=True, size=9, color='555555')
ws5.merge_cells('A2:G2')

h2 = ['Dependent','Variable','Coef.','Std. Err','T-stat','P-value','Sig.']
for i, h in enumerate(h2, 1):
    ws5.cell(4, i, h)
style_header(ws5, 4, len(h2), color='8B0000')

row_idx = 5
carbon_labels = {'build_elec_e_pc': 'Building Electricity',
                 'transport_e_pc':  'Transport Energy',
                 'absor_pc':        'Absorption'}
var_labels = {'uci_score': 'UCI Score', 'disparity_score': 'Disparity Score'}

for cv, res in iv_results.items():
    params_iv = res.params
    se_iv     = res.std_errors
    tstat_iv  = res.tstats
    pval_iv   = res.pvalues
    first_row = True
    for var in ['uci_score', 'disparity_score']:
        dep_label = carbon_labels[cv] if first_row else ''
        p = pval_iv[var]
        ws5.cell(row_idx, 1, dep_label)
        ws5.cell(row_idx, 2, var_labels[var])
        ws5.cell(row_idx, 3, round(params_iv[var], 4)).alignment = Alignment(horizontal='center')
        ws5.cell(row_idx, 4, round(se_iv[var], 4)).alignment     = Alignment(horizontal='center')
        ws5.cell(row_idx, 5, round(tstat_iv[var], 3)).alignment  = Alignment(horizontal='center')
        ws5.cell(row_idx, 6, round(p, 4)).alignment              = Alignment(horizontal='center')
        ws5.cell(row_idx, 7, sig_star(p)).alignment              = Alignment(horizontal='center')
        first_row = False
        row_idx += 1

border_range(ws5, 4, 1, row_idx-1, len(h2))

wb.save(f'{OUT}/SEM_2SLS_Tables.xlsx')
print("엑셀 저장 완료")


# ════════════════════════════════════════════════════════════════
# 그래프 1: Path Diagram — 잠재변수 3개만, 화살표에 beta+star 표시
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')

def draw_box(ax, x, y, w, h, lines, color='#2C5F8A', text_color='white'):
    box = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
                                   boxstyle='round,pad=0.15', linewidth=2,
                                   edgecolor='white', facecolor=color, zorder=3)
    ax.add_patch(box)
    if isinstance(lines, str):
        lines = [lines]
    total_h = len(lines) * 0.32
    for i, line in enumerate(lines):
        ty = y + total_h/2 - 0.32*i - 0.16
        fs = 11 if i == 0 else 9
        fw = 'bold' if i == 0 else 'normal'
        ax.text(x, ty, line, ha='center', va='center',
                fontsize=fs, fontweight=fw, color=text_color, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, label='', color='#333333', lw=2.0, curvature=0.0, label_offset=(0, 0)):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle='->', color=color,
                             connectionstyle=f'arc3,rad={curvature}',
                             linewidth=lw, mutation_scale=16, zorder=2)
    ax.add_patch(arrow)
    if label:
        # 곡선 중간점 보정
        mx = (x1+x2)/2 + label_offset[0]
        my = (y1+y2)/2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='center', fontsize=10,
                color=color, fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          alpha=0.92, edgecolor=color, linewidth=0.8))

# ── 3개 박스 ──────────────────────────────────────────────────
a_p = pv[('UCI',             '~', 'urban_disparity')]
b_p = pv[('carbon',          '~', 'UCI')]
c_p = pv[('carbon',          '~', 'urban_disparity')]
d_p = pv[('urban_disparity', '~', 'UCI')]

draw_box(ax, 2.2, 3.5, 3.4, 1.4,
         ['Urban Disparity', f'({", ".join(vars_disparity)})'], color='#1A5276')
draw_box(ax, 6.5, 3.5, 2.8, 1.4,
         ['UCI', f'({", ".join(vars_uci)})'], color='#117A65')
draw_box(ax, 10.8, 3.5, 3.4, 1.4,
         ['Carbon Emissions', f'({", ".join(vars_carbon)})'], color='#7B241C')

# ── 화살표 ─────────────────────────────────────────────────────
# disparity → UCI (위쪽)
draw_arrow(ax, 3.92, 3.9, 5.1, 3.9,
           f'β={a:.3f}{sig_star(a_p)}', color='#117A65', curvature=-0.2,
           label_offset=(0, 0.52))

# UCI → disparity (아래쪽, 역방향)
draw_arrow(ax, 5.1, 3.1, 3.92, 3.1,
           f'β={d:.3f}{sig_star(d_p)}', color='#1A5276', curvature=-0.2,
           label_offset=(0, -0.52))

# UCI → carbon
draw_arrow(ax, 7.85, 3.9, 9.1, 3.9,
           f'β={b:.3f}{sig_star(b_p)}', color='#7B241C', curvature=-0.2,
           label_offset=(0, 0.48))

# disparity → carbon (직접, 위쪽 큰 아치)
draw_arrow(ax, 3.92, 4.3, 9.1, 4.3,
           f"β={c:.3f}{sig_star(c_p)}\n(direct)", color='#884EA0',
           curvature=-0.28, lw=1.8, label_offset=(0, 1.0))

# ── 효과 분해 박스 ─────────────────────────────────────────────
ax.text(6.5, 1.35,
        f'Indirect (a×b): {a:.3f} × {b:.3f} = {indirect:.3f}\n'
        f"Direct  (c'):   {c:.3f}\n"
        f'Total Effect:   {total:.3f}',
        ha='center', va='center', fontsize=10, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDFEFE',
                  edgecolor='#999999', linewidth=1.2))

ax.set_title('Non-recursive SEM Path Diagram  (Urban Disparity ↔ UCI → Carbon Emissions)',
             fontsize=12, fontweight='bold', pad=12)
ax.text(0.01, 0.01, '*** p<.001  ** p<.01  * p<.05',
        transform=ax.transAxes, fontsize=8.5, color='#666666')

plt.tight_layout()
plt.savefig(f'{OUT}/Fig1_PathDiagram.png', dpi=180, bbox_inches='tight')
plt.close()
print("경로도 저장 완료")


# ════════════════════════════════════════════════════════════════
# 그래프 2: 2SLS 계수 비교 (탄소변수별)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=False)
fig.patch.set_facecolor('#F8F9FA')

carbon_full = {'build_elec_e_pc': 'Building\nElectricity',
               'transport_e_pc':  'Transport\nEnergy',
               'absor_pc':        'Absorption'}
colors_iv = {'uci_score': '#117A65', 'disparity_score': '#1A5276'}
var_names  = {'uci_score': 'UCI Score', 'disparity_score': 'Disparity Score'}

for ax, (cv, res) in zip(axes, iv_results.items()):
    params_iv = res.params
    ci        = res.conf_int()
    pval_iv   = res.pvalues

    vars_plot = ['uci_score', 'disparity_score']
    coefs  = [params_iv[v] for v in vars_plot]
    lowers = [params_iv[v] - ci.loc[v, 'lower'] for v in vars_plot]
    uppers = [ci.loc[v, 'upper'] - params_iv[v] for v in vars_plot]
    clrs   = [colors_iv[v] for v in vars_plot]
    xlabels= [var_names[v] for v in vars_plot]

    bars = ax.bar(xlabels, coefs, color=clrs, width=0.5, edgecolor='white', linewidth=1.2, zorder=3)
    ax.errorbar(xlabels, coefs, yerr=[lowers, uppers], fmt='none',
                color='#333333', capsize=6, linewidth=1.5, zorder=4)

    for bar, v, var in zip(bars, vars_plot, vars_plot):
        p = pval_iv[var]
        star = sig_star(p)
        coef = params_iv[v]
        ypos = coef + (uppers[vars_plot.index(v)] + 0.03) * np.sign(coef) if coef >= 0 \
               else coef - (lowers[vars_plot.index(v)] + 0.03)
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{coef:.3f}{star}',
                ha='center', va='bottom' if coef >= 0 else 'top',
                fontsize=10, fontweight='bold', color='#1B2631')

    ax.axhline(0, color='#888888', linewidth=0.8, linestyle='--')
    ax.set_title(carbon_full[cv], fontsize=11, fontweight='bold')
    ax.set_ylabel('2SLS Coefficient' if ax == axes[0] else '', fontsize=9)
    ax.set_facecolor('#F8F9FA')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    # y축 여유: 텍스트가 축 밖으로 나가지 않도록 padding 추가
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.28
    ax.set_ylim(ymin - pad, ymax + pad)

fig.suptitle('2SLS Estimates by Carbon Emission Variable\n(Robust SE, 95% CI)',
             fontsize=13, fontweight='bold', y=1.01)

legend_patches = [mpatches.Patch(color=c, label=l)
                  for c, l in [('#117A65','UCI Score'), ('#1A5276','Disparity Score')]]
fig.legend(handles=legend_patches, loc='lower center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, -0.05), framealpha=0.9)

plt.tight_layout()
plt.savefig(f'{OUT}/Fig2_2SLS_Coefficients.png', dpi=180, bbox_inches='tight')
plt.close()
print("2SLS 계수도 저장 완료")


# ════════════════════════════════════════════════════════════════
# 그래프 3: SEM 측정모델 Factor Loadings
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
fig.patch.set_facecolor('#F8F9FA')

constructs = [
    ('urban_disparity', vars_disparity, '#1A5276', 'Urban Disparity'),
    ('UCI',             vars_uci,       '#117A65', 'UCI'),
    ('carbon',          vars_carbon,    '#7B241C', 'Carbon'),
]

# semopy: 측정모델은 'indicator ~ latent' 형태로 저장 (lval=indicator, rval=latent)
meas_rows = params_nr[measurement_mask].copy()

for ax, (latent, indicators, color, title) in zip(axes, constructs):
    rows = meas_rows[meas_rows['rval'] == latent]
    loadings = []
    for ind in indicators:
        r = rows[rows['lval'] == ind]
        loadings.append(r['Estimate'].values[0] if len(r) else np.nan)

    pvals = []
    for ind in indicators:
        r = rows[rows['lval'] == ind]
        pvals.append(r['p-value'].values[0] if len(r) else np.nan)

    def is_sig(p):
        try: return float(p) < 0.05
        except: return False
    bar_colors = [color if is_sig(p) else '#AAAAAA' for p in pvals]
    bars = ax.barh(indicators, loadings, color=bar_colors, edgecolor='white', height=0.45, zorder=3)

    # x축 여유: 라벨이 잘리지 않도록
    xmin, xmax = ax.get_xlim()
    x_range = xmax - xmin
    ax.set_xlim(xmin - x_range * 0.05, xmax + x_range * 0.25)

    for bar, lv, p in zip(bars, loadings, pvals):
        if pd.isna(lv): continue
        star = sig_star(p)
        by = bar.get_y() + bar.get_height() / 2
        if lv >= 0:
            ax.text(lv + 0.02, by, f'{lv:.3f}{star}',
                    va='center', ha='left', fontsize=8.5, color='#1B2631')
        else:
            # 음수: 0 오른쪽에 음수값 표기 — y축 겹침 방지
            ax.text(0.02, by, f'{lv:.3f}{star}',
                    va='center', ha='left', fontsize=8.5, color='#1B2631')

    ax.axvline(0, color='#888888', linewidth=0.8, linestyle='--')
    ax.set_title(title, fontsize=11, fontweight='bold', color=color)
    ax.set_xlabel('Factor Loading', fontsize=9)
    ax.set_facecolor('#F8F9FA')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x', alpha=0.3, zorder=0)
    # y축 레이블 폰트 크기 조정 & 겹침 방지
    ax.tick_params(axis='y', labelsize=9, pad=4)

fig.suptitle('SEM Measurement Model — Factor Loadings\n(gray = p≥.05)',
             fontsize=13, fontweight='bold')
plt.tight_layout(pad=1.5)
plt.savefig(f'{OUT}/Fig3_FactorLoadings.png', dpi=180, bbox_inches='tight')
plt.close()
print("측정모델도 저장 완료")


# ════════════════════════════════════════════════════════════════
# 그래프 4: 효과 분해 요약 (Stacked bar)
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor('#F8F9FA')

effect_labels = ['Direct\n(c\')', 'Indirect\n(a×b)', 'Total']
effect_vals   = [c, indirect, total]
eff_colors    = ['#884EA0', '#117A65', '#2C3E50']

bars = ax.bar(effect_labels, effect_vals, color=eff_colors, width=0.45,
              edgecolor='white', linewidth=1.2, zorder=3)
ax.axhline(0, color='#888888', linewidth=0.8, linestyle='--')

for bar, val in zip(bars, effect_vals):
    ypos = val + 0.012 if val >= 0 else val - 0.018
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f'{val:.4f}', ha='center',
            va='bottom' if val >= 0 else 'top',
            fontsize=11, fontweight='bold', color='#1B2631')

# y축 여유: 값 레이블과 축 레이블 겹침 방지
ymin, ymax = ax.get_ylim()
pad = (ymax - ymin) * 0.25
ax.set_ylim(ymin - pad, ymax + pad)

ax.set_title('Effect Decomposition\nurban_disparity → carbon (via UCI)',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Standardized Effect Size', fontsize=10)
ax.set_facecolor('#F8F9FA')
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig(f'{OUT}/Fig4_EffectDecomp.png', dpi=180, bbox_inches='tight')
plt.close()
print("효과분해도 저장 완료")


# ════════════════════════════════════════════════════════════════
# 그래프 5: 개별 관측변수 → Carbon 총 기여도
# 계산: 각 지표의 loading × 해당 잠재변수가 carbon에 미치는 총효과
#   disparity 지표: loading × total_effect(disparity→carbon)
#   UCI 지표:       loading × b (UCI→carbon)
#   carbon 지표:    loading (측정변수 자체가 carbon 잠재변수의 지표)
# ════════════════════════════════════════════════════════════════
total_disp_to_carbon = total   # 직접 + 간접
uci_to_carbon        = b

contrib_data = []
for ind in vars_disparity:
    r = meas_rows[meas_rows['lval'] == ind]
    lv = r['Estimate'].values[0] if len(r) else np.nan
    contrib_data.append({'variable': ind, 'group': 'Urban Disparity',
                         'loading': lv, 'total_contrib': lv * total_disp_to_carbon})
for ind in vars_uci:
    r = meas_rows[meas_rows['lval'] == ind]
    lv = r['Estimate'].values[0] if len(r) else np.nan
    contrib_data.append({'variable': ind, 'group': 'UCI',
                         'loading': lv, 'total_contrib': lv * uci_to_carbon})
for ind in vars_carbon:
    r = meas_rows[meas_rows['lval'] == ind]
    lv = r['Estimate'].values[0] if len(r) else np.nan
    contrib_data.append({'variable': ind, 'group': 'Carbon',
                         'loading': lv, 'total_contrib': lv})

df_contrib = pd.DataFrame(contrib_data)

# 엑셀에 기여도 시트 추가
ws6 = wb.create_sheet('Indicator_Contribution')
ws6.column_dimensions['A'].width = 22
ws6.column_dimensions['B'].width = 18
ws6.column_dimensions['C'].width = 14
ws6.column_dimensions['D'].width = 18

ws6['A1'] = 'Indicator Variable Contribution to Carbon (via Latent Paths)'
ws6['A1'].font = Font(bold=True, size=12)
ws6.merge_cells('A1:D1')
ws6['A2'] = 'Total Contribution = Factor Loading × Latent→Carbon Total Effect'
ws6['A2'].font = Font(italic=True, size=9, color='555555')
ws6.merge_cells('A2:D2')

for i, h in enumerate(['Variable','Group','Factor Loading','Total Contribution'], 1):
    ws6.cell(4, i, h)
style_header(ws6, 4, 4, color='4A235A')

for r_i, row in df_contrib.iterrows():
    ws6.cell(r_i+5, 1, row['variable'])
    ws6.cell(r_i+5, 2, row['group'])
    ws6.cell(r_i+5, 3, safe_round(row['loading'])).alignment   = Alignment(horizontal='center')
    ws6.cell(r_i+5, 4, safe_round(row['total_contrib'])).alignment = Alignment(horizontal='center')
border_range(ws6, 4, 1, len(df_contrib)+5, 4)
wb.save(f'{OUT}/SEM_2SLS_Tables.xlsx')

# 그래프
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.patch.set_facecolor('#F8F9FA')

groups = [
    ('Urban Disparity', '#1A5276', 'loading',       'Factor Loading\n(onto Disparity latent)'),
    ('UCI',             '#117A65', 'loading',        'Factor Loading\n(onto UCI latent)'),
    ('Carbon',          '#7B241C', 'total_contrib',  'Factor Loading\n(onto Carbon latent)'),
]

for ax, (grp, color, col, ylabel) in zip(axes, groups):
    sub = df_contrib[df_contrib['group'] == grp].copy()
    vals = sub[col].values
    labels = sub['variable'].values
    bar_colors = [color if not pd.isna(v) and v != 0 else '#CCCCCC' for v in vals]
    bars = ax.barh(labels, vals, color=bar_colors, edgecolor='white', height=0.42, zorder=3)

    # x축 여유: 숫자 레이블 공간 확보
    xmin, xmax = ax.get_xlim()
    x_range = max(abs(xmin), abs(xmax), 0.1)
    ax.set_xlim(xmin - x_range * 0.05, xmax + x_range * 0.3)

    for bar, v in zip(bars, vals):
        if pd.isna(v): continue
        by = bar.get_y() + bar.get_height() / 2
        if v >= 0:
            ax.text(v + 0.01, by, f'{v:.3f}',
                    va='center', ha='left', fontsize=8.5, color='#1B2631')
        else:
            # 음수: 0 오른쪽에 음수값 표기 — y축 겹침 방지
            ax.text(0.01, by, f'{v:.3f}',
                    va='center', ha='left', fontsize=8.5, color='#1B2631')
    ax.axvline(0, color='#888888', linewidth=0.8, linestyle='--')
    ax.set_title(grp, fontsize=11, fontweight='bold', color=color)
    ax.set_xlabel(ylabel, fontsize=8.5)
    ax.set_facecolor('#F8F9FA')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x', alpha=0.3, zorder=0)
    # y축 레이블 겹침 방지
    ax.tick_params(axis='y', labelsize=9, pad=5)

fig.suptitle('Individual Variable Factor Loadings by Construct\n'
             '(Disparity & UCI: loading onto latent variable  |  Carbon: loading onto carbon latent)',
             fontsize=12, fontweight='bold')
plt.tight_layout(pad=1.5)
plt.savefig(f'{OUT}/Fig5_IndicatorContribution.png', dpi=180, bbox_inches='tight')
plt.close()
print("개별 변수 기여도 저장 완료")


# ════════════════════════════════════════════════════════════════
# ZIP 묶기
# ════════════════════════════════════════════════════════════════
zip_path = 'SEM_PathAnalysis_Results.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUT):
        zf.write(os.path.join(OUT, fname), fname)

print(f"\n완료: {zip_path}")
print("포함 파일:")
with zipfile.ZipFile(zip_path) as zf:
    for name in zf.namelist():
        print(f"  {name}")
