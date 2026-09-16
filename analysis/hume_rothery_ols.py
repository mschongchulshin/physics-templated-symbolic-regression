from pathlib import Path
import pandas as pd, numpy as np
from mendeleev import element
import statsmodels.api as sm
from scipy import stats

BASE = str(Path(__file__).resolve().parent.parent)
SRC  = str(Path(BASE) / 'data' / 'CoCrCuFeNi_684.csv')
Path(BASE, 'results', 'generated').mkdir(parents=True, exist_ok=True)
OUT  = str(Path(BASE) / 'results' / 'generated' / 'SuppTable13_recomputed_228.xlsx')
ELS  = ['Co', 'Cr', 'Cu', 'Fe', 'Ni']
DESIGN = np.array([[20,20,0,40,20],[40,15,20,20,5],[35,25,0,35,5],[40,25,5,25,5]], float)

df = pd.read_csv(SRC)
df = df[df['T_K'] == 300].reset_index(drop=True)
Xa = df[[f'{e}(%)' for e in ELS]].values.astype(float)
msk = np.zeros(len(df), bool)
for v in DESIGN:
    msk |= np.all(np.isclose(Xa, v, atol=1e-6), axis=1)
assert msk.sum() == 4
X = Xa[~msk] / 100.0
print(f'compositions used: {len(X)}')

R = 8.314
OMEGA = {('Co','Cr'):-4,('Co','Cu'):6,('Co','Fe'):-1,('Co','Ni'):0,('Cr','Cu'):12,
         ('Cr','Fe'):-1,('Cr','Ni'):-7,('Cu','Fe'):13,('Cu','Ni'):4,('Fe','Ni'):-2}
def h_mix(x):
    h = 0.0
    for i, a in enumerate(ELS):
        for j in range(i+1, 5):
            b = ELS[j]
            k = (a,b) if (a,b) in OMEGA else (b,a)
            h += 4 * OMEGA[k] * x[i] * x[j]
    return h

S_mix = np.array([-R * np.sum(x[x>0] * np.log(x[x>0])) for x in X])
H_mix = np.array([h_mix(x) for x in X])
VEC   = X @ np.array([9, 6, 11, 8, 10])
r_i   = np.array([float(element(e).atomic_radius) for e in ELS]); r_avg = X @ r_i
delta = np.sqrt(((1 - r_i[None,:]/r_avg[:,None])**2 * X).sum(1))
chi_i = np.array([float(element(e).electronegativity('pauling')) for e in ELS]); chi_avg = X @ chi_i
Dchi  = np.sqrt(((chi_i[None,:] - chi_avg[:,None])**2 * X).sum(1))
DESC  = {'S_mix': S_mix, 'ΔH_mix': H_mix, 'VEC': VEC, 'δ': delta, 'Δχ': Dchi}

PROPS = ['atomic_volume','electron_affinity','fusion_heat','metallic_radius',
         'molar_heat_capacity','vdw_radius_batsanov']
P = {e: {p: float(getattr(element(e), p)) for p in PROPS} for e in ELS}
def avg(x, p):   return x @ np.array([P[e][p] for e in ELS])
def dev(x, p):
    v = np.array([P[e][p] for e in ELS]); m = x @ v
    return float(np.sqrt(((v - m)**2 * x).sum()))

F = pd.DataFrame([dict(
        Co=x[0]*100, Cr=x[1]*100, Cu=x[2]*100, Fe=x[3]*100, Ni=x[4]*100,
        atomic_volume_delta=dev(x,'atomic_volume'),
        electron_affinity_delta=dev(x,'electron_affinity'),
        fusion_heat_delta=dev(x,'fusion_heat'),
        metallic_radius_avg=avg(x,'metallic_radius'),
        molar_heat_capacity_delta=dev(x,'molar_heat_capacity'),
        vdw_radius_batsanov_avg=avg(x,'vdw_radius_batsanov'),
        vdw_radius_batsanov_delta=dev(x,'vdw_radius_batsanov')) for x in X])

tables, coefs, rhos = {}, [], []
for drop in (True, False):
    feat = [c for c in F.columns if not (drop and c == 'fusion_heat_delta')]
    rows = []
    for name, y in DESC.items():
        m = sm.OLS(y, sm.add_constant(F[feat])).fit()
        rho = {c: abs(stats.spearmanr(F[c], y).statistic) for c in feat}
        top = sorted(rho, key=rho.get, reverse=True)[:3]
        rows.append({'Descriptor': name, 'OLS R2': round(m.rsquared, 3),
                     'adj R2': round(m.rsquared_adj, 3), 'n_obs': int(m.nobs),
                     'k regressors': len(feat),
                     'Top-3 single-feature predictors (Spearman |rho|)':
                         ', '.join(f'{c} ({rho[c]:.2f})' for c in top)})
        if drop:
            for term in ['const'] + feat:
                coefs.append(dict(Descriptor=name, term=term, coef=m.params[term],
                                  std_err=m.bse[term], t=m.tvalues[term],
                                  p=m.pvalues[term], R2=m.rsquared, n_obs=int(m.nobs)))
            for c in feat:
                rhos.append(dict(Descriptor=name, feature=c, spearman_abs_rho=rho[c]))
    tables['excl' if drop else 'incl'] = pd.DataFrame(rows)

print(tables['excl'].to_string(index=False))
piv = pd.DataFrame(rhos).pivot(index='feature', columns='Descriptor', values='spearman_abs_rho')
with pd.ExcelWriter(OUT, engine='openpyxl') as w:
    tables['excl'].to_excel(w, sheet_name='Table13_11regressors', index=False)
    tables['incl'].to_excel(w, sheet_name='Table13_12regressors', index=False)
    pd.DataFrame(coefs).to_excel(w, sheet_name='OLS_coefficients', index=False)
    piv.reset_index().to_excel(w, sheet_name='Spearman_full', index=False)
print(f'saved -> {OUT}')
