import io, contextlib
import numpy as np, pandas as pd
import train_xgboost as t

def cv(df, cols, n=5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    y = df["label"].values
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=t.RANDOM_STATE)
    a=[]
    for tr,te in skf.split(df[cols].values, y):
        if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2: continue
        m=t._make_model(); m.fit(df[cols].values[tr], y[tr])
        a.append(roc_auc_score(y[te], m.predict_proba(df[cols].values[te])[:,1]))
    a=np.array(a); return a.mean(), a.std()

def build():
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        df = t.build_dataset(t.DATASET_DIR)
    return df

# impact ON (current behaviour) -- just reuse the saved CSV
df_on = pd.read_csv("ott_features.csv")

# impact OFF: neutralise _impact_frame so hand paths run to the wrist-rise end
orig = t._impact_frame
t._impact_frame = lambda seq, search_start, fallback: fallback
df_off = build()
t._impact_frame = orig

for name, df in [("impact OFF (old behaviour)", df_off),
                 ("impact ON  (new change)", df_on)]:
    prim = t.primary_columns(df); allc = t.feature_columns(df)
    pm = cv(df, prim); am = cv(df, allc)
    print(f"{name}")
    print(f"   primary-only : {pm[0]:.3f} +/- {pm[1]:.3f}")
    print(f"   all-expanded : {am[0]:.3f} +/- {am[1]:.3f}")
