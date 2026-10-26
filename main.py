"""Discover exploratory latent demand regimes using a Gaussian Hidden Markov Model."""
from __future__ import annotations
import argparse, json, urllib.request, zipfile
from pathlib import Path
DATA_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"

def regime_counts(states):
    counts={}
    for state in states:
        key=int(state); counts[key]=counts.get(key,0)+1
    return dict(sorted(counts.items()))

def load_data(cache_dir="data"):
    import pandas as pd
    path=Path(cache_dir); path.mkdir(parents=True,exist_ok=True); archive=path/"Bike-Sharing-Dataset.zip"
    if not archive.exists(): urllib.request.urlretrieve(DATA_URL,archive)
    with zipfile.ZipFile(archive) as zf:
        member=next((n for n in zf.namelist() if n.endswith("hour.csv")),None)
        if member is None: raise RuntimeError("hour.csv not found in UCI archive")
        with zf.open(member) as stream: df=pd.read_csv(stream)
    df["dteday"]=pd.to_datetime(df["dteday"]); return df.sort_values(["dteday","hr"]).reset_index(drop=True)

def main(output_dir="outputs", states=3, seed=42):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from hmmlearn.hmm import GaussianHMM
    if states<2: raise ValueError("states must be at least 2")
    df=load_data();
    # Sequential observations: log-demand, temperature, humidity and wind speed.
    features=np.column_stack([np.log1p(df["cnt"].to_numpy(float)),df["temp"].to_numpy(float),df["hum"].to_numpy(float),df["windspeed"].to_numpy(float)])
    scaled=StandardScaler().fit_transform(features)
    model=GaussianHMM(n_components=states,covariance_type="diag",n_iter=200,random_state=seed)
    model.fit(scaled); labels=model.predict(scaled)
    df["latent_regime"]=labels
    summary=df.groupby("latent_regime").agg(rows=("cnt","size"),mean_demand=("cnt","mean"),median_demand=("cnt","median"),mean_temperature=("temp","mean"),mean_humidity=("hum","mean")).reset_index()
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); df[["dteday","hr","cnt","latent_regime"]].to_csv(out/"regime_assignments.csv",index=False)
    report={"dataset_rows":len(df),"states":states,"state_counts":regime_counts(labels),"regimes":summary.to_dict(orient="records"),
      "converged":bool(model.monitor_.converged),"iterations":int(model.monitor_.iter),
      "warning":"Latent states are descriptive. Check convergence, state stability, serial assumptions and out-of-sample utility before interpreting them."}
    (out/"metrics.json").write_text(json.dumps(report,indent=2,default=float),encoding="utf-8")
    return report

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir",default="outputs"); p.add_argument("--states",type=int,default=3); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); print(json.dumps(main(a.output_dir,a.states,a.seed),indent=2,default=float))
