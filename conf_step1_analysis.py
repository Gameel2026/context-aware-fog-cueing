import os
import json
import pickle
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

import bmel_dev as bd
import bmel_posthoc as bp
import bmel_sensitivity as bs
import fog_lstm_controller as flc
from fog_pipeline import Config, contiguous_runs, FOG, NOFOG
from fog_study import sequences, event_metrics, CUE_ON
from fog_deep import past_index
from fog_stats import holm

OUT = "results_conference"
QUICK = False                      
N_SHIFTS = 2000
MIN_EPISODES = 10                 
GROUPS = ["Rest", "Balance", "Turning", "Complex gait", "Straight walking"]
CONTROLLERS = ["C-global", "C-oracle", "C-predicted"]
DISABLED = (1.01, 1.01)
warnings.filterwarnings("ignore")


def group_of(task):
    t = str(task)
    if t.startswith("Rest"): return "Rest"
    if t.startswith("MB"): return "Balance"
    if t.startswith("Turning"): return "Turning"
    if t.startswith("Hotspot") or t.startswith("TUG"): return "Complex gait"
    if t.startswith("4MW"): return "Straight walking"
    return None                                                       


def hysteresis_var(p, seqs, th_on, th_off):
    out = np.zeros(len(p), bool)
    for idx in seqs:
        pp, a, b = p[idx].tolist(), th_on[idx].tolist(), th_off[idx].tolist(); o = [False] * len(idx); st = False
        for t in range(len(idx)):
            st = pp[t] >= (b[t] if st else a[t]); o[t] = st
        out[idx] = o
    return out


def thresholds_per_window(grp, pairs, global_pair):
    on = np.full(len(grp), global_pair[0]); off = np.full(len(grp), global_pair[1])
    for g, (a, b) in pairs.items():
        m = grp == g; on[m] = a; off[m] = b
    return on, off


def f1(pred, truth):
    tp = np.sum(pred & truth); fp = np.sum(pred & ~truth); fn = np.sum(~pred & truth)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def select_pairs(p_oof, y_on, seqs_tr, grp_true, tr, n_ep_tr):
    best = {"global": (None, -1.0), **{g: (None, -1.0) for g in GROUPS}}
    t = y_on.astype(bool)
    for th in [(float(a), float(b)) for a, b in bd.GRID]:
        o = hysteresis_var(p_oof, seqs_tr, np.full(len(p_oof), th[0]), np.full(len(p_oof), th[1]))
        v = f1(o[tr], t[tr])
        if v > best["global"][1]: best["global"] = (th, v)
        for g in GROUPS:
            m = tr & (grp_true == g)
            if m.any():
                v = f1(o[m], t[m])
                if v > best[g][1]: best[g] = (th, v)
    glob = best["global"][0]
    pairs = {g: (best[g][0] if (n_ep_tr.get(g, 0) >= MIN_EPISODES and best[g][0] is not None) else DISABLED) for g in GROUPS}
    return glob, pairs


def causal_context(X, seqs):
    F = X.to_numpy(np.float32); F = np.nan_to_num(F)
    m4 = np.zeros_like(F); m8 = np.zeros_like(F); s8 = np.zeros_like(F)
    for idx in seqs:
        d = pd.DataFrame(F[idx])
        m4[idx] = d.rolling(4, min_periods=1).mean().to_numpy()
        m8[idx] = d.rolling(8, min_periods=1).mean().to_numpy()
        s8[idx] = d.rolling(8, min_periods=2).std().fillna(0).to_numpy()
    return np.hstack([F, m4, m8, s8])


def smooth_mode(lab, seqs, n_cls, w=8):
    out = lab.copy()
    for idx in seqs:
        oh = np.eye(n_cls)[lab[idx]]; cs = np.vstack([np.zeros(n_cls), np.cumsum(oh, 0)])
        k = np.arange(1, len(idx) + 1); cnt = cs[k] - cs[np.maximum(k - w, 0)]
        out[idx] = np.argmax(cnt + 0.5 * oh, 1)
    return out


def fa_runs_by_group(on, meta, grp, cfg):
    y2 = meta["y2"].to_numpy(); cnt = {}
    for _, idx in sequences(meta):
        zone = np.zeros(len(idx), bool)
        for s, e in contiguous_runs(y2[idx] == 2):
            zone[max(0, s - cfg.pre_w):min(len(idx), e + cfg.post_w)] = True
        for a, b in contiguous_runs(on[idx]):
            if not zone[a:b].any():
                g = grp[idx[a]] or "(no task)"; cnt[g] = cnt.get(g, 0) + 1
    return cnt


def per_patient(on, y4, meta, cfg):
    y2 = meta["y2"].to_numpy(); pred = np.where(on, FOG, NOFOG); rows = []
    for sub in sorted(meta.subject.unique()):
        m = meta.subject.eq(sub).to_numpy(); e = event_metrics(y2, pred, meta, cfg, m)
        ons = bd.eligible_onsets(y2, meta, cfg, m)
        rows.append({"subject": sub, "false_alarms_per_hour": e["false_alarms_per_hour"], "episodes": e["episodes"],
                     "coverage_%": e["detected_%"] if e["episodes"] else np.nan,
                     "timely_%": bd.timely_elig(on, ons, cfg) if ons else np.nan})
    return pd.DataFrame(rows)


def wtest(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 5 or np.allclose(a[ok], b[ok]): return np.nan, int(ok.sum())
    return float(wilcoxon(a[ok], b[ok]).pvalue), int(ok.sum())


def main():
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",))
    epochs = json.load(open("frozen_config.json"))["lstm_final_epochs"]
    n_shifts = 20 if QUICK else N_SHIFTS
    X, meta, y4, _ = pickle.load(open(os.path.join(bp.TR, "T_probs_defog.pkl"), "rb"))
    y2 = meta["y2"].to_numpy(); y_on = np.isin(y4, CUE_ON).astype(int); g = meta.subject.to_numpy()
    tp = os.path.join(OUT, "defog_task_per_window.npy")
    if os.path.exists(tp):
        task = np.load(tp, allow_pickle=True)
    else:
        from conf_step0_inspect import task_labels
        task = task_labels(meta); np.save(tp, task)
    grp = np.array([group_of(t) for t in task], dtype=object)
    unmapped = sorted(set(task[grp == None]) - {"(no task)"})          
    if unmapped: print("WARNING - tasks not mapped to a group:", unmapped)
    seqs_all = [idx for _, idx in sequences(meta)]
    onset_win = np.array([idx[s] for idx in seqs_all for s, _ in contiguous_runs(y2[idx] == 2)], int)
    print(f"defog: {len(np.unique(g))} patients, {len(meta)} windows, {len(onset_win)} episodes; LSTM epochs {epochs}")
    print("windows per group:", pd.Series([x or "(no task)" for x in grp]).value_counts().to_dict())

    F = np.nan_to_num(X.to_numpy(np.float64)); P8 = past_index(meta, 8)
    C = causal_context(X, seqs_all); y_act = np.array([GROUPS.index(x) if x else -1 for x in grp])
    ref_path = os.path.join(bp.OUT, "ref_lstm_defog.npy")
    p_ref = np.load(ref_path) if os.path.exists(ref_path) else None
    print("test probabilities:", "reused from " + ref_path if p_ref is not None else "trained here")

    p_test = np.zeros(len(y4)); act_raw = np.zeros(len(y4), int); on = {c: np.zeros(len(y4), bool) for c in CONTROLLERS}
    choices = []; act_smooth = np.zeros(len(y4), int)
    splits = list(GroupKFold(min(10, len(np.unique(g)))).split(X, groups=g))[: (2 if QUICK else None)]
    for k, (tr_i, te_i) in enumerate(splits):
        t0 = time.time(); ck = os.path.join(OUT, f"ckpt_fold{k}.pkl")
        tr = np.zeros(len(y4), bool); tr[tr_i] = True; te = np.zeros(len(y4), bool); te[te_i] = True
        if os.path.exists(ck):
            oof, pt, ar = pickle.load(open(ck, "rb"))
        else:
            oof = np.zeros(len(y4))                                   
            for ia, ib in GroupKFold(3).split(tr_i, groups=g[tr_i]):
                Z = flc.standardise(F, tr_i[ia], P8)
                m, _, _ = flc.train(Z, y_on, tr_i[ia], epochs, cfg.seed); oof[tr_i[ib]] = flc.prob(m, Z[tr_i[ib]])
            if p_ref is not None:
                pt = p_ref[te_i]
            else:
                Z = flc.standardise(F, tr_i, P8); m, _, _ = flc.train(Z, y_on, tr_i, epochs, cfg.seed); pt = flc.prob(m, Z[te_i])
            fit = tr & (y_act >= 0)                                   
            rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=cfg.seed)
            rf.fit(C[fit], y_act[fit]); ar = rf.predict(C[te_i]).astype(int)
            pickle.dump((oof, pt, ar), open(ck, "wb"))
        p_test[te_i] = pt; act_raw[te_i] = ar
        seqs_tr = [idx for idx in seqs_all if tr[idx[0]]]
        seqs_te = [idx for idx in seqs_all if te[idx[0]]]
        ons_tr = onset_win[tr[onset_win]]
        n_ep_tr = pd.Series([grp[i] for i in ons_tr if grp[i]]).value_counts().to_dict()
        glob, pairs = select_pairs(oof, y_on, seqs_tr, grp, tr, n_ep_tr)
        act_s = act_raw.copy(); act_s[te_i] = smooth_mode(act_raw, seqs_te, len(GROUPS))[te_i]
        grp_pred = np.array([GROUPS[i] for i in act_s], dtype=object)
        for c, gr in [("C-global", np.array([None] * len(y4), dtype=object)), ("C-oracle", grp), ("C-predicted", grp_pred)]:
            a, b = thresholds_per_window(gr, pairs if c != "C-global" else {}, glob)
            on[c][te_i] = hysteresis_var(p_test, seqs_te, a, b)[te_i]
        act_smooth[te_i] = act_s[te_i]
        choices.append({"fold": k, "group": "global", "theta_on": glob[0], "theta_off": glob[1], "train_episodes": len(ons_tr)})
        for gname, (a, b) in pairs.items():
            choices.append({"fold": k, "group": gname, "theta_on": a, "theta_off": b, "train_episodes": n_ep_tr.get(gname, 0),
                            "disabled": (a, b) == DISABLED})
        print(f"  fold {k}: global {glob}, groups {pairs}  ({(time.time() - t0) / 60:.1f} min)")

    ev = np.zeros(len(y4), bool); ev[np.concatenate([s[1] for s in splits])] = True
    meta_e = meta[ev].reset_index(drop=True); y4_e = y4[ev]; grp_e = grp[ev]; onsets = bd.eligible_onsets(meta_e["y2"].to_numpy(), meta_e, cfg)
    fm = bs.FastMetrics(meta_e, cfg, onsets); segs = bs.recording_segments(meta_e)
    rows, per, fag = [], [], []
    hours = {gname: (grp_e == gname).sum() * cfg.win_sec / 3600 for gname in GROUPS}
    for c in CONTROLLERS:
        o = on[c][ev]; s = bd.summarise(y4_e, o, meta_e, cfg, onsets)
        sg = bs.surrogate_segments(o, meta_e, cfg, onsets, segs, n_shifts, fm=fm)
        rows.append({"controller": c, "cue_specificity": s["cue_specificity"], "cue_sensitivity": s["cue_sensitivity"],
                     "false_alarms_per_hour": s["false_alarms_per_hour"], "coverage_%": s["detected_%"],
                     "coverage_surrogate": sg["detection_surrogate_mean"], "p_coverage": sg["detection_p"],
                     "timely_%": s["timely_eligible_%"], "timely_surrogate": sg["timely_surrogate_mean"], "p_timely": sg["timely_p"],
                     "eligible_episodes": s["eligible_episodes"], "episodes": s["episodes"], "cue_on_%": 100 * o.mean()})
        pp = per_patient(o, y4_e, meta_e, cfg); pp["controller"] = c; per.append(pp)
        cnt = fa_runs_by_group(o, meta_e, grp_e, cfg)
        for gname in GROUPS:
            fag.append({"controller": c, "group": gname, "hours": hours[gname], "false_alarms": cnt.get(gname, 0),
                        "false_alarms_per_hour": cnt.get(gname, 0) / max(hours[gname], 1e-9)})
    C1 = pd.DataFrame(rows); C2 = pd.concat(per); C4 = pd.DataFrame(fag)

    W = {c: C2[C2.controller == c].set_index("subject") for c in CONTROLLERS}; tests = []
    for metric in ["false_alarms_per_hour", "coverage_%"]:
        fam = []
        for c in ["C-predicted", "C-oracle"]:
            a = W[c][metric].to_numpy(); b = W["C-global"][metric].to_numpy(); p, n = wtest(a, b)
            fam.append({"comparison": f"{c} vs C-global", "metric": metric, "n_patients": n,
                        "median_difference": float(np.nanmedian(a - b)), "mean_controller": float(np.nanmean(a)),
                        "mean_global": float(np.nanmean(b)), "p_wilcoxon": p})
        ph = holm(np.array([f["p_wilcoxon"] for f in fam]))
        for f, h in zip(fam, ph): f["p_holm"] = h
        tests += fam
    C3 = pd.DataFrame(tests)
    pr = C3[(C3.comparison == "C-predicted vs C-global") & (C3.metric == "false_alarms_per_hour")].iloc[0]
    cov_drop = C1.set_index("controller").loc["C-global", "coverage_%"] - C1.set_index("controller").loc["C-predicted", "coverage_%"]
    fa_lower = C1.set_index("controller").loc["C-predicted", "false_alarms_per_hour"] < C1.set_index("controller").loc["C-global", "false_alarms_per_hour"]
    supported = bool(fa_lower and pr.p_holm < 0.05 and cov_drop <= 5.0)
    C3["verdict_pre_specified"] = ""; C3.loc[0, "verdict_pre_specified"] = (
        f"{'SUPPORTED' if supported else 'NOT supported'}: FA/h lower={fa_lower}, Holm p={pr.p_holm:.4g}, coverage drop={cov_drop:.2f} pp")

    yt = y_act[ev]; ps = act_smooth[ev]; okm = yt >= 0
    rec = [{"group": gname, "windows": int((yt == i).sum()), "recall_%": 100 * float((ps[yt == i] == i).mean()) if (yt == i).any() else np.nan,
            "recall_raw_%": 100 * float((act_raw[ev][yt == i] == i).mean()) if (yt == i).any() else np.nan} for i, gname in enumerate(GROUPS)]
    rec.append({"group": "ALL (accuracy)", "windows": int(okm.sum()), "recall_%": 100 * float((ps[okm] == yt[okm]).mean()),
                "recall_raw_%": 100 * float((act_raw[ev][okm] == yt[okm]).mean())})
    rec.append({"group": "balanced accuracy", "windows": int(okm.sum()), "recall_%": float(np.nanmean([r["recall_%"] for r in rec[:len(GROUPS)]])),
                "recall_raw_%": float(np.nanmean([r["recall_raw_%"] for r in rec[:len(GROUPS)]]))})
    cm = pd.crosstab(pd.Series([GROUPS[i] for i in yt[okm]], name="annotated"), pd.Series([GROUPS[i] for i in ps[okm]], name="predicted"))

    C1.round(4).to_csv(f"{OUT}/C1_summary.csv", index=False); C2.round(4).to_csv(f"{OUT}/C2_per_patient.csv", index=False)
    C3.round(5).to_csv(f"{OUT}/C3_tests.csv", index=False); C4.round(3).to_csv(f"{OUT}/C4_fa_by_group.csv", index=False)
    pd.DataFrame(choices).to_csv(f"{OUT}/C5_thresholds.csv", index=False)
    pd.DataFrame(rec).round(2).to_csv(f"{OUT}/C6_recogniser.csv", index=False); cm.to_csv(f"{OUT}/C6_confusion.csv")
    print("\nC1 pooled results (surrogate: %d shifts)\n" % n_shifts, C1.round(3).to_string(index=False))
    print("\nC3 per-patient tests\n", C3.round(4).to_string(index=False))
    print("\nC4 false alarms per hour by group\n", C4.pivot(index="group", columns="controller", values="false_alarms_per_hour").round(2).to_string())
    print("\nC6 activity recogniser\n", pd.DataFrame(rec).round(1).to_string(index=False), "\n", cm.to_string())
    print("\nThresholds per fold\n", pd.DataFrame(choices).to_string(index=False))
    print(f"\nPRE-SPECIFIED VERDICT: {C3.loc[0, 'verdict_pre_specified']}\nFiles in {OUT}/")


if __name__ == "__main__":
    main()