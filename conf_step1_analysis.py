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
RUN = ["defog", "FoG-STAR"]       
FOGSTAR_GROUP = {1: "Walking", 6: "Turning", 7: "Turning", 3: "Standing/transitions", 5: "Standing/transitions",
                 2: "Sitting", 4: "Sitting"}                          
CONTROLLERS = ["C-global", "C-oracle", "C-predicted", "C-gate"]   
DISABLED = (1.01, 1.01)
warnings.filterwarnings("ignore")


def fogstar_group(code):
    return FOGSTAR_GROUP.get(int(code))


DATASETS = {
    "defog": {"pkl": "T_probs_defog.pkl", "ref": "ref_lstm_defog.npy", "labels": "defog_task_per_window.npy",
              "groups": ["Rest", "Balance", "Turning", "Complex gait", "Straight walking"],
              "mapper": None, "few_policy": "disable", "always_disabled": [], "prefix": ""},
    "FoG-STAR": {"pkl": "T_probs_FoG-STAR.pkl", "ref": "ref_lstm_FoG-STAR.npy", "labels": "fogstar_activity_per_window.npy",
                 "groups": ["Sitting", "Standing/transitions", "Walking", "Turning"],
                 "mapper": fogstar_group, "few_policy": "global", "always_disabled": ["Sitting"], "prefix": "FS_"},
}
GROUPS = DATASETS["defog"]["groups"]; FEW_POLICY = "disable"; ALWAYS_DISABLED = []


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
    few = DISABLED if FEW_POLICY == "disable" else glob
    pairs = {g: (best[g][0] if (n_ep_tr.get(g, 0) >= MIN_EPISODES and best[g][0] is not None) else few) for g in GROUPS}
    for g in ALWAYS_DISABLED:
        pairs[g] = DISABLED
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


def coverage_by_group(on, meta, grp):
    y2 = meta["y2"].to_numpy(); n, d = {}, {}
    for _, idx in sequences(meta):
        for s, e in contiguous_runs(y2[idx] == 2):
            g = grp[idx[s]] or "(no task)"; n[g] = n.get(g, 0) + 1; d[g] = d.get(g, 0) + int(on[idx[s:e]].any())
    return n, d


def task_segments(meta, task):
    seg = np.full(len(meta), -1); k = 0
    for _, idx in sequences(meta):
        start = 0
        for j in range(1, len(idx) + 1):
            if j == len(idx) or task[idx[j]] != task[idx[start]]:
                if task[idx[start]] != "(no task)":
                    seg[idx[start:j]] = k; k += 1
                start = j
    return seg


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
    for ds in RUN:
        print(f"\n==================== {ds} ====================")
        run(ds)


def run(ds):
    global GROUPS, FEW_POLICY, ALWAYS_DISABLED
    D = DATASETS[ds]; GROUPS, FEW_POLICY, ALWAYS_DISABLED = D["groups"], D["few_policy"], D["always_disabled"]
    pre = os.path.join(OUT, D["prefix"])
    os.makedirs(OUT, exist_ok=True); pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    cfg = Config(causal_filter=True, sensors=("tr",))
    epochs = json.load(open("frozen_config.json"))["lstm_final_epochs"]
    n_shifts = 20 if QUICK else N_SHIFTS
    X, meta, y4, _ = pickle.load(open(os.path.join(bp.TR, D["pkl"]), "rb"))
    y2 = meta["y2"].to_numpy(); y_on = np.isin(y4, CUE_ON).astype(int); g = meta.subject.to_numpy()
    tp = os.path.join(OUT, D["labels"])
    if os.path.exists(tp):
        task = np.load(tp, allow_pickle=True)
    elif ds == "defog":
        from conf_step0_inspect import task_labels
        task = task_labels(meta); np.save(tp, task)
    else:
        raise FileNotFoundError(f"{tp} missing - run conf_step0b_fogstar_inventory.py first")
    assert len(task) == len(meta), "activity labels do not match the windows"
    grp = np.array([(D["mapper"] or group_of)(t) for t in task], dtype=object)
    unmapped = sorted(set(task[grp == None]) - {"(no task)", 0})         
    if unmapped: print("WARNING - tasks not mapped to a group:", unmapped)
    seqs_all = [idx for _, idx in sequences(meta)]
    onset_win = np.array([idx[s] for idx in seqs_all for s, _ in contiguous_runs(y2[idx] == 2)], int)
    print(f"{ds}: {len(np.unique(g))} patients, {len(meta)} windows, {len(onset_win)} episodes; LSTM epochs {epochs}")
    print("windows per group:", pd.Series([x or "(no task)" for x in grp]).value_counts().to_dict())

    F = np.nan_to_num(X.to_numpy(np.float64)); P8 = past_index(meta, 8)
    C = causal_context(X, seqs_all); y_act = np.array([GROUPS.index(x) if x else -1 for x in grp])
    ref_path = os.path.join(bp.OUT, D["ref"])
    p_ref = np.load(ref_path) if os.path.exists(ref_path) else None
    print("test probabilities:", "reused from " + ref_path if p_ref is not None else "trained here")

    p_test = np.zeros(len(y4)); act_raw = np.zeros(len(y4), int); on = {c: np.zeros(len(y4), bool) for c in CONTROLLERS}
    choices = []; act_smooth = np.zeros(len(y4), int)
    th_or = np.full((len(y4), 2), np.nan); th_pr = np.full((len(y4), 2), np.nan)
    splits = list(GroupKFold(min(10, len(np.unique(g)))).split(X, groups=g))[: (2 if QUICK else None)]
    for k, (tr_i, te_i) in enumerate(splits):
        t0 = time.time(); ck = f"{pre}ckpt_fold{k}.pkl"
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
        gate = {gn: DISABLED for gn, pr in pairs.items() if pr == DISABLED}    
        for c, gr, pr in [("C-global", np.array([None] * len(y4), dtype=object), {}), ("C-oracle", grp, pairs),
                          ("C-predicted", grp_pred, pairs), ("C-gate", grp_pred, gate)]:
            a, b = thresholds_per_window(gr, pr, glob)
            on[c][te_i] = hysteresis_var(p_test, seqs_te, a, b)[te_i]
            if c == "C-oracle": th_or[te_i] = np.c_[a, b][te_i]
            if c == "C-predicted": th_pr[te_i] = np.c_[a, b][te_i]
        act_smooth[te_i] = act_s[te_i]
        choices.append({"fold": k, "group": "global", "theta_on": glob[0], "theta_off": glob[1], "train_episodes": len(ons_tr)})
        for gname, (a, b) in pairs.items():
            choices.append({"fold": k, "group": gname, "theta_on": a, "theta_off": b, "train_episodes": n_ep_tr.get(gname, 0),
                            "disabled": (a, b) == DISABLED})
        print(f"  fold {k}: global {glob}, groups {pairs}  ({(time.time() - t0) / 60:.1f} min)")

    ev = np.zeros(len(y4), bool); ev[np.concatenate([s[1] for s in splits])] = True
    meta_e = meta[ev].reset_index(drop=True); y4_e = y4[ev]; grp_e = grp[ev]; onsets = bd.eligible_onsets(meta_e["y2"].to_numpy(), meta_e, cfg)
    fm = bs.FastMetrics(meta_e, cfg, onsets); segs = bs.recording_segments(meta_e)
    tsegs = task_segments(meta_e, np.asarray(task, dtype=object)[ev]) if ds == "defog" else None
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
        if tsegs is not None:                                           
            st = bs.surrogate_segments(o, meta_e, cfg, onsets, tsegs, n_shifts, fm=fm)
            rows[-1].update({"coverage_surrogate_task": st["detection_surrogate_mean"], "p_coverage_task": st["detection_p"],
                             "timely_surrogate_task": st["timely_surrogate_mean"], "p_timely_task": st["timely_p"]})
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
                        "mean_reference": float(np.nanmean(b)), "p_wilcoxon": p})
        ph = holm(np.array([f["p_wilcoxon"] for f in fam]))
        for f, h in zip(fam, ph): f["p_holm"] = h
        tests += fam
    sec = []                                                            
    for metric in ["false_alarms_per_hour", "coverage_%"]:
        a = W["C-predicted"][metric].to_numpy(); b = W["C-gate"][metric].to_numpy(); p, n = wtest(a, b)
        sec.append({"comparison": "C-predicted vs C-gate", "metric": metric, "n_patients": n,
                    "median_difference": float(np.nanmedian(a - b)), "mean_controller": float(np.nanmean(a)),
                    "mean_reference": float(np.nanmean(b)), "p_wilcoxon": p})
    for f, h in zip(sec, holm(np.array([f["p_wilcoxon"] for f in sec]))): f["p_holm"] = h
    tests += sec
    C3 = pd.DataFrame(tests)
    pr = C3[(C3.comparison == "C-predicted vs C-global") & (C3.metric == "false_alarms_per_hour")].iloc[0]
    cov_drop = C1.set_index("controller").loc["C-global", "coverage_%"] - C1.set_index("controller").loc["C-predicted", "coverage_%"]
    fa_lower = C1.set_index("controller").loc["C-predicted", "false_alarms_per_hour"] < C1.set_index("controller").loc["C-global", "false_alarms_per_hour"]
    if ds == "defog":
        label = "SUPPORTED" if (fa_lower and pr.p_holm < 0.05 and cov_drop <= 5.0) else "NOT supported"
    else:                                                               
        label = "REPLICATED (direction)" if (fa_lower and cov_drop <= 5.0) else "NOT replicated"
    C3["verdict_pre_specified"] = ""; C3.loc[0, "verdict_pre_specified"] = (
        f"{label}: FA/h lower={fa_lower}, Holm p={pr.p_holm:.4g}, coverage drop={cov_drop:.2f} pp")
    S1 = C1.set_index("controller"); pg = C3[(C3.comparison == "C-predicted vs C-gate") & (C3.metric == "false_alarms_per_hour")].iloc[0]
    g_lower = S1.loc["C-predicted", "false_alarms_per_hour"] < S1.loc["C-gate", "false_alarms_per_hour"]
    g_drop = S1.loc["C-gate", "coverage_%"] - S1.loc["C-predicted", "coverage_%"]
    g_ok = g_lower and g_drop <= 5.0 and (pg.p_holm < 0.05 if ds == "defog" else True)
    gate_verdict = (f"{'SUPPORTED' if g_ok else 'NOT supported'}: FA/h lower than C-gate={g_lower}, "
                    f"Holm p={pg.p_holm:.4g}, coverage drop vs C-gate={g_drop:.2f} pp")
    C3.loc[C3.index[-2], "verdict_pre_specified"] = "activity-specific thresholds beyond muting: " + gate_verdict

    yt = y_act[ev]; ps = act_smooth[ev]; okm = yt >= 0
    rec = [{"group": gname, "windows": int((yt == i).sum()), "recall_%": 100 * float((ps[yt == i] == i).mean()) if (yt == i).any() else np.nan,
            "recall_raw_%": 100 * float((act_raw[ev][yt == i] == i).mean()) if (yt == i).any() else np.nan} for i, gname in enumerate(GROUPS)]
    rec.append({"group": "ALL (accuracy)", "windows": int(okm.sum()), "recall_%": 100 * float((ps[okm] == yt[okm]).mean()),
                "recall_raw_%": 100 * float((act_raw[ev][okm] == yt[okm]).mean())})
    rec.append({"group": "balanced accuracy", "windows": int(okm.sum()), "recall_%": float(np.nanmean([r["recall_%"] for r in rec[:len(GROUPS)]])),
                "recall_raw_%": float(np.nanmean([r["recall_raw_%"] for r in rec[:len(GROUPS)]]))})
    cm = pd.crosstab(pd.Series([GROUPS[i] for i in yt[okm]], name="annotated"), pd.Series([GROUPS[i] for i in ps[okm]], name="predicted"))

    oe, pe = on["C-oracle"][ev], on["C-predicted"][ev]
    n_o, d_o = coverage_by_group(oe, meta_e, grp_e); _, d_p = coverage_by_group(pe, meta_e, grp_e)
    agree = np.all(np.isclose(th_or[ev], th_pr[ev]), 1)
    fa_o = C4[C4.controller == "C-oracle"].set_index("group")["false_alarms_per_hour"]
    fa_p = C4[C4.controller == "C-predicted"].set_index("group")["false_alarms_per_hour"]
    c7 = []
    for i, gname in enumerate(GROUPS):
        m = grp_e == gname
        c7.append({"group": gname, "hours": hours[gname], "recogniser_recall_%": rec[i]["recall_%"],
                   "threshold_agreement_%": 100 * agree[m].mean() if m.any() else np.nan,
                   "fa_per_hour_oracle": fa_o.get(gname, np.nan), "fa_per_hour_predicted": fa_p.get(gname, np.nan),
                   "episodes": n_o.get(gname, 0),
                   "coverage_oracle_%": 100 * d_o.get(gname, 0) / n_o[gname] if n_o.get(gname) else np.nan,
                   "coverage_predicted_%": 100 * d_p.get(gname, 0) / n_o[gname] if n_o.get(gname) else np.nan})
    c7.append({"group": "ALL", "hours": float(sum(hours.values())), "recogniser_recall_%": rec[len(GROUPS)]["recall_%"],
               "threshold_agreement_%": 100 * agree.mean(),
               "fa_per_hour_oracle": S1.loc["C-oracle", "false_alarms_per_hour"], "fa_per_hour_predicted": S1.loc["C-predicted", "false_alarms_per_hour"],
               "episodes": int(sum(n_o.values())), "coverage_oracle_%": S1.loc["C-oracle", "coverage_%"],
               "coverage_predicted_%": S1.loc["C-predicted", "coverage_%"]})
    C7 = pd.DataFrame(c7)
    pp_acc = pd.DataFrame({"subject": meta_e.subject.to_numpy(), "ok": (ps == yt), "lab": okm}).query("lab").groupby("subject")["ok"].mean() * 100
    gap = (W["C-predicted"]["false_alarms_per_hour"] - W["C-oracle"]["false_alarms_per_hour"]).rename("fa_gap_pred_minus_oracle")
    C7b = pd.concat([pp_acc.rename("recogniser_accuracy_%"), gap], axis=1).reset_index()
    from scipy.stats import spearmanr
    rho = spearmanr(C7b["recogniser_accuracy_%"], C7b["fa_gap_pred_minus_oracle"], nan_policy="omit")
    C7.round(3).to_csv(f"{pre}C7_recognition_value.csv", index=False); C7b.round(3).to_csv(f"{pre}C7_per_patient.csv", index=False)

    C1.round(4).to_csv(f"{pre}C1_summary.csv", index=False); C2.round(4).to_csv(f"{pre}C2_per_patient.csv", index=False)
    C3.round(5).to_csv(f"{pre}C3_tests.csv", index=False); C4.round(3).to_csv(f"{pre}C4_fa_by_group.csv", index=False)
    pd.DataFrame(choices).to_csv(f"{pre}C5_thresholds.csv", index=False)
    pd.DataFrame(rec).round(2).to_csv(f"{pre}C6_recogniser.csv", index=False); cm.to_csv(f"{pre}C6_confusion.csv")
    print("\nC1 pooled results (surrogate: %d shifts)\n" % n_shifts, C1.round(3).to_string(index=False))
    print("\nC3 per-patient tests\n", C3.round(4).to_string(index=False))
    print("\nC4 false alarms per hour by group\n", C4.pivot(index="group", columns="controller", values="false_alarms_per_hour").round(2).to_string())
    print("\nC6 activity recogniser\n", pd.DataFrame(rec).round(1).to_string(index=False), "\n", cm.to_string())
    print("\nC7 value of activity recognition (oracle vs predicted)\n", C7.round(2).to_string(index=False))
    print(f"   per patient: Spearman rho(recogniser accuracy, FA/h gap) = {rho.statistic:.3f}, p = {rho.pvalue:.4g}")
    print(f"\n{ds} C-gate check: {gate_verdict}")
    print("\nThresholds per fold\n", pd.DataFrame(choices).to_string(index=False))
    print(f"\n{ds} PRE-SPECIFIED VERDICT: {C3.loc[0, 'verdict_pre_specified']}\nFiles: {pre}C*.csv")


if __name__ == "__main__":
    main()
