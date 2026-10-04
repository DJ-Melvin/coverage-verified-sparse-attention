#!/usr/bin/env python3
"""FA->error analysis: how much output error do the proxy's false accepts cost?

Consumes the per-cell dump from production_proxy_calibration.py (*.cells.npz:
proxy=ĉ, token=c*, err=||O_sel-O_full||, layer, budget, task) and, at a given
tau, reports:
  (1) the false-accept rate (of proxy-accepts) — the headline "76.3%" analog;
  (2) FA cells' c* distribution — are they "just below tau" or catastrophically
      low? (expect most in 0.80-0.90; flag any 0.3-0.5 tail);
  (3) FA cells' output-error quantiles vs true-accept cells — if FA error ~= TA
      error, the false-accepts are benign (the number is scary but harmless).

Pure numpy; run locally on the saved .npz. No GPU.

Usage:
  python eval/analyze_fa_error.py --cells results/camera_ready/<name>.cells.npz --tau 0.90
"""
import argparse
import numpy as np


def q(x, ps):
    return {f"p{p}": float(np.percentile(x, p)) for p in ps} if len(x) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="*.cells.npz from producer")
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--budget", type=float, default=None,
                    help="Filter to a single budget (e.g. 0.05); default all.")
    args = ap.parse_args()

    d = np.load(args.cells, allow_pickle=True)
    proxy, token, err = d["proxy"], d["token"], d["err"]
    budget = d["budget"] if "budget" in d else np.full_like(proxy, -1.0)
    if args.budget is not None:
        m = np.isclose(budget, args.budget)
        proxy, token, err = proxy[m], token[m], err[m]
        print(f"[filter budget={args.budget}] {len(proxy)} cells")

    tau = args.tau
    pa = proxy >= tau            # proxy accept
    ta = token >= tau            # token accept (ground truth)
    FA = pa & ~ta                # false accept: proxy says ok, token says no
    TAcc = pa & ta               # true accept
    n, npa = len(proxy), int(pa.sum())

    print("=" * 66)
    print(f"FA->error analysis   tau={tau}   n_cells={n}")
    print("=" * 66)
    print(f"proxy_accept_rate      = {npa/n:.3f}")
    print(f"false_accept / all     = {int(FA.sum())/n:.4f}")
    print(f"false_accept / proxy_accept = {int(FA.sum())/max(npa,1):.4f}   <- the '76.3%' analog")
    print()

    print("--- FA cells: token coverage c* distribution (are they 'just below', or catastrophic?) ---")
    fc = token[FA]
    if len(fc):
        print(f"  n_FA={len(fc)}  mean={fc.mean():.3f}  " +
              "  ".join(f"{k}={v:.3f}" for k, v in q(fc, [1,5,10,50,90]).items()))
        print(f"  c* in [{tau-0.05:.2f},{tau:.2f}) : {100*np.mean((fc>=tau-0.05)&(fc<tau)):.0f}%  (just below)")
        print(f"  c* in [0.50,{tau-0.05:.2f}) : {100*np.mean((fc>=0.5)&(fc<tau-0.05)):.0f}%")
        print(f"  c* < 0.50 (catastrophic)  : {100*np.mean(fc<0.5):.0f}%   <- WATCH THIS TAIL")
    print()

    print("--- output error ||O_sel - O_full||: FA vs true-accept (is the FA harmless?) ---")
    ps = [50, 90, 99]
    fe, te = err[FA], err[TAcc]
    print(f"  true-accept  n={len(te)}  " + "  ".join(f"{k}={v:.4f}" for k,v in q(te, ps).items()))
    print(f"  false-accept n={len(fe)}  " + "  ".join(f"{k}={v:.4f}" for k,v in q(fe, ps).items()))
    if len(fe) and len(te):
        print(f"  median(FA)/median(TA) error ratio = {np.median(fe)/max(np.median(te),1e-9):.2f}x")
    print(f"  token-reject overall n={int((~ta).sum())}  " +
          "  ".join(f"{k}={v:.4f}" for k,v in q(err[~ta], ps).items()))
    print()
    print("Interpretation guide:")
    print("  - If FA c* median ~ tau-0.05 and FA error ~ TA error → the 76.3% is benign")
    print("    ('scary number, small consequence'); report c* median + error ratio with it.")
    print("  - If FA has a c*<0.5 tail with large error → characterize those cells (which")
    print("    tasks/layers) and how recovery in neighboring steps absorbs them.")


if __name__ == "__main__":
    main()
