import json, glob, os, sys
S = sys.argv[1]; prefix = sys.argv[2]
rows = []
for d in sorted(glob.glob(f"{S}/a1/{prefix}_*")):
    n = os.path.basename(d)
    q = f"{d}/quantization_report.json"; c = f"{d}/float_control_report.json"
    if os.path.exists(q):
        r = json.load(open(q)); p = r["parity"]; a = r["acceptance"]["measured"]
        cp = r.get("conversion_parity", {}); sel = r.get("checkpoint_selection", {})
        rows.append(f"| {n} | {r['quantization']['method']} | {r['quantization'].get('calibration',{}).get('images')} | {r['fp32']['roc_auc']:.3f} | {r['int8']['roc_auc']:.3f} | {a['fbeta_drop']*100:+.1f} | {a['recall_drop']*100:+.1f} | {a['sqnr_db']} | {a['kl_nats']:.3f} | {p.get('vs_fake_quant',{}).get('sqnr_db')} | {cp.get('score_corr_converted_vs_fake_quant')} | {sel.get('selected_epoch')} | {'PASS' if r['acceptance']['passed'] else 'fail'} |")
    elif os.path.exists(c):
        r = json.load(open(c)); dd = r["drift_vs_original"]
        rows.append(f"| {n} | fp32 fine-tune (no quant) | - | {r['fp32_original']['roc_auc']:.3f} | {r['fp32_fine_tuned']['roc_auc']:.3f} | {dd['fbeta_drop']*100:+.1f} | {dd['recall_drop']*100:+.1f} | {dd['sqnr_db']} | {dd['kl_divergence_nats']:.3f} | - | corr {dd['score_corr']} | {r['fine_tune']['selected_epoch']} | - |")
print("| run | method | calib imgs | fp32 val AUC | int8 val AUC | F-beta drop pp | recall drop pp | SQNR vs fp32 dB | KL nats | SQNR vs fake-quant dB | conv corr | sel. epoch | gate |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
