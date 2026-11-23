#!/usr/bin/env bash
# A1 variant sweep: calibration size, QAT epochs / lr, float control. Resumable.
set -u
S=${SCRATCH}
R=${REPO}
V=${WORKSPACE}/v1_view_pre0830
OUT=$S/a1

run() {  # name workdir args...
  local name=$1 wd=$2; shift 2
  local d=$OUT/$name
  if [ -f "$d/quantization_report.json" ] || [ -f "$d/float_control_report.json" ]; then
    echo "skip $name"; return; fi
  mkdir -p "$d"
  echo "$(date +%T) start $name"
  (cd "$wd" && python -m nina_thermal.quantization.pipelines.run_quantize "$@" \
      --out_dir "$d" --log_dir "$S/logs" > "$d/run.out" 2>&1)
  echo "$(date +%T) done $name exit=$?"
}

# ── v1 (main labels + fixed quantization package) ───────────────────────────
W1=$S/wt_v1fix
B1=(--checkpoint $R/models/nina_thermal_frp20000/mobilenetv3s_fire_best.pt
    --roots $V/data $V/data_202408 $V/data_202409 $V/data_202509 --profiles B
    --frp_threshold 20000 --batch_size 16 --num_workers 2 --select_every 5 --no_smoke_test)
run v1_ptq_c1      $W1 "${B1[@]}" --epochs 0 --calib_batches 1
run v1_ptq_c4      $W1 "${B1[@]}" --epochs 0 --calib_batches 4
run v1_ptq_call    $W1 "${B1[@]}" --epochs 0 --calib_batches 1000
run v1_qat_e10_lr5e-5  $W1 "${B1[@]}" --epochs 10 --lr 5e-5
run v1_qat_e40_lr5e-5  $W1 "${B1[@]}" --epochs 40 --lr 5e-5
run v1_qat_e40_lr5e-5_s1 $W1 "${B1[@]}" --epochs 40 --lr 5e-5 --seed 1
run v1_qat_e40_lr5e-5_s2 $W1 "${B1[@]}" --epochs 40 --lr 5e-5 --seed 2
run v1_qat_e40_lr1e-5  $W1 "${B1[@]}" --epochs 40 --lr 1e-5
run v1_qat_e40_lr1e-4  $W1 "${B1[@]}" --epochs 40 --lr 1e-4
run v1_qat_e100_lr5e-5 $W1 "${B1[@]}" --epochs 100 --lr 5e-5
run v1_ctrl_e10_lr5e-5 $W1 "${B1[@]}" --epochs 10 --lr 5e-5 --float_control
run v1_ctrl_e40_lr5e-5 $W1 "${B1[@]}" --epochs 40 --lr 5e-5 --float_control

# ── v4 (A1 branch code) ─────────────────────────────────────────────────────
W4=$S/wt_a1
ROOTS4=$(python3 -c "import json;print(' '.join('$R/'+r for r in json.load(open('$R/models/nina_thermal_frp20000_v4_abc/training_report.json'))['data']['roots']))")
B4=(--checkpoint $R/models/nina_thermal_frp20000_v4_abc/mobilenetv3s_fire_best.pt
    --roots $ROOTS4 --profiles A B C --recall_target 0.88
    --frp_threshold 20000 --frp_negative_max 1000
    --batch_size 16 --num_workers 2 --select_every 5 --no_smoke_test)
run v4_ctrl_e40_lr5e-5 $W4 "${B4[@]}" --epochs 40 --lr 5e-5 --float_control
run v4_ptq_c16     $W4 "${B4[@]}" --epochs 0 --calib_batches 16
run v4_ptq_call    $W4 "${B4[@]}" --epochs 0 --calib_batches 100000
run v4_qat_e40_lr5e-5  $W4 "${B4[@]}" --epochs 40 --lr 5e-5
run v4_qat_e10_lr5e-5  $W4 "${B4[@]}" --epochs 10 --lr 5e-5
run v4_qat_e40_lr1e-5  $W4 "${B4[@]}" --epochs 40 --lr 1e-5
run v4_qat_e40_lr5e-5_call $W4 "${B4[@]}" --epochs 40 --lr 5e-5 --calib_batches 100000
echo "$(date +%T) SWEEP DONE"
