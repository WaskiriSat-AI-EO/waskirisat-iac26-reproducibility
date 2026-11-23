#!/usr/bin/env bash
# Score every v1 sweep int8 artifact on the v1 test set (main's evaluator, v1 view).
set -u
S=${SCRATCH}
R=${REPO}
V=${WORKSPACE}/v1_view_pre0830
M=$R/models/nina_thermal_frp20000
cd $S/wt_main
for d in $S/a1/v1_*; do
  [ -f $d/mobilenetv3s_fire_int8.tflite ] || continue
  [ -f $d/test_eval/evaluation_summary.json ] && continue
  TH=$(python3 -c "import json;print(json.load(open('$d/quantization_report.json'))['int8']['threshold'])")
  python -m nina_thermal.evaluation.pipelines.run_evaluate --roots $V/data $V/data_202408 $V/data_202409 $V/data_202509 \
    --checkpoint $M/mobilenetv3s_fire_best.pt --training_report $M/training_report.json --frp_threshold 20000 \
    --tflite $d/mobilenetv3s_fire_int8.tflite --tflite_threshold $TH --out_dir $d/test_eval --log_dir $S/logs > $d/test_eval.out 2>&1
  echo "$(basename $d) exit $?"
done
