#!/usr/bin/env bash
set -u
S=${SCRATCH}
R=${REPO}
M=$R/models/nina_thermal_frp20000_v4_abc
ROOTS4=$(python3 -c "import json;print(' '.join('$R/'+r for r in json.load(open('$M/training_report.json'))['data']['roots']))")
FTH=$(python3 -c "import json;print(json.load(open('$M/operating_point.json'))['threshold'])" 2>/dev/null || echo 0.6266)
cd $S/wt_a1
for d in $S/a1/v4_*; do
  [ -f $d/mobilenetv3s_fire_int8.tflite ] || continue
  [ -f $d/test_eval/evaluation_summary.json ] && continue
  TH=$(python3 -c "import json;print(json.load(open('$d/quantization_report.json'))['int8']['threshold'])")
  python -m nina_thermal.evaluation.pipelines.run_evaluate --roots $ROOTS4 \
    --checkpoint $M/mobilenetv3s_fire_best.pt --frp_threshold 20000 --frp_negative_max 1000 --threshold $FTH \
    --tflite $d/mobilenetv3s_fire_int8.tflite --tflite_threshold $TH --out_dir $d/test_eval --log_dir $S/logs > $d/test_eval.out 2>&1
  echo "$(date +%T) $(basename $d) exit $?"
done
echo "TEST DONE"
