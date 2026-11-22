#!/usr/bin/env bash
# scensim roadmap 094 P4, the stock side: the lead-brake task, trained by Isaac Lab's own train
# CLI under wheeledlab_bench, with the same launcher gate as P1 (bench/p1/run_p1.sh).
#
#   bench/p4_leadbrake/run_p4.sh <stage> <out-dir>
#
# stages (each writes under <out-dir>/<stage>/):
#   smoke   3 iterations at 64 envs, through the harness
#   seeds   5 seeds (1-5) at 1024 envs, 1500 iterations, a checkpoint every 50; then each
#           seed's checkpoints evaluated every 100 iterations up to its first pass, and its last
#   sweep   throughput at N = 256, 1024, 4096 (60 iterations, 20 warm-up); timers sync, and
#           off at 1024 (the harness's overhead)
#
# Host settings, from the environment (as run_p1.sh): OV (the Isaac Lab overlay's python,
# required), ISAACLAB (its source checkout, required), GPU (default 0), IDLE_OTHERS=1, SHARED=1
# (a shared GPU: samples-to-pass is still a measurement, wall clock is not).
set -euo pipefail
stage=${1:?usage: run_p4.sh smoke|seeds|sweep <out-dir>}
out=${2:?usage: run_p4.sh <stage> <out-dir>}
OV=${OV:?OV: the Isaac Lab overlay python, e.g. /var/tmp/aeon/isaaclab-3.0.0/venv/bin/python}
ISAACLAB=${ISAACLAB:?ISAACLAB: the Isaac Lab checkout, e.g. /var/tmp/aeon/isaaclab-3.0.0/src}
GPU=${GPU:-0}
here=$(cd "$(dirname "$0")" && pwd)
export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}
gate=(--require-idle); [ "${IDLE_OTHERS:-0}" = 1 ] && gate+=(--idle-others-ok)
[ "${SHARED:-0}" = 1 ] && gate=(--allow-shared)
out=$(mkdir -p "$out/$stage" && cd "$out" && pwd)
export P4_LEADBRAKE_CACHE=$out/cache

# one training run through the harness, in its own directory (Isaac Lab writes logs/ there):
# <name> <timers> <seed> <n_envs> <iterations> <warmup>
run() {
    local name=$1 timers=$2 seed=$3 n=$4 iters=$5 warmup=$6
    local d="$out/$stage/$name"
    if [ -f "$d/summary.json" ] && grep -q '"closed": true' "$d/summary.json"; then
        echo "$name: done already"; return 0
    fi
    mkdir -p "$d"
    echo "$name: timers=$timers seed=$seed envs=$n iterations=$iters ($(date -Is))"
    (cd "$d" && env -u DISPLAY "$OV" -m wheeledlab_bench.launch --out "$d" --form stock \
        --timers "$timers" --warmup "$warmup" --seed "$seed" --gpu "$GPU" \
        --task-package p4_leadbrake "${gate[@]}" -- \
        "$ISAACLAB/scripts/reinforcement_learning/train.py" --rl_library rsl_rl \
        --task P4-LeadBrake-v0 --num_envs "$n" --max_iterations "$iters" --seed "$seed" \
        > "$d.log" 2>&1)
}

# evaluate one checkpoint: <name> <iteration>; prints the evaluator's JSON line
evaluate() {
    local name=$1 it=$2 ck
    ck=$(ls "$out/$stage/$name"/logs/rsl_rl/p4_leadbrake/*/model_"$it".pt 2>/dev/null | head -1)
    [ -n "$ck" ] || { echo "{\"iteration\": $it, \"missing\": true}"; return 0; }
    env -u DISPLAY "$OV" "$here/tools/evaluate.py" --checkpoint "$ck" \
        > "$out/$stage/$name.eval$it.log" 2>&1
    grep '^{' "$out/$stage/$name.eval$it.log" | sed "s/^{/{\"iteration\": $it, /"
}

case $stage in
smoke)
    run smoke sync 1 64 3 1
    ;;
seeds)
    for s in 1 2 3 4 5; do
        run "seed$s" sync "$s" 1024 1500 20
        evals="$out/$stage/seed$s.evals.jsonl"
        [ -s "$evals" ] && { echo "seed$s: evaluated already"; continue; }
        for it in $(seq 100 100 1400) 1499; do
            line=$(evaluate "seed$s" "$it"); echo "$line" >> "$evals"
            echo "seed$s it$it: $(echo "$line" | grep -oE '"success_rate": [0-9.]+')"
            if echo "$line" | grep -q '"pass": true' && [ "$it" != 1499 ]; then
                line=$(evaluate "seed$s" 1499); echo "$line" >> "$evals"; break
            fi
        done
    done
    ;;
sweep)
    for n in 256 1024 4096; do run "n$n" sync 1 "$n" 60 20; done
    run n1024-timers-off off 1 1024 60 20
    ;;
*) echo "unknown stage $stage" >&2; exit 2 ;;
esac
