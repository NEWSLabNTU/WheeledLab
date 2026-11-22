#!/usr/bin/env bash
# scensim roadmap 094 P1: WheeledLab's MuSHR drift task, ported (this branch), measured with
# wheeledlab_bench. One script for every host, so every host runs the same thing.
#
#   bench/p1/run_p1.sh <stage> <out-dir>
#
# stages (in order; each writes under <out-dir>/<stage>/):
#   check   the stack's versions, the GPU, its kernel+sync latency; refuses a shared GPU
#   smoke   3 iterations at 64 envs (cameras off), through the harness; then the eval on it
#   seeds   5 seeds (1-5) at 1024 envs, 5000 iterations, cameras off; the eval on each
#   sweep   throughput at N = 256, 1024, 4096 (60 iterations, 20 warm-up), cameras off and as
#           configured; timers sync, and off at 1024 (the harness's overhead)
#
# Host settings, from the environment:
#   OV        the Isaac Lab overlay's python (required)
#   GPU       the GPU index to run on (default 0); runs set CUDA_VISIBLE_DEVICES to it (Kit
#             warns that this "can lead to undesired behavior"; it ran clean on 241)
#   IDLE_OTHERS=1  accept another user's process on the GPU while it runs nothing (0 % over a
#             10 s watch, and the sync probe passes), e.g. a loaded model server; recorded
#   SHARED=1  run on a shared GPU anyway (a smoke: the probe is recorded, the numbers are not
#             a measurement)
set -euo pipefail
stage=${1:?usage: run_p1.sh check|smoke|seeds|sweep <out-dir>}
out=${2:?usage: run_p1.sh <stage> <out-dir>}
OV=${OV:?OV: the Isaac Lab overlay python, e.g. /var/tmp/aeon/isaaclab-3.0.0/venv/bin/python}
GPU=${GPU:-0}
here=$(cd "$(dirname "$0")/../.." && pwd)        # the fork's checkout
cd "$here"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}
# --gpu is nvidia-smi's index (the physical GPU); the run itself sees it as cuda:0
gate=(--require-idle); [ "${IDLE_OTHERS:-0}" = 1 ] && gate+=(--idle-others-ok)
[ "${SHARED:-0}" = 1 ] && gate=(--allow-shared)
mkdir -p "$out/$stage"
# training writes to source/wheeledlab_rl/logs/<run_name>; the out-dir's name is in it, so two
# attempts (two out-dirs) never share a training directory
tag=$(basename "$(cd "$out" && pwd)")
TRAIN=(source/wheeledlab_rl/scripts/train_rl.py -r RSS_DRIFT_CONFIG train.log.no_wandb=True
       train.log.video=False)

# one training run through the harness: <name> <cameras 0|1> <timers> <seed> <n_envs> <iters> <warmup>
run() {
    local name=$1 cam=$2 timers=$3 seed=$4 n=$5 iters=$6 warmup=$7
    local d="$out/$stage/$name"
    if [ -f "$d/summary.json" ] && grep -q '"closed": true' "$d/summary.json"; then
        echo "$name: done already"; return 0
    fi
    echo "$name: cameras=$cam timers=$timers seed=$seed envs=$n iterations=$iters ($(date -Is))"
    WHEELEDLAB_ENABLE_CAMERAS=$cam env -u DISPLAY "$OV" -m wheeledlab_bench.launch \
        --out "$d" --form "ported-cam$cam" --timers "$timers" --warmup "$warmup" --seed "$seed" \
        --gpu "$GPU" --task-package wheeledlab_tasks "${gate[@]}" -- "${TRAIN[@]}" agent.seed="$seed" \
        train.num_iterations="$iters" env_setup.num_envs="$n" \
        train.log.run_name="p1-$tag-$stage-$name" > "$d.log" 2>&1
}

evaluate() {
    local name=$1 rd
    rd=$(ls -d "source/wheeledlab_rl/logs/p1-$tag-$stage-$name" 2>/dev/null) || {
        echo "$name: no run directory source/wheeledlab_rl/logs/p1-$tag-$stage-$name"; return 1; }
    WHEELEDLAB_ENABLE_CAMERAS=0 env -u DISPLAY "$OV" bench/p1/eval_drift.py --run-path "$rd" \
        --out "$out/$stage/$name.eval.json" > "$out/$stage/$name.eval.log" 2>&1
    grep EVAL "$out/$stage/$name.eval.log" | cut -c1-400
}

case $stage in
check)
    "$OV" - <<'EOF' | tee "$out/check/check.txt"
import importlib.metadata as m, json, subprocess, torch
from wheeledlab_bench import gpu
v = {p: m.version(p) for p in ("isaacsim", "isaaclab", "rsl-rl-lib", "torch", "torchvision",
                               "warp-lang", "wheeledlab", "wheeledlab_rl", "wheeledlab-bench")}
print("releases", json.dumps(v))
print("torch", torch.__version__, "cuda", torch.version.cuda, "arch", torch.cuda.get_arch_list()[-1])
print("gpu", json.dumps(gpu.gpus()))
print("sync probe (this process's cuda:0)", json.dumps(gpu.sync_probe(0)))
print("fork", subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip())
EOF
    ;;
smoke)
    run smoke 0 sync 1 64 3 1
    evaluate smoke
    ;;
seeds)
    for s in 1 2 3 4 5; do
        run "seed$s" 0 sync "$s" 1024 5000 20
        evaluate "seed$s"
    done
    ;;
sweep)
    # alternate forms within each size, so a drift in the host's state does not fall on one form
    for n in 256 1024 4096; do
        for cam in 0 1; do run "n${n}-cam${cam}" "$cam" sync 1 "$n" 60 20; done
    done
    run n1024-cam0-timers-off 0 off 1 1024 60 20
    ;;
*) echo "unknown stage $stage" >&2; exit 2 ;;
esac
