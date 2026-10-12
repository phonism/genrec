#!/bin/bash
# Orchestrator: wait for Beauty v2b/v2c (task 32, 30), pick the best
# (R@10 desc, tie-break R@1 desc, baseline v2=0.0862 always in the pool),
# then run sports + toys in parallel and home after sports.
set -e
cd /root/workspace/code/genrec
source venv/bin/activate

T_V2B=/tmp/thanos-tasks/task-32.exit       # v2b (no LS)
T_V2C=/tmp/thanos-tasks/task-33.exit       # v2c (no LS + lr5e-5)
WINNER_FILE=/tmp/genrec_winner_cfg.txt
SPORTS_DONE=/tmp/genrec_sports_done.flag

echo "=== [stage A] waiting for Beauty v2b (task 32) + v2c (task 33) ==="
while [ ! -f "$T_V2B" ] || [ ! -f "$T_V2C" ]; do sleep 60; done
echo "  v2b exit=$(cat $T_V2B), v2c exit=$(cat $T_V2C)"

echo ""
echo "=== [stage B] selecting winning config ==="
WINNER=$(python <<'PY'
from scripts.extract_sft_metrics import parse_log
import glob, sys
candidates = [
    ('v2',  'out/onerec/amazon/beauty/sft_k256_v2_1.7b_seq50/train_*.log',
            'config/onerec/amazon/sft_k256_v2.gin'),
    ('v2b', 'out/onerec/amazon/beauty/sft_k256_v2b_no_ls_1.7b_seq50/train_*.log',
            'config/onerec/amazon/sft_k256_v2b_no_ls.gin'),
    ('v2c', 'out/onerec/amazon/beauty/sft_k256_v2c_no_ls_lr5e5_1.7b_seq50/train_*.log',
            'config/onerec/amazon/sft_k256_v2c_no_ls_lr5e5.gin'),
]
results = []
for tag, pat, cfg in candidates:
    best = None
    for path in sorted(glob.glob(pat)):
        r = parse_log(path)
        if r['best_valid_epoch'] is not None:
            if best is None or r['best_valid_recall@10'] > best['best_valid_recall@10']:
                best = r
    if best:
        m = best['best_test_metrics']
        r10 = m.get('Recall@10', 0.0)
        r1  = m.get('Recall@1', 0.0)
        results.append((tag, cfg, r10, r1))
        print(f'{tag}: R@10={r10:.4f}, R@1={r1:.4f}', file=sys.stderr)
if not results:
    print('NO_RESULTS_FOUND', file=sys.stderr); sys.exit(2)
results.sort(key=lambda x: (x[2], x[3]), reverse=True)
print(f'WINNER: {results[0][0]} -> {results[0][1]}', file=sys.stderr)
print(results[0][1])
PY
)
echo "$WINNER" > "$WINNER_FILE"
echo "  winner config: $WINNER (written to $WINNER_FILE)"

echo ""
echo "=== [stage C] launching sports (GPU 3-6) + toys (GPU 0,1,2,7) in parallel ==="
( SFT_CFG="$WINNER" bash scripts/run_v2_sports.sh 2>&1 | tee out/orch_sports.log; \
  echo $? > /tmp/genrec_sports_exit; touch "$SPORTS_DONE" ) &
SPORTS_PID=$!
( SFT_CFG="$WINNER" MAIN_PORT=29502 bash scripts/run_v2_toys.sh 2>&1 | tee out/orch_toys.log; \
  echo $? > /tmp/genrec_toys_exit ) &
TOYS_PID=$!

echo "=== [stage D] launching home (will idle until sports finishes) ==="
( SFT_CFG="$WINNER" SPORTS_DONE="$SPORTS_DONE" bash scripts/run_v2_home.sh 2>&1 | tee out/orch_home.log; \
  echo $? > /tmp/genrec_home_exit ) &
HOME_PID=$!

wait $SPORTS_PID $TOYS_PID $HOME_PID
echo ""
echo "=== ✅ orchestrator finished ==="
echo "sports exit: $(cat /tmp/genrec_sports_exit 2>/dev/null || echo ?)"
echo "toys exit:   $(cat /tmp/genrec_toys_exit 2>/dev/null || echo ?)"
echo "home exit:   $(cat /tmp/genrec_home_exit 2>/dev/null || echo ?)"
