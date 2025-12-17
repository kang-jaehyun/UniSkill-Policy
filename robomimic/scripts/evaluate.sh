#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (assuming script is in robomimic/scripts)
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Default values
CONFIG="$PROJECT_ROOT/configs/uniskill_policy.json"
CKPT=""
TASK=""
ROLLOUT_NUM=10
HORIZON=500
SEED=0

# Help function
usage() {
    echo "Usage: $0 --ckpt <checkpoint_path> --task <task_name> [options]"
    echo "Options:"
    echo "  --config <config_path>    Path to config file (default: $CONFIG)"
    echo "  --ckpt <checkpoint_path>  Path to model checkpoint (required)"
    echo "  --task <task_name>        Name of the task to evaluate (required)"
    echo "  --rollout_num <num>       Number of rollouts (default: $ROLLOUT_NUM)"
    echo "  --horizon <num>           Horizon length (default: $HORIZON)"
    echo "  --seed <num>              Random seed (default: $SEED)"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift ;;
        --ckpt) CKPT="$2"; shift ;;
        --task) TASK="$2"; shift ;;
        --rollout_num) ROLLOUT_NUM="$2"; shift ;;
        --horizon) HORIZON="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check required arguments
if [ -z "$CKPT" ] || [ -z "$TASK" ]; then
    echo "Error: --ckpt and --task are required."
    usage
fi

# Activate environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Run evaluation
python "$SCRIPT_DIR/evaluate.py" \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --task "$TASK" \
    --rollout_num "$ROLLOUT_NUM" \
    --horizon "$HORIZON" \
    --seed "$SEED"
