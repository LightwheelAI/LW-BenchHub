#!/bin/bash

docker_run=off #on, off

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--docker_run)
            docker_run="$2"
            shift 2
            ;;
        -m|--model)
            model="$2"
            shift 2
            ;;
        -c|--config)
            config="$2"
            shift 2
            ;;
        -t|--task)
            task="$2"
            shift 2
            ;;
        -i|--instruction)
            instruction="$2"
            shift 2
            ;;
        -l|--layout)
            layout="$2"
            shift 2
            ;;
        *)
            echo "Unknown input: $1"
            exit 1
            ;;
    esac
done


run_python="python"
run_para=""
if [[ "$docker_run" == "on" ]]; then
    if [[ "$model" == gr00t* ]]; then
        echo "model: GR00T"
        source /opt/conda/etc/profile.d/conda.sh
        conda activate gr00t_man
        python -c "import torch; torch.cuda.set_per_process_memory_fraction(0.8)" 
    elif [[ "$model" == pi* ]]; then
        echo "model: pi"
        export PYTHON_MULTIPROCESSING_START_METHOD=spawn
        export PYTORCH_CUDA_FORK_SAFE=1
        run_python="/.venv/bin/python"
    elif [[ "$model" == go* ]]; then
        echo "model: go"
        run_python="/.venv/bin/python -m torch.distributed.run --nproc_per_node=1"
    else
        echo "Error: unknown model: '$model'"
        exit 1
    fi
fi

${run_python} lw_benchhub/scripts/policy/eval_policy.py \
    --config ${config} \
    --overrides \
    --env_cfg:task ${task} \
    --env_cfg:layout ${layout} \
    --instruction "${instruction}"
    # --test_num 1
    # --time_out_limit 60
    
