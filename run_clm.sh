set -x
export BS=${1:-12}
export MEMCAP=${2:-0}
export MODEL=${3:-"2.7b"}
export GPUNUM=${4:-2}

# make directory for logs
mkdir -p ./logs

export MODLE_PATH="facebook/opt-${MODEL}"

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
torchrun \
  --nproc_per_node ${GPUNUM} \
  --master_port 19198 \
  run_clm.py \
  --preprocessing_num_workers 24 \
  --dataset_name data/dna_function_data.jsonl \
  --dataset_config_name wikitext-2-raw-v1 \
  --output_dir genomicGPT\
  --mem_cap ${MEMCAP} \
  --with_tracking \
  --model_name_or_path ${MODLE_PATH} \
  --per_device_train_batch_size ${BS} 2>&1 | tee ./logs/colo_${MODEL}_bs_${BS}_cap_${MEMCAP}_gpu_${GPUNUM}.log


