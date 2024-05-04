#!/bin/bash

#PBS -l ncpus=12
#PBS -l mem=8GB
#PBS -l jobfs=32GB
#PBS -q gpuvolta
#PBS -P ik70
#PBS -l walltime=07:00:00
#PBS -l ngpus=1
#PBS -l storage=scratch/ik70
#PBS -l wd
#PBS -M dai.dai@csiro.au
#PBS -m e
#PBS -r y
#PBS -J 0-14


source "/scratch/ik70/miniconda3/etc/profile.d/conda.sh"

conda activate /scratch/ik70/conda_env/dainlp

root_dir=/scratch/ik70
dataset=wiesp
code_id=2218AG/2

cd ../code


SEEDs=(52 869 1001)
CHECKPOINTs=(600 1200 1800 2400 3000)
seed=${SEEDs[$PBS_ARRAY_INDEX/${#CHECKPOINTs[@]}]}
checkpoint=${CHECKPOINTs[$PBS_ARRAY_INDEX%${#CHECKPOINTs[@]}]}

lr=2e-5
span=14
context=0

running_id=${dataset}_checkpoint_${checkpoint}_seed_${seed}
output_dir=$PBS_JOBFS/TEMP/${code_id}/${running_id}_$(date +%F-%H-%M-%S-%N)
logging_dir=$root_dir/logging_dir/${code_id}/${running_id}
result_dir=${root_dir}/results/${code_id}
data_dir=$root_dir/ProcessedData/WIESP2022/2
train_filepath=${data_dir}/train.json
dev_filepath=${data_dir}/dev.json
test_filepath=${data_dir}/test.json
label_filepath=${data_dir}/ner2idx.json
train_metrics_filepath=${result_dir}/train/${running_id}.json
test_metrics_filepath=${result_dir}/test/${running_id}.json
pred_metrics_filepath=${result_dir}/pred/${running_id}.json
model_dir=$root_dir/saved_models/2218AB/1/wiesp2022-roberta-large/checkpoint-${checkpoint}

if ! test -f "${test_metrics_filepath}"; then
  python3 train.py \
  --task_name span-ner \
  --dataset_name $dataset \
  --output_metrics_filepath $train_metrics_filepath \
  --model_dir $model_dir \
  --seed $seed \
  --train_filepath $train_filepath \
  --dev_filepath $dev_filepath \
  --label_filepath $label_filepath \
  --output_dir $output_dir \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --logging_dir $logging_dir \
  --learning_rate $lr \
  --task_learning_rate 5e-4 \
  --num_train_epochs 30 \
  --metric_for_best_model micro_f1 \
  --greater_is_better \
  --context_window $context \
  --max_span_length $span

  python3 eval.py \
  --task_name span-ner \
  --dataset_name $dataset \
  --output_metrics_filepath $test_metrics_filepath \
  --model_dir $output_dir \
  --test_filepath $test_filepath \
  --output_dir $output_dir \
  --logging_dir $logging_dir \
  --output_predictions_filepath $pred_metrics_filepath \
  --context_window $context \
  --max_span_length $span

  python3 eval.py \
  --task_name span-ner \
  --dataset_name $dataset \
  --output_metrics_filepath ${result_dir}/test1/${running_id}.json \
  --model_dir $output_dir \
  --test_filepath ${data_dir}/test1.json \
  --output_dir $output_dir \
  --logging_dir $logging_dir \
  --output_predictions_filepath ${result_dir}/pred1/${running_id}.json \
  --context_window $context \
  --max_span_length $span

  python3 eval.py \
  --task_name span-ner \
  --dataset_name $dataset \
  --output_metrics_filepath ${result_dir}/test2/${running_id}.json \
  --model_dir $output_dir \
  --test_filepath ${data_dir}/test2.json \
  --output_dir $output_dir \
  --logging_dir $logging_dir \
  --output_predictions_filepath ${result_dir}/pred2/${running_id}.json \
  --context_window $context \
  --max_span_length $span

  rm -r $output_dir
fi
