#!/bin/bash
set -e
# export CUDA_VISIBLE_DEVICES="4,5"
# export TRANSFORMERS_CACHE="/lfs/local/0/sttruong/env/.huggingface"
# export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
# export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"

MODEL_ID=$1
TGI=$2
# Description: Run toxicity analysis experiments
echo "#Language modeling Experiment\n"
echo "## Experiment 1 - 7b-ura-llama Seed 42"
echo "## mlqa_MLM - Original - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name mlqa_MLM \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## mlqa_MLM - Fairness - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name mlqa_MLM_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## mlqa_MLM - Original - Zeroshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name mlqa_MLM \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## VSEC - Original - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name VSEC \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## VSEC - Fairness - Fewshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name VSEC_fairness \
               --prompting_strategy 0 \
               --fewshot_prompting True  \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

echo "## VSEC - Original - Zeroshot"
python src/evaluate.py --model_name ${MODEL_ID} \
               --dataset_name VSEC \
               --prompting_strategy 0 \
               --fewshot_prompting False \
               --seed 42 \
               --continue_infer True \
               --tgi ${TGI}

# echo "## Experiment 2 - 7b-ura-llama Seed 123"
# echo "## mlqa_MLM - Original - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name mlqa_MLM \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123

# echo "## mlqa_MLM - Original - Zeroshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name mlqa_MLM \
#                --prompting_strategy 0 \
#                --fewshot_prompting False \
#                --seed 123

# echo "## VSEC - Original - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name VSEC \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 123


# echo "## VSEC - Original - Zeroshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name VSEC \
#                --prompting_strategy 0 \
#                --fewshot_prompting False \
#                --seed 123

# echo "## Experiment 3 - 7b-ura-llama Seed 456"
# echo "## mlqa_MLM - Original - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name mlqa_MLM \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456

# echo "## mlqa_MLM - Original - Zeroshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name mlqa_MLM \
#                --prompting_strategy 0 \
#                --fewshot_prompting False \
#                --seed 456

# echo "## VSEC - Original - Fewshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name VSEC \
#                --prompting_strategy 0 \
#                --fewshot_prompting True \
#                --seed 456


# echo "## VSEC - Original - Zeroshot"
# python src/evaluate.py --model_name ${MODEL_ID} \
#                --dataset_name VSEC \
#                --prompting_strategy 0 \
#                --fewshot_prompting False \
#                --seed 456
