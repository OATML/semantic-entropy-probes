#!/bin/bash
# SBATCH --cpus-per-task=24
# SBATCH --partition=your_partition_name
# SBATCH --gres=gpu:a100:2
# SBATCH --job-name="nlg_uncertainty_linearprobe"

# Update conda environment (adjust as appropriate)
~/miniconda3/bin/conda-env update -f ../sep_environment.yaml
source ~/miniconda3/bin/activate se_probes

datasets=("squad" "nq" "trivia_qa" "bioasq")

# Short-form generation. Run the scripts with specified parameters.
for dataset in "${datasets[@]}"; do
    srun python ../semantic_uncertainty/generate_answers.py \
        --model_name=Llama-2-7b-chat \
        --dataset=$dataset \
        --num_samples=2000 \
        --random_seed=20 \
        --no-compute_p_ik \
        --no-compute_p_ik_answerable
    # e.g. Mistral-7B-Instruct-v0.1, Llama-2-7b-chat, Phi-3-mini-128k-instruct, Meta-Llama-3-8B-Instruct, etc.
done

# Long-form generation. Run the scripts with specified parameters.
for dataset in "${datasets[@]}"; do
    srun python ../semantic_uncertainty/generate_answers.py \
        --model_name=Llama-2-70b-chat \
        --dataset=$dataset \
        --num_samples=1000 \
        --random_seed=20 \
        --no-compute_p_ik \
        --no-compute_p_ik_answerable \
        --p_true_num_fewshot=10 \
        --num_generations=10 \
        --num_few_shot=0 \
        --model_max_new_tokens=100 \
        --brief_prompt=chat \
        --metric=llm_gpt-4 \
        --entailment_model=llm_gpt-3.5
    # e.g. Meta-Llama-3-70B-Instruct, Llama-2-70b-chat, etc.
done

