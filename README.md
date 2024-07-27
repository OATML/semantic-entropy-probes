# Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs

Jannik Kossen*, Jiatong Han*, Muhammed Razzak*, Lisa Schut, Shreshth Malik, Yarin Gal

| **[Abstract](#Abstract)**
| **[Citation](#Citation)**
| **[Requirements](#Requirements)**
| **[Installation](#Installation)**
| **[Tutorial](#Tutorial)**
| **[Codebase](#Codebase)**

[![arXiv](https://img.shields.io/badge/arXiv-2406.15927-b31b1b.svg)](https://arxiv.org/abs/2406.15927)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/get-started/locally/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

## Abstract
We propose semantic entropy probes (SEPs), a cheap and reliable method for uncertainty quantification in Large Language Models (LLMs). Hallucinations, which are plausible-sounding but factually incorrect and arbitrary model generations, present a major challenge to the practical adoption of LLMs. Recent work by [Farquhar et al. (2024)](https://www.nature.com/articles/s41586-024-07421-0) proposes semantic entropy (SE), which can detect hallucinations by estimating uncertainty in the space semantic meaning for a set of model generations. However, the 5-to-10-fold increase in computation cost associated with SE computation hinders practical adoption. To address this, we propose SEPs, which directly approximate SE from the hidden states of a single generation. SEPs are simple to train and do not require sampling multiple model generations at test time, reducing the overhead of semantic uncertainty quantification to almost zero. We show that SEPs retain high performance for hallucination detection and generalize better to out-of-distribution data than previous probing methods that directly predict model accuracy. Our results across models and tasks suggest that model hidden states capture SE, and our ablation studies give further insights into the token positions and model layers for which this is the case.

## Citation
```
@misc{kossen2024semanticentropyprobesrobust,
      title={Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs}, 
      author={Jannik Kossen and Jiatong Han and Muhammed Razzak and Lisa Schut and Shreshth Malik and Yarin Gal},
      year={2024},
      eprint={2406.15927},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.15927}, 
}
```

## Requirements

### Hardware Dependencies

To obtain the hidden states of the large language model, you are required to do forward passes through the model on the relevant prompts/answers. Our code makes use of GPUs doing inference at FP16.

Common memory requirements per model size:
- 7B models: ~24 GB
- 13B models: ~48GB
- 70B model: ~160GB


### Software Dependencies

Dependecies for this code include Python 3.11 and PyTorch 2.1.

In `environment_export.yaml`, we list the precise versions for all Python packages.

## Installation


To install Python with all necessary dependencies, we recommend you use conda.

We refer to [https://conda.io/](https://conda.io/) for an installation guide.

After installing conda, you can set up and activate the conda environment by executing the following commands at the root folder of this repository:

```
conda-env update -f sep_enviroment.yaml
conda activate se_probes
```


Our experiments rely on [Weights & Biases](https://wandb.ai/) to log results. You may need to log in with your wandb API key upon initial execution.

Our experiments rely on HuggingFace for all LLM models and most of the datasets. Set the environment variable `HUGGING_FACE_HUB_TOKEN` to the token associated with your Hugging Face account. For Llama models, [apply for access](https://huggingface.co/meta-llama) to use the official repository of Meta's LLaMa-2 models.


Our experiments with sentence-length generation use GPT models from the OpenAI API.
Please set the environment variable `OPENAI_API_KEY` to your OpenAI API key in order to use these models.
Costs for reproducing our results vary depending on experiment configuration, but, without any guarantee, should lie somewhere between 10 and 100 USD.


For almost all tasks, the dataset is downloaded automatically from HuggingFace Datasets library upon first execution.
Only for bioasq, data needs to be [downloaded](http://participants-area.bioasq.org/datasets) manually.


## Tutorial

### Generate Semantic Entropy Probes Dataset

Execute

```
python generate_answers.py --model_name=Llama-2-7b-chat --dataset=trivia_qa
```

to reproduce results for short-phrase generation with LLaMa-2 Chat (7B) on the TriviaQA dataset.

The expected runtime of this demo is 1 hour using an A100 GPU, 24 cores of a Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz, and 192 GB of RAM.
Runtime may be longer upon first execution, as models need to be downloaded first.

Note down the wandb id assigned to your demo run.

To obtain a barplot similar to those of the paper, open the the iPython notebook in `semantic_entropy_probes/train-latent-probe.ipynb`, populate `wandb_id` with the id of your demo run, and execute all cells.

### Training Semantic Entropy Probes

We retrieve saved model hidden states on two token positions (TBG, SLT) with which we train linear probes to predict model semantic uncertainty and further predict correctness.

See [this notebook](./semantic_entropy_probes/latent-probe.ipynb) for step-by-step guide on training SEPs, which also contains handy tools for data loading, visualizations, and computing baselines. 

## Codebase
### Repository Structure

* Code to generate the semantic entropy is contained in the semantic uncertainty folder, and adapted from the repo for [semantic uncertainty](https://github.com/jlko/semantic_uncertainty). With in this, a standard SE generation run executes the following three scripts in order:

    1. `generate_answers.py`: Sample responses (and their likelihods/hidden states) from the models for the questions.
    2. `compute_uncertainties.py`: Compute uncertainty metrics given responses.
    3. `analyze_results.py`: Compute aggregate performance metrics.
* Once this is calculated, you can use the train_latent-probe.ipynb notebook, contained in the semantic_entropy_probes folder to train your SEPs.

### Reproducing the Experiments

To reproduce the experiments of the paper, one just needs to run the above demo for the various combinations of models and datasets.

The simplest way is to execute `slurm/run.sh` (with commands to generate both short-form and long-form answers to all datasets) if you are using `slurm`. 

Or you may directly execute iteratively

```
python generate_answers.py --model_name=$MODEL --dataset=$DATASET $EXTRA_CFG
```

where

* `$MODEL` is one of: [`Llama-2-7b, Llama-2-13b, Llama-2-70b, Llama-2-7b-chat, Llama-2-13b-chat, Llama-2-70b-chat, falcon-7b, falcon-40b, falcon-7b-instruct, falcon-40b-instruct, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.1, Phi-3-mini-128k-instruct`],
* `$DATASET` is one of [`trivia_qa, squad, med_qa, bioasq, record, nq, svamp`],
* and `$EXTRA_CFG` is empty for short-phrase generation and for sentence-length generation, `EXTRA_CFG=--num_few_shot=0 --model_max_new_tokens=100 --brief_prompt=chat --metric=llm_gpt-4 --entailment_model=gpt-3.5 --no-compute_accuracy_at_all_temps`.

The results for any run can be obtained by passing their `wandb_id` to an evaluation notebook identical to the demonstration in `semantic_entropy_probes/train-latent-probe.ipynb`.
