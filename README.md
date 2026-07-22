# A-MokA-CoT for Generative Multi-Label Audio Tagging

This repository provides the implementation of A-MokA-CoT, a generative
multi-label audio tagging framework that combines acoustic chain-of-thought
supervision with modality-aware low-rank adaptation.

The current implementation is based on Qwen2-Audio-7B-Instruct and uses
FSD50K as the primary evaluation benchmark.

## Overview

The repository contains code for:

- FSD50K A-CoT supervision construction;
- Controlled acoustic template management;
- A-MokA adapter injection;
- LoRA baseline training;
- A-MokA ablation experiments;
- Generative multi-label audio inference;
- Strict and semantic evaluation;
- Multi-label error analysis.

A-MokA supports separate adaptation branches for:

1. Audio and multimodal projector modules;
2. Language-model MLP modules;
3. Attention Q/V projection modules.

## Repository Structure

```text
.
├── config/
│   ├── dataset_fsd50k.yaml
│   ├── train_fsd50k.yaml
│   ├── infer_fsd50k.yaml
│   └── evaluate_fsd50k.yaml
│
├── dataset/
│   ├── dataset_fsd50k_acot.py
│   ├── fsd50k_feature_aliases.json
│   ├── fsd50k_features.json
│   └── timbre_pools.json
│
├── main/
│   ├── train_fsd50k.py
│   ├── infer_fsd50k.py
│   └── evaluate_fsd50k.py
│
├── requirements.txt
├── LICENSE
└── README.md
Requirements
The recommended environment is:

Python >= 3.10;
PyTorch with CUDA support;
Transformers;
Hugging Face Datasets;
ms-swift;
Sentence Transformers;
scikit-learn.
Install the required packages:

pip install -r requirements.txt
The exact versions of PyTorch, Transformers, and ms-swift should be
consistent between training and inference.

Installation
Clone the repository:

git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>
For example, if your GitHub username is zhangsan and the repository name is
Amoka, use:

git clone https://k779011051-svg/Amoka.git
cd Amoka
A conda environment can be created as follows:

conda create -n amoka python=3.10
conda activate amoka
pip install -r requirements.txt
Dataset
This project uses FSD50K. Please download the dataset from its official source
and comply with the corresponding dataset license and usage terms.

The expected directory structure is:

FSD50K/
├── FSD50K.dev_audio/
│   ├── <file_id>.wav
│   └── ...
│
└── FSD50K.metadata/
    ├── vocabulary.csv
    └── FSD50K.ground_truth/
        ├── dev.csv
        ├── eval.csv
        └── train.csv
The repository does not redistribute the original FSD50K audio files.

Configuration
Configuration files are stored in the config/ directory.

Before running the code, replace all placeholder paths such as:

/path/to/Qwen2-Audio-7B-Instruct
/path/to/FSD50K
/path/to/fsd50k_processed_train_cache_cot
with paths on your own machine.

Do not upload private model paths, access tokens, passwords, or API keys to
the repository.

A-CoT Dataset Construction
The acoustic resources are stored in:

dataset/
├── fsd50k_features.json
├── timbre_pools.json
└── fsd50k_feature_aliases.json
These files define:

Class-level acoustic descriptions;
Controlled timbre vocabularies;
Official-label-to-template mappings.
Edit the paths in:

config/dataset_fsd50k.yaml
Then run:

python dataset/dataset_fsd50k_acot.py \
    --config config/dataset_fsd50k.yaml
The generated files are written to the configured output directory:

data/processed/
├── fsd50k_train_acot.jsonl
└── fsd50k_val_acot.jsonl
The generated JSONL format is:

{
  "id": "sample_id",
  "query": "Identify all sound events and provide a detailed acoustic analysis for each template-covered source.",
  "response": "Reasoning Steps: ... Final Conclusion: ...",
  "audios": ["/path/to/audio.wav"]
}
The dataset construction script validates the official label space and checks
the expected template coverage before generating supervision data.

Training
Edit the general training configuration:

config/train_fsd50k.yaml
The same configuration file can be used for the main experiment, ablation
experiments, and the LoRA baseline.

A-MokA Main Experiment
adapter_type: amoka
supervision_type: cot
amoka_variant: full
Run:

python main/train_fsd50k.py \
    --config config/train_fsd50k.yaml
A-MokA Ablation Experiments
The following A-MokA variants are supported:

Variant	Description
full	Audio/projector, MLP, and broad Q/V branches
wo_audio	Removes the audio/projector branch
wo_mlp	Removes the language-model MLP branch
wo_attn	Removes the broad Q/V attention branch
all_r32	Uses rank 32 for all A-MokA branches
all_r16	Uses rank 16 for all A-MokA branches
To run an ablation, only change the following field:

amoka_variant: wo_audio
or:

amoka_variant: wo_mlp
amoka_variant: wo_attn
amoka_variant: all_r32
The corresponding training command remains unchanged:

python main/train_fsd50k.py \
    --config config/train_fsd50k.yaml
LoRA Baseline
To train the homogeneous LoRA baseline, change:

adapter_type: lora
The amoka_variant field is ignored when adapter_type is set to lora.

Supervision Types
Two supervision formats are supported:

supervision_type: cot
and:

supervision_type: direct
For A-CoT supervision, use the corresponding A-CoT cache:

training_cache: /path/to/fsd50k_processed_train_cache_cot
validation_cache: /path/to/fsd50k_processed_val_cache_cot
For direct supervision, use:

training_cache: /path/to/fsd50k_processed_train_cache_direct
validation_cache: /path/to/fsd50k_processed_val_cache_direct
For direct supervision, generation length is usually shorter:

max_new_tokens: 128
For A-CoT supervision, a larger generation length is usually required:

max_new_tokens: 600
Inference
Edit:

config/infer_fsd50k.yaml
The following settings must match the training configuration:

adapter_type: amoka
supervision_type: cot
amoka_variant: full
For example, if the model was trained using:

amoka_variant: wo_audio
the inference configuration must also use:

amoka_variant: wo_audio
Otherwise, the injected adapter structure may not match the checkpoint.

Run inference:

python main/infer_fsd50k.py \
    --config config/infer_fsd50k.yaml
The inference script supports:

JSONL input files;
Hugging Face datasets saved with save_to_disk.
For cached SFT data, the script identifies the response boundary from
labels != -100 and removes the target response from the generation input.
This is used to avoid ground-truth response leakage.

The output JSONL contains fields such as:

{
  "sample_index": 0,
  "run_name": "FSD50K-AMOKA-COT",
  "adapter_type": "amoka",
  "supervision_type": "cot",
  "amoka_variant": "full",
  "ground_truth_strict": ["Speech"],
  "prediction_strict": ["Speech"],
  "prediction_raw": ["Speech"],
  "sample_precision": 1.0,
  "sample_recall": 1.0,
  "sample_f1": 1.0,
  "full_response": "Reasoning Steps: ..."
}
A summary file is generated next to the inference result:

inference_results_summary.json
Evaluation
Edit:

config/evaluate_fsd50k.yaml
Run:

python main/evaluate_fsd50k.py \
    --config config/evaluate_fsd50k.yaml
The evaluation script reports:

Samples F1;
Macro precision;
Macro recall;
Macro F1;
Micro precision;
Micro recall;
Micro F1;
Strict exact-label results;
Sentence-BERT semantic projection results.
The strict exact-label evaluation should be used as the primary metric.

Semantic evaluation should be reported as an auxiliary analysis because the
semantic projection procedure depends on the selected embedding model and
similarity threshold.

Reproducibility Notes
For a valid reproduction, keep the following settings consistent:

The base model version;
The official FSD50K vocabulary;
The adapter type;
The A-MokA variant;
The supervision type;
The training and validation cache;
The maximum sequence length;
The generation configuration;
The random seed used for A-CoT construction.
In particular:

Training adapter type  = Inference adapter type
Training variant       = Inference variant
Training supervision   = Inference supervision
The following combinations must be matched:

adapter_type: amoka
supervision_type: cot
amoka_variant: full
or:

adapter_type: amoka
supervision_type: cot
amoka_variant: wo_audio
or:

adapter_type: lora
supervision_type: cot
Output and Temporary Files
The following directories are normally generated locally and should not be
committed to GitHub:

outputs/
checkpoints/
data/processed/
data/cache/
__pycache__/
Recommended .gitignore entries:

outputs/
checkpoints/
data/processed/
data/cache/
__pycache__/
*.pyc
*.pt
*.pth
*.bin
*.safetensors
*.jsonl
*.log

config/*.local.yaml
License
This project is distributed under the license specified in LICENSE.

The FSD50K dataset, Qwen2-Audio, Transformers, ms-swift, and other
third-party components are subject to their respective licenses.

Please check and comply with all third-party license requirements before
redistributing derived data, model weights, or modified source code.

Citation
If you find this repository useful, please cite the associated paper:

@article{<citation_key>,
  title   = {<Paper Title>},
  author  = {<Author Names>},
  journal = {Neurocomputing},
  year    = {2026}
}
The citation information will be updated after the paper is published.

Contact
For questions, bug reports, or reproduction issues, please open a GitHub
Issue or contact:

Email: <779011051@qq.com>
