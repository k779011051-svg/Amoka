# -*- coding: utf-8 -*-
"""
Run generative multi-label inference on FSD50K.

Supported adapters
------------------
- A-MokA with full and ablation variants
- Homogeneous LoRA baseline

Supported evaluation inputs
---------------------------
1. A JSONL file containing ``query``, ``response``, and ``audios``.
2. A Hugging Face dataset cache created with ``save_to_disk``.

For cached SFT data, the script identifies the response boundary from
``labels != -100`` and retains only the prompt tokens during generation.
This prevents the ground-truth response from leaking into model inputs.

Example
-------
python main/infer_fsd50k.py \
    --model-path /path/to/Qwen2-Audio-7B-Instruct \
    --checkpoint-path outputs/FSD50K-AMOKA-COT/checkpoint-1000 \
    --test-data-path data/cache/fsd50k_val_cot \
    --vocabulary-path /path/to/FSD50K.metadata/vocabulary.csv \
    --output-path outputs/inference_results.jsonl \
    --adapter-type amoka \
    --supervision-type cot \
    --amoka-variant full
"""

import argparse
import csv
import gc
import glob
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import yaml
import numpy as np
import torch
import torch.nn as nn
import tqdm
import transformers
from datasets import concatenate_datasets, load_from_disk
from safetensors.torch import load_file
from swift.llm import ModelType, get_model_tokenizer, get_template

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# ----------------------------------------------------------------------
# Official FSD50K vocabulary
# ----------------------------------------------------------------------
def load_official_vocabulary(
    vocabulary_path: Path,
    label_column: int = 1,
) -> List[str]:
    """Load the official 200-category FSD50K vocabulary."""

    if not vocabulary_path.exists():
        raise FileNotFoundError(
            f"Official FSD50K vocabulary not found: {vocabulary_path}"
        )

    labels = []

    with vocabulary_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.reader(file)

        for row_index, row in enumerate(reader):
            if not row or label_column >= len(row):
                continue

            candidate = row[label_column].strip().strip("\"'")

            if not candidate:
                continue

            if row_index == 0 and candidate.lower() in {
                "label",
                "labels",
                "display_name",
                "display name",
                "category",
                "name",
            }:
                continue

            labels.append(candidate)

    labels = list(dict.fromkeys(labels))

    if len(labels) != 200:
        raise ValueError(
            f"Expected 200 official FSD50K labels, but found {len(labels)}. "
            "Check --vocabulary-path and --vocab-label-column."
        )

    return labels

# ----------------------------------------------------------------------
# A-MokA adapter
# ----------------------------------------------------------------------
class MokALinear(nn.Module):
    """A modality-aware low-rank adaptation layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 32,
        alpha: int = 64,
        module_name: str = "",
    ):
        super().__init__()

        self.base_layer = base_layer
        self.r = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        input_features = base_layer.in_features
        output_features = base_layer.out_features

        self.layer_type = "attention"

        if (
            "audio_tower" in module_name
            or "multi_modal_projector" in module_name
        ):
            self.layer_type = "audio"
        elif any(
            key in module_name
            for key in ("gate_proj", "up_proj", "down_proj")
        ):
            self.layer_type = "mlp"

        if self.layer_type == "audio":
            self.A_audio = nn.Linear(
                input_features,
                rank,
                bias=False,
            )
        elif self.layer_type == "mlp":
            self.A_mlp = nn.Linear(
                input_features,
                rank,
                bias=False,
            )
            self.memory_gate = nn.Parameter(
                torch.full((1, rank), 0.5)
            )
        else:
            self.A_attn = nn.Linear(
                input_features,
                rank,
                bias=False,
            )
            self.attn_interaction = nn.Parameter(
                torch.zeros(1, rank)
            )

        self.B = nn.Linear(
            rank,
            output_features,
            bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base layer and the A-MokA offset."""

        base_output = self.base_layer(inputs)
        adapter_dtype = self.B.weight.dtype
        adapter_inputs = inputs.to(adapter_dtype)

        if self.layer_type == "audio":
            adapter_offset = self.B(
                self.A_audio(adapter_inputs)
            )
        elif self.layer_type == "mlp":
            gate = self.memory_gate.to(
                device=inputs.device,
                dtype=adapter_dtype,
            )
            adapter_offset = self.B(
                self.A_mlp(adapter_inputs) * (1.0 + gate)
            )
        else:
            interaction = self.attn_interaction.to(
                device=inputs.device,
                dtype=adapter_dtype,
            )
            adapter_offset = self.B(
                self.A_attn(adapter_inputs)
                * (1.0 + interaction)
            )

        return base_output + (
            adapter_offset * self.scaling
        ).to(base_output.dtype)
def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate a YAML configuration file."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(
            "The top level of the YAML configuration must be a mapping."
        )

    return config
def get_amoka_branch_configs(
    variant: str,
) -> Dict[str, Dict[str, int]]:
    """Return branch-specific ranks and scaling factors."""

    if variant == "all_r32":
        return {
            "Audio": {"rank": 32, "alpha": 64},
            "MLP": {"rank": 32, "alpha": 64},
            "Attention": {"rank": 32, "alpha": 64},
        }

    if variant == "all_r16":
        return {
            "Audio": {"rank": 16, "alpha": 64},
            "MLP": {"rank": 16, "alpha": 64},
            "Attention": {"rank": 16, "alpha": 64},
        }

    return {
        "Audio": {"rank": 32, "alpha": 64},
        "MLP": {"rank": 32, "alpha": 64},
        "Attention": {"rank": 16, "alpha": 64},
    }

def inject_amoka(
    model: nn.Module,
    variant: str,
) -> nn.Module:
    """
    Inject A-MokA adapters using the same routing as training.

    Audio/projector modules are checked before generic Q/V projections.
    This prevents internal audio-attention projections from being assigned
    incorrectly to the language attention branch.
    """

    supported_variants = {
        "full",
        "wo_audio",
        "wo_mlp",
        "wo_attn",
        "all_r32",
        "all_r16",
    }

    if variant not in supported_variants:
        raise ValueError(
            f"Unsupported A-MokA variant: {variant}"
        )

    audio_keys = (
        "audio_tower.layers",
        "multi_modal_projector",
    )
    mlp_keys = (
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    attention_keys = (
        "self_attn.q_proj",
        "self_attn.v_proj",
        "q_proj",
        "v_proj",
    )

    use_audio = variant != "wo_audio"
    use_mlp = variant != "wo_mlp"
    use_attention = variant != "wo_attn"

    configs = get_amoka_branch_configs(variant)

    target_keys = []

    if use_audio:
        target_keys.extend(audio_keys)
    if use_mlp:
        target_keys.extend(mlp_keys)
    if use_attention:
        target_keys.extend(attention_keys)

    visited_modules = set()
    injected_counts = {
        "Audio": 0,
        "MLP": 0,
        "Attention": 0,
    }

    print(
        f"Injecting A-MokA adapters for inference: "
        f"variant={variant}"
    )

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if not any(key in module_name for key in target_keys):
            continue

        if module in visited_modules:
            continue

        is_audio = any(
            key in module_name for key in audio_keys
        )
        is_mlp = any(
            key in module_name for key in mlp_keys
        )
        is_attention = any(
            key in module_name for key in attention_keys
        )

        if is_audio:
            if not use_audio:
                continue
            branch = "Audio"
        elif is_mlp:
            if not use_mlp:
                continue
            branch = "MLP"
        elif is_attention:
            if not use_attention:
                continue
            branch = "Attention"
        else:
            continue

        parent_name = ".".join(
            module_name.split(".")[:-1]
        )
        child_name = module_name.split(".")[-1]
        parent_module = model.get_submodule(parent_name)

        config = configs[branch]

        replacement = MokALinear(
            base_layer=module,
            rank=config["rank"],
            alpha=config["alpha"],
            module_name=module_name,
        )
        replacement.to(
            device=module.weight.device,
            dtype=module.weight.dtype,
        )

        setattr(parent_module, child_name, replacement)

        visited_modules.add(module)
        injected_counts[branch] += 1

    print("A-MokA injection completed.")
    print(f"Variant                 : {variant}")
    print(f"Use audio/projector     : {use_audio}")
    print(f"Use MLP                 : {use_mlp}")
    print(f"Use broad Q/V attention : {use_attention}")
    print(
        f"Audio/projector layers  : "
        f"{injected_counts['Audio']}"
    )
    print(f"MLP layers              : {injected_counts['MLP']}")
    print(
        f"Attention Q/V layers    : "
        f"{injected_counts['Attention']}"
    )

    if use_audio and injected_counts["Audio"] == 0:
        raise RuntimeError(
            "No audio/projector layers were injected."
        )

    if use_mlp and injected_counts["MLP"] == 0:
        raise RuntimeError("No MLP layers were injected.")

    if use_attention and injected_counts["Attention"] == 0:
        raise RuntimeError(
            "No attention Q/V layers were injected."
        )

    return model

# ----------------------------------------------------------------------
# Homogeneous LoRA baseline
# ----------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Standard homogeneous low-rank adaptation layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 32,
        alpha: int = 64,
    ):
        super().__init__()

        self.base_layer = base_layer
        self.r = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(
            base_layer.in_features,
            rank,
            bias=False,
        )
        self.lora_B = nn.Linear(
            rank,
            base_layer.out_features,
            bias=False,
        )

        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=np.sqrt(5),
        )
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base layer and the LoRA offset."""

        base_output = self.base_layer(inputs)
        adapter_inputs = inputs.to(
            self.lora_B.weight.dtype
        )

        adapter_offset = self.lora_B(
            self.lora_A(adapter_inputs)
        )

        return base_output + (
            adapter_offset * self.scaling
        ).to(base_output.dtype)

def inject_lora(model: nn.Module) -> nn.Module:
    """Inject the homogeneous LoRA baseline used during training."""

    target_keys = (
        "audio_tower.layers",
        "multi_modal_projector",
        "gate_proj",
        "up_proj",
        "down_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    )

    visited_modules = set()
    injected_count = 0

    print("Injecting homogeneous LoRA adapters for inference.")

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if not any(key in module_name for key in target_keys):
            continue

        if module in visited_modules:
            continue

        if any(
            key in module_name
            for key in (
                "self_attn.q_proj",
                "self_attn.v_proj",
            )
        ):
            if "language_model" not in module_name:
                continue

        parent_name = ".".join(
            module_name.split(".")[:-1]
        )
        child_name = module_name.split(".")[-1]
        parent_module = model.get_submodule(parent_name)

        replacement = LoRALinear(
            base_layer=module,
            rank=32,
            alpha=64,
        )
        replacement.to(
            device=module.weight.device,
            dtype=module.weight.dtype,
        )

        setattr(parent_module, child_name, replacement)

        visited_modules.add(module)
        injected_count += 1

    if injected_count == 0:
        raise RuntimeError("No LoRA layers were injected.")

    print("LoRA injection completed.")
    print(f"Injected LoRA layers: {injected_count}")

    return model

# ----------------------------------------------------------------------
# Checkpoint discovery and loading
# ----------------------------------------------------------------------
def checkpoint_step(path: Path) -> int:
    """Extract the training step from a checkpoint directory name."""

    match = re.search(
        r"checkpoint-(\d+)$",
        path.name,
    )
    return int(match.group(1)) if match else -1

def find_weight_files(
    checkpoint_path: Path,
) -> List[Path]:
    """Find model weight files within one checkpoint directory."""

    safetensor_files = sorted(
        checkpoint_path.glob("*.safetensors")
    )

    if safetensor_files:
        return safetensor_files

    preferred_binary_files = sorted(
        checkpoint_path.glob("pytorch_model*.bin")
    )

    if preferred_binary_files:
        return preferred_binary_files

    return sorted(checkpoint_path.glob("*.bin"))

def find_latest_checkpoint(
    run_name: str,
    output_root: Path,
) -> Path:
    """Find the most recent valid checkpoint for one run."""

    search_roots = [
        output_root / run_name,
        output_root / run_name / run_name,
        output_root,
    ]

    candidates = []

    for root in search_roots:
        if root.exists():
            candidates.extend(
                path
                for path in root.glob("checkpoint-*")
                if path.is_dir()
            )

    valid_checkpoints = [
        path
        for path in candidates
        if find_weight_files(path)
    ]

    if not valid_checkpoints:
        searched = "\n".join(
            f"  {root}" for root in search_roots
        )
        raise FileNotFoundError(
            "No valid checkpoint was found. "
            "Specify --checkpoint-path explicitly.\n"
            f"Searched under:\n{searched}"
        )

    valid_checkpoints.sort(key=checkpoint_step)

    return valid_checkpoints[-1]

def get_trainable_markers(
    adapter_type: str,
) -> Tuple[str, ...]:
    """Return parameter-name markers for one adapter type."""

    if adapter_type == "amoka":
        return (
            "A_audio",
            "A_mlp",
            "A_attn",
            "B",
            "memory_gate",
            "attn_interaction",
        )

    return (
        "lora_A",
        "lora_B",
    )

def load_binary_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """Load a PyTorch checkpoint while preferring weights-only mode."""

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )

def load_checkpoint_with_report(
    model: nn.Module,
    checkpoint_path: Path,
    adapter_type: str,
    amoka_variant: str,
) -> nn.Module:
    """Load checkpoint tensors and report adapter compatibility."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {checkpoint_path}"
        )

    weight_files = find_weight_files(checkpoint_path)

    if not weight_files:
        raise FileNotFoundError(
            f"No model weight files found in: {checkpoint_path}"
        )

    print("\nLoading checkpoint files:")
    for weight_file in weight_files:
        print(f"  {weight_file}")

    checkpoint_state = {}

    for weight_file in weight_files:
        if weight_file.suffix == ".safetensors":
            state_dict = load_file(str(weight_file))
        else:
            state_dict = load_binary_state_dict(weight_file)

        checkpoint_state.update(state_dict)

    markers = get_trainable_markers(adapter_type)
    model_parameter_names = {
        name for name, _ in model.named_parameters()
    }

    loaded_keys = []
    loaded_adapter_keys = []
    missing_adapter_keys = []
    unexpected_keys = []
    shape_mismatches = []

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in checkpoint_state:
                if any(marker in name for marker in markers):
                    missing_adapter_keys.append(name)
                continue

            source = checkpoint_state[name]

            if tuple(source.shape) != tuple(parameter.shape):
                shape_mismatches.append(
                    (
                        name,
                        tuple(source.shape),
                        tuple(parameter.shape),
                    )
                )
                continue

            parameter.copy_(
                source.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
            loaded_keys.append(name)

            if any(marker in name for marker in markers):
                loaded_adapter_keys.append(name)

    for key in checkpoint_state:
        if key not in model_parameter_names:
            unexpected_keys.append(key)

    print("\nCheckpoint loading report")
    print("-" * 64)
    print(f"Adapter type             : {adapter_type}")
    print(f"A-MokA variant           : {amoka_variant}")
    print(f"Checkpoint               : {checkpoint_path}")
    print(f"Checkpoint tensors       : {len(checkpoint_state)}")
    print(f"Loaded tensors           : {len(loaded_keys)}")
    print(f"Loaded adapter tensors   : {len(loaded_adapter_keys)}")
    print(f"Missing adapter tensors  : {len(missing_adapter_keys)}")
    print(f"Unexpected tensors       : {len(unexpected_keys)}")
    print(f"Shape mismatches         : {len(shape_mismatches)}")

    if missing_adapter_keys:
        print("\nMissing adapter tensors:")
        for key in missing_adapter_keys[:40]:
            print(f"  {key}")

    if unexpected_keys:
        print("\nUnexpected checkpoint tensors:")
        for key in unexpected_keys[:20]:
            print(f"  {key}")

    if shape_mismatches:
        print("\nShape mismatches:")
        for name, checkpoint_shape, model_shape in shape_mismatches[:20]:
            print(
                f"  {name}: checkpoint={checkpoint_shape}, "
                f"model={model_shape}"
            )

    print("-" * 64)

    if not loaded_adapter_keys:
        raise RuntimeError(
            "No adapter parameters were loaded. Check the adapter type, "
            "A-MokA variant, checkpoint path, and injection configuration."
        )

    if shape_mismatches:
        raise RuntimeError(
            "Checkpoint loading produced shape mismatches. "
            "The inference adapter configuration does not match training."
        )

    return model

# ----------------------------------------------------------------------
# Label extraction and sample-level metrics
# ----------------------------------------------------------------------
def unique_keep_order(items: Sequence[str]) -> List[str]:
    """Remove duplicates while preserving input order."""

    seen = set()
    output = []

    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)

    return output

def extract_labels(
    text: str,
    label_space: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """
    Extract strict and open-form labels from a generated response.

    For A-CoT responses, parsing starts after the last
    ``Final Conclusion`` marker. Direct responses are parsed in full.
    """

    if not text:
        return [], []

    label_mapping = {
        label.lower(): label
        for label in label_space
    }
    lower_text = text.lower()

    if "final conclusion" in lower_text:
        marker_index = lower_text.rfind(
            "final conclusion"
        )
        parse_text = lower_text[marker_index:]
    else:
        parse_text = lower_text

    if "categories are" in parse_text:
        labels_part = parse_text.split(
            "categories are",
            1,
        )[-1]
    elif "category is" in parse_text:
        labels_part = parse_text.split(
            "category is",
            1,
        )[-1]
    else:
        return [], []

    labels_part = labels_part.replace(
        "(additional sounds detected:",
        ",",
    )
    labels_part = labels_part.split("\n", 1)[0]
    labels_part = re.sub(
        r"\.\s*$",
        "",
        labels_part.strip(),
    )

    raw_items = [
        item.strip()
        for item in labels_part.split(",")
        if item.strip()
    ]

    strict_labels = []
    raw_labels = []

    for item in raw_items:
        candidate = item.strip(
            " .。,:;，；\"'"
        )
        candidate = candidate.replace(" ", "_")

        if (
            candidate.endswith(")")
            and candidate.lower() not in label_mapping
        ):
            without_closing_parenthesis = candidate[:-1]

            if (
                without_closing_parenthesis.lower()
                in label_mapping
            ):
                candidate = without_closing_parenthesis

        raw_labels.append(candidate)

        canonical = label_mapping.get(
            candidate.lower()
        )

        if canonical is not None:
            strict_labels.append(canonical)

    return (
        unique_keep_order(strict_labels),
        unique_keep_order(raw_labels),
    )

def compute_sample_metrics(
    target_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> Tuple[float, float, float, int, int, int]:
    """Compute sample-level precision, recall, and F1."""

    target_set = set(target_labels)
    prediction_set = set(predicted_labels)

    true_positive = len(target_set & prediction_set)
    false_positive = len(prediction_set - target_set)
    false_negative = len(target_set - prediction_set)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return (
        precision,
        recall,
        f1_score,
        true_positive,
        false_positive,
        false_negative,
    )

# ----------------------------------------------------------------------
# Evaluation data loading
# ----------------------------------------------------------------------
def load_jsonl_dataset(
    path: Path,
    max_samples: Optional[int],
) -> List[Dict[str, Any]]:
    """Load JSONL evaluation data and validate referenced audio files."""

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exception:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exception}"
                ) from exception

            required_fields = {
                "query",
                "response",
                "audios",
            }
            missing_fields = required_fields - set(record)

            if missing_fields:
                raise KeyError(
                    f"Line {line_number} is missing fields: "
                    f"{sorted(missing_fields)}"
                )

            audios = record["audios"]

            if isinstance(audios, str):
                audios = [audios]

            if not isinstance(audios, list) or not audios:
                raise ValueError(
                    f"Line {line_number} has an invalid audios field."
                )

            for audio_path in audios:
                if not Path(audio_path).exists():
                    raise FileNotFoundError(
                        f"Audio file referenced at line {line_number} "
                        f"does not exist: {audio_path}"
                    )

            records.append(
                {
                    "query": record["query"],
                    "response": record["response"],
                    "audios": audios,
                    "id": record.get("id"),
                }
            )

            if (
                max_samples is not None
                and len(records) >= max_samples
            ):
                break

    return records

def load_cached_dataset(path: Path):
    """Load a regular or sharded Hugging Face dataset cache."""

    manifest_path = path / "manifest.json"

    if not manifest_path.exists():
        return load_from_disk(str(path))

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    if "shards" not in manifest:
        raise KeyError(
            f"Cache manifest does not contain 'shards': {manifest_path}"
        )

    datasets = []

    for shard in manifest["shards"]:
        shard_index = shard["shard_idx"]
        shard_directory = path / f"shard_{shard_index:05d}"

        if not shard_directory.exists():
            raise FileNotFoundError(
                f"Cache shard not found: {shard_directory}"
            )

        print(f"Loading evaluation cache shard: {shard_directory}")
        datasets.append(
            load_from_disk(str(shard_directory))
        )

    if not datasets:
        raise ValueError(
            f"No cache shards were loaded from: {path}"
        )

    return concatenate_datasets(datasets)

def load_evaluation_data(
    data_path: Path,
    max_samples: Optional[int],
) -> Dict[str, Any]:
    """Load evaluation data from JSONL or a dataset cache."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Evaluation data path does not exist: {data_path}"
        )

    if data_path.is_dir():
        dataset = load_cached_dataset(data_path)

        required_columns = {
            "input_ids",
            "labels",
            "input_features",
            "feature_attention_mask",
        }
        missing_columns = (
            required_columns - set(dataset.column_names)
        )

        if missing_columns:
            raise ValueError(
                f"Evaluation cache is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if max_samples is not None:
            dataset = dataset.select(
                range(min(max_samples, len(dataset)))
            )

        print(f"Loaded Hugging Face cache: {data_path}")
        print(f"Samples: {len(dataset)}")
        print(f"Columns: {dataset.column_names}")

        return {
            "type": "cache",
            "data": dataset,
        }

    records = load_jsonl_dataset(
        path=data_path,
        max_samples=max_samples,
    )

    print(f"Loaded JSONL evaluation data: {data_path}")
    print(f"Samples: {len(records)}")

    return {
        "type": "jsonl",
        "data": records,
    }

# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------
def decode_cached_target(
    labels: Sequence[int],
    tokenizer,
) -> str:
    """Decode the target response stored in cached SFT labels."""

    labels_array = np.asarray(labels)
    labels_array = np.where(
        labels_array == -100,
        tokenizer.pad_token_id,
        labels_array,
    )

    return tokenizer.decode(
        labels_array.tolist(),
        skip_special_tokens=True,
    )

def find_response_start(
    labels: Sequence[int],
) -> Optional[int]:
    """Find the first token position supervised by the SFT objective."""

    labels_array = np.asarray(labels)
    supervised_positions = np.where(
        labels_array != -100
    )[0]

    if len(supervised_positions) == 0:
        return None

    return int(supervised_positions[0])

def maybe_to_tensor(value: Any) -> Any:
    """Convert array-like cached values to PyTorch tensors."""

    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)

    if isinstance(value, list):
        return torch.tensor(value)

    return value

def build_generation_item_from_cache(
    item: Dict[str, Any],
    prefix_ids: Sequence[int],
) -> Dict[str, Any]:
    """Construct a prompt-only generation item from an SFT cache."""

    if "labels" not in item:
        raise KeyError(
            "Cached item does not contain labels."
        )

    response_start = find_response_start(
        item["labels"]
    )

    if response_start is None:
        raise ValueError(
            "Cannot identify the response boundary because "
            "all cached labels are -100."
        )

    generation_item = {}

    for key, value in item.items():
        if key == "labels":
            continue

        if key == "input_ids":
            prompt_ids = list(value[:response_start])
            prompt_ids.extend(prefix_ids)
            generation_item[key] = torch.tensor(
                prompt_ids,
                dtype=torch.long,
            )

        elif key == "attention_mask":
            prompt_mask = list(value[:response_start])
            prompt_mask.extend([1] * len(prefix_ids))
            generation_item[key] = torch.tensor(
                prompt_mask,
                dtype=torch.long,
            )

        elif key == "position_ids":
            prompt_positions = list(
                value[:response_start]
            )

            start_position = (
                prompt_positions[-1] + 1
                if prompt_positions
                else 0
            )
            prompt_positions.extend(
                range(
                    start_position,
                    start_position + len(prefix_ids),
                )
            )
            generation_item[key] = torch.tensor(
                prompt_positions,
                dtype=torch.long,
            )

        elif key == "audio_info":
            generation_item[key] = value

        else:
            generation_item[key] = maybe_to_tensor(value)

    return generation_item

def encode_generation_item_from_jsonl(
    template,
    item: Dict[str, Any],
    prefix_ids: Sequence[int],
) -> Dict[str, Any]:
    """Encode a JSONL sample without including its target response."""

    encoded = template.encode(
        {
            "query": item["query"],
            "response": "",
            "audios": item["audios"],
        }
    )

    if isinstance(encoded, tuple):
        encoded = encoded[0]

    if not isinstance(encoded, dict):
        raise TypeError(
            "template.encode() must return a dictionary, "
            f"but returned {type(encoded)}."
        )

    encoded.pop("labels", None)

    if not encoded.get("input_ids"):
        raise KeyError(
            "Encoded generation item does not contain input_ids."
        )

    encoded["input_ids"] = list(
        encoded["input_ids"]
    )
    encoded["input_ids"].extend(prefix_ids)

    if encoded.get("attention_mask") is not None:
        encoded["attention_mask"] = list(
            encoded["attention_mask"]
        )
        encoded["attention_mask"].extend(
            [1] * len(prefix_ids)
        )

    if encoded.get("position_ids") is not None:
        positions = list(encoded["position_ids"])
        start_position = (
            positions[-1] + 1
            if positions
            else 0
        )
        positions.extend(
            range(
                start_position,
                start_position + len(prefix_ids),
            )
        )
        encoded["position_ids"] = positions

    for key in (
        "input_ids",
        "attention_mask",
        "position_ids",
    ):
        if encoded.get(key) is not None:
            encoded[key] = torch.tensor(
                encoded[key],
                dtype=torch.long,
            )

    return encoded

# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
def build_run_name(args: argparse.Namespace) -> str:
    """Construct a reproducible experiment identifier."""

    run_name = (
        f"FSD50K-{args.adapter_type.upper()}-"
        f"{args.supervision_type.upper()}"
    )

    if (
        args.adapter_type == "amoka"
        and args.amoka_variant != "full"
    ):
        run_name += f"-{args.amoka_variant.upper()}"

    return run_name

def resolve_input_device(model: nn.Module) -> torch.device:
    """Resolve the device used for primary model inputs."""

    if not torch.cuda.is_available():
        return torch.device("cpu")

    if hasattr(model, "hf_device_map"):
        for device_value in model.hf_device_map.values():
            if isinstance(device_value, int):
                return torch.device(f"cuda:{device_value}")

            if (
                isinstance(device_value, str)
                and device_value.startswith("cuda")
            ):
                return torch.device(device_value)

    return torch.device("cuda:0")

def move_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Move tensor fields to the primary input device."""

    for key, value in list(batch.items()):
        if key == "audio_info":
            continue

        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)

    return batch

def run_inference(args: argparse.Namespace) -> None:
    """Execute FSD50K inference and save sample-level results."""

    run_name = build_run_name(args)

    if args.checkpoint_path is None:
        if args.output_root is None:
            raise ValueError(
                "--output-root is required when "
                "--checkpoint-path is not specified."
            )

        checkpoint_path = find_latest_checkpoint(
            run_name=run_name,
            output_root=args.output_root,
        )
    else:
        checkpoint_path = args.checkpoint_path

    output_path = args.output_path

    if output_path is None:
        if args.output_root is None:
            raise ValueError(
                "--output-path or --output-root must be specified."
            )

        output_path = (
            args.output_root
            / run_name
            / f"inference_results_{run_name}.jsonl"
        )

    label_space = load_official_vocabulary(
        vocabulary_path=args.vocabulary_path,
        label_column=args.vocab_label_column,
    )

    use_response_prefix = (
        args.use_response_prefix
        if args.use_response_prefix is not None
        else args.supervision_type == "cot"
    )

    if args.max_new_tokens is None:
        max_new_tokens = (
            128
            if args.supervision_type == "direct"
            else 800
        )
    else:
        max_new_tokens = args.max_new_tokens

    print("\nInference configuration")
    print("=" * 64)
    print(f"Run name            : {run_name}")
    print(f"Adapter type        : {args.adapter_type}")
    print(f"Supervision type    : {args.supervision_type}")
    print(f"A-MokA variant      : {args.amoka_variant}")
    print(f"Base model          : {args.model_path}")
    print(f"Checkpoint          : {checkpoint_path}")
    print(f"Evaluation data     : {args.test_data_path}")
    print(f"Vocabulary          : {args.vocabulary_path}")
    print(f"Output              : {output_path}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Maximum samples     : {args.max_samples}")
    print(f"Use response prefix : {use_response_prefix}")
    print(f"Response prefix     : {args.response_prefix!r}")
    print(f"Maximum new tokens  : {max_new_tokens}")
    print("=" * 64)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Loading Qwen2-Audio base model.")

    model, tokenizer = get_model_tokenizer(
        ModelType.qwen2_audio_7b_instruct,
        (
            torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32
        ),
        model_id_or_path=str(args.model_path),
        device_map=(
            "auto"
            if torch.cuda.is_available()
            else None
        ),
        model_kwargs={
            "attn_implementation": args.attention_implementation,
        },
    )

    if args.adapter_type == "amoka":
        model = inject_amoka(
            model=model,
            variant=args.amoka_variant,
        )
    else:
        model = inject_lora(model)

    model = load_checkpoint_with_report(
        model=model,
        checkpoint_path=checkpoint_path,
        adapter_type=args.adapter_type,
        amoka_variant=args.amoka_variant,
    )
    model.eval()

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True

    model.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )
    model.generation_config.eos_token_id = (
        tokenizer.eos_token_id
    )
    model.generation_config.use_cache = True
    model.generation_config.max_new_tokens = max_new_tokens
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.generation_config.repetition_penalty = 1.0

    template = get_template(
        "qwen2-audio",
        tokenizer,
        max_length=args.max_length,
    )

    evaluation_pack = load_evaluation_data(
        data_path=args.test_data_path,
        max_samples=args.max_samples,
    )
    evaluation_type = evaluation_pack["type"]
    dataset = evaluation_pack["data"]

    prefix_ids = (
        tokenizer.encode(
            args.response_prefix,
            add_special_tokens=False,
        )
        if use_response_prefix
        else []
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use --overwrite to replace it."
        )

    if output_path.exists():
        output_path.unlink()

    input_device = resolve_input_device(model)
    official_labels_lower = {
        label.lower()
        for label in label_space
    }

    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0
    sample_f1_values = []

    print(
        f"Starting inference with batch_size={args.batch_size}, "
        f"max_new_tokens={max_new_tokens}."
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for start in tqdm.tqdm(
            range(0, len(dataset), args.batch_size),
            desc="Inference",
        ):
            end = min(
                start + args.batch_size,
                len(dataset),
            )

            if evaluation_type == "cache":
                batch_items = [
                    dataset[index]
                    for index in range(start, end)
                ]
            else:
                batch_items = dataset[start:end]

            encoded_items = []
            target_texts = []

            for item in batch_items:
                if evaluation_type == "cache":
                    target_text = decode_cached_target(
                        item["labels"],
                        tokenizer,
                    )
                    encoded = build_generation_item_from_cache(
                        item=item,
                        prefix_ids=prefix_ids,
                    )
                else:
                    target_text = item["response"]
                    encoded = encode_generation_item_from_jsonl(
                        template=template,
                        item=item,
                        prefix_ids=prefix_ids,
                    )

                target_texts.append(target_text)
                encoded_items.append(encoded)

            batch = template.data_collator(
                encoded_items
            )
            batch = move_batch_to_device(
                batch=batch,
                device=input_device,
            )

            with torch.inference_mode():
                generated_ids = model.generate(
                    **batch,
                    generation_config=model.generation_config,
                )

            padded_input_length = batch["input_ids"].shape[1]

            for batch_index, item in enumerate(batch_items):
                new_tokens = generated_ids[
                    batch_index,
                    padded_input_length:,
                ]

                generated_text = tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )

                full_response = (
                    args.response_prefix + generated_text
                    if use_response_prefix
                    else generated_text
                )

                target_strict, target_raw = extract_labels(
                    target_texts[batch_index],
                    label_space,
                )
                prediction_strict, prediction_raw = extract_labels(
                    full_response,
                    label_space,
                )

                invalid_predictions = [
                    label
                    for label in prediction_raw
                    if label.lower() not in official_labels_lower
                ]

                (
                    precision,
                    recall,
                    sample_f1,
                    true_positive,
                    false_positive,
                    false_negative,
                ) = compute_sample_metrics(
                    target_labels=target_strict,
                    predicted_labels=prediction_strict,
                )

                total_true_positive += true_positive
                total_false_positive += false_positive
                total_false_negative += false_negative
                sample_f1_values.append(sample_f1)

                record = {
                    "sample_index": start + batch_index,
                    "sample_id": item.get("id"),
                    "run_name": run_name,
                    "adapter_type": args.adapter_type,
                    "supervision_type": args.supervision_type,
                    "amoka_variant": args.amoka_variant,
                    "evaluation_type": evaluation_type,
                    "checkpoint_path": str(checkpoint_path),
                    "audio": item.get("audios"),
                    "query": item.get("query"),
                    "ground_truth_strict": target_strict,
                    "ground_truth_raw": target_raw,
                    "prediction_strict": prediction_strict,
                    "prediction_raw": prediction_raw,
                    "invalid_prediction_raw": invalid_predictions,
                    "num_invalid_predictions": len(
                        invalid_predictions
                    ),
                    "sample_precision": precision,
                    "sample_recall": recall,
                    "sample_f1": sample_f1,
                    "ground_truth_text": target_texts[batch_index],
                    "full_response": full_response,
                }

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            del (
                batch,
                generated_ids,
                encoded_items,
                batch_items,
            )
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    micro_precision = (
        total_true_positive
        / (total_true_positive + total_false_positive)
        if total_true_positive + total_false_positive > 0
        else 0.0
    )
    micro_recall = (
        total_true_positive
        / (total_true_positive + total_false_negative)
        if total_true_positive + total_false_negative > 0
        else 0.0
    )
    micro_f1 = (
        2.0
        * micro_precision
        * micro_recall
        / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )
    samples_f1 = (
        float(np.mean(sample_f1_values))
        if sample_f1_values
        else 0.0
    )

    summary = {
        "run_name": run_name,
        "adapter_type": args.adapter_type,
        "supervision_type": args.supervision_type,
        "amoka_variant": args.amoka_variant,
        "evaluation_type": evaluation_type,
        "checkpoint_path": str(checkpoint_path),
        "test_data_path": str(args.test_data_path),
        "output_result_path": str(output_path),
        "num_samples": len(dataset),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "samples_f1": samples_f1,
        "total_true_positive": total_true_positive,
        "total_false_positive": total_false_positive,
        "total_false_negative": total_false_negative,
    }

    summary_path = output_path.with_name(
        f"{output_path.stem}_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nInference summary")
    print("=" * 64)
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=" * 64)
    print(f"Results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")

# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and an optional YAML configuration."""

    config_parser = argparse.ArgumentParser(
        add_help=False,
    )
    config_parser.add_argument(
        "--configs",
        type=Path,
        default=None,
        help="Path to a YAML configuration file.",
    )

    known_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description=(
            "Run A-MokA or LoRA generative multi-label inference "
            "on FSD50K."
        ),
        parents=[config_parser],
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to Qwen2-Audio-7B-Instruct.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Adapter checkpoint directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing training runs.",
    )
    parser.add_argument(
        "--test-data-path",
        type=Path,
        default=None,
        help="Evaluation JSONL file or Hugging Face cache directory.",
    )
    parser.add_argument(
        "--vocabulary-path",
        type=Path,
        default=None,
        help="Path to the official FSD50K vocabulary.csv.",
    )
    parser.add_argument(
        "--vocab-label-column",
        type=int,
        default=1,
        help="Column index containing category names.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--adapter-type",
        choices=["amoka", "lora"],
        default="amoka",
    )
    parser.add_argument(
        "--supervision-type",
        choices=["cot", "direct"],
        default="cot",
    )
    parser.add_argument(
        "--amoka-variant",
        choices=[
            "full",
            "wo_audio",
            "wo_mlp",
            "wo_attn",
            "all_r32",
            "all_r16",
        ],
        default="full",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--response-prefix",
        type=str,
        default="Reasoning Steps:\n",
    )
    parser.add_argument(
        "--use-response-prefix",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--attention-implementation",
        choices=[
            "eager",
            "sdpa",
            "flash_attention_2",
        ],
        default="sdpa",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    if known_args.config is not None:
        config = load_yaml_config(
            known_args.config
        )

        valid_destinations = {
            action.dest
            for action in parser._actions
        }
        unknown_keys = set(config) - valid_destinations

        if unknown_keys:
            parser.error(
                "Unknown configuration fields: "
                f"{sorted(unknown_keys)}"
            )

        # YAML values become defaults. Explicit command-line arguments
        # still take precedence.
        parser.set_defaults(**config)

    args = parser.parse_args()

    path_fields = [
        "model_path",
        "checkpoint_path",
        "output_root",
        "test_data_path",
        "vocabulary_path",
        "output_path",
    ]

    for field in path_fields:
        value = getattr(args, field, None)

        if value is not None and not isinstance(value, Path):
            setattr(args, field, Path(value))

    # Continue with required-field validation.

    required_fields = {
        "model_path": args.model_path,
        "test_data_path": args.test_data_path,
        "vocabulary_path": args.vocabulary_path,
    }

    missing_fields = [
        name
        for name, value in required_fields.items()
        if value is None
    ]

    if missing_fields:
        parser.error(
            "Missing required configuration fields: "
            f"{missing_fields}"
        )

    if args.batch_size <= 0:
        parser.error(
            "--batch-size must be greater than zero."
        )

    if (
        args.max_samples is not None
        and args.max_samples <= 0
    ):
        parser.error(
            "--max-samples must be greater than zero."
        )

    if (
        args.max_new_tokens is not None
        and args.max_new_tokens <= 0
    ):
        parser.error(
            "--max-new-tokens must be greater than zero."
        )

    return args

if __name__ == "__main__":
    run_inference(parse_args())
