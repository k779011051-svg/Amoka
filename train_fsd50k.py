# -*- coding: utf-8 -*-
"""
Fine-tune Qwen2-Audio on FSD50K using A-MokA or homogeneous LoRA.

Supported adapter configurations
--------------------------------
A-MokA variants:
    full:
        Audio/projector branch + MLP branch + broad Q/V attention branch.
    wo_audio:
        Remove the audio tower and multimodal projector branch.
    wo_mlp:
        Remove the language-model MLP branch.
    wo_attn:
        Remove the Q/V attention branch.
    all_r32:
        Use rank 32 for all three branches.

Supervision types:
    cot:
        Acoustic chain-of-thought supervision.
    direct:
        Direct label-only supervision.
"""

import argparse
import csv
import gc
import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from sklearn.metrics import (
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from swift.llm import (
    ModelType,
    SftArguments,
    get_model_tokenizer,
    get_template,
)
from swift.trainers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

import swanlab
from swanlab.integration.huggingface import SwanLabCallback

warnings.filterwarnings("ignore")

tokenizer = None
sft_args = None
label_space: List[str] = []
multilabel_binarizer = None

# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------
def load_official_vocabulary(
    vocabulary_path: Path,
    label_column: int = 1,
) -> List[str]:
    """Load the 200 official FSD50K labels."""

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

        for row in reader:
            if not row or label_column >= len(row):
                continue

            candidate = row[label_column].strip().strip("\"'")

            if candidate:
                labels.append(candidate)

    labels = list(dict.fromkeys(labels))

    if len(labels) != 200:
        raise ValueError(
            f"Expected 200 FSD50K labels, but found {len(labels)}."
        )

    return labels

# ----------------------------------------------------------------------
# Adapter layers
# ----------------------------------------------------------------------
class MokALinear(nn.Module):
    """Modality- and knowledge-aware low-rank adaptation layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 32,
        alpha: int = 64,
        module_name: str = "",
    ):
        super().__init__()

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        input_features = base_layer.in_features
        output_features = base_layer.out_features

        if (
            "audio_tower" in module_name
            or "multi_modal_projector" in module_name
        ):
            self.branch_type = "audio"
        elif any(
            key in module_name
            for key in ("gate_proj", "up_proj", "down_proj")
        ):
            self.branch_type = "mlp"
        else:
            self.branch_type = "attention"

        if self.branch_type == "audio":
            self.A_audio = nn.Linear(
                input_features,
                rank,
                bias=False,
            )
            nn.init.kaiming_uniform_(
                self.A_audio.weight,
                a=np.sqrt(5),
            )

        elif self.branch_type == "mlp":
            self.A_mlp = nn.Linear(
                input_features,
                rank,
                bias=False,
            )
            nn.init.kaiming_uniform_(
                self.A_mlp.weight,
                a=np.sqrt(5),
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
            nn.init.kaiming_uniform_(
                self.A_attn.weight,
                a=np.sqrt(5),
            )
            self.attn_interaction = nn.Parameter(
                torch.zeros(1, rank)
            )

        self.B = nn.Linear(
            rank,
            output_features,
            bias=False,
        )
        nn.init.normal_(self.B.weight, std=0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base layer and the trainable adapter offset."""

        base_output = self.base_layer(inputs)

        adapter_dtype = self.B.weight.dtype
        adapter_inputs = inputs.to(adapter_dtype)

        if self.branch_type == "audio":
            offset = self.B(self.A_audio(adapter_inputs))

        elif self.branch_type == "mlp":
            gate = self.memory_gate.to(
                device=inputs.device,
                dtype=adapter_dtype,
            )
            offset = self.B(
                self.A_mlp(adapter_inputs) * (1.0 + gate)
            )

        else:
            interaction = self.attn_interaction.to(
                device=inputs.device,
                dtype=adapter_dtype,
            )
            offset = self.B(
                self.A_attn(adapter_inputs)
                * (1.0 + interaction)
            )

        return base_output + (
            offset * self.scaling
        ).to(base_output.dtype)

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
        self.rank = rank
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
        adapter_inputs = inputs.to(self.lora_B.weight.dtype)

        offset = self.lora_B(
            self.lora_A(adapter_inputs)
        )

        return base_output + (
            offset * self.scaling
        ).to(base_output.dtype)

# ----------------------------------------------------------------------
# Adapter injection
# ----------------------------------------------------------------------
def inject_amoka(
    model: nn.Module,
    variant: str,
) -> nn.Module:
    """Inject A-MokA layers according to the selected ablation variant."""

    supported_variants = {
        "full",
        "wo_audio",
        "wo_mlp",
        "wo_attn",
        "all_r32",
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

    if variant == "all_r32":
        branch_configs = {
            "Audio": {"rank": 32, "alpha": 64},
            "MLP": {"rank": 32, "alpha": 64},
            "Attention": {"rank": 32, "alpha": 64},
        }
    else:
        branch_configs = {
            "Audio": {"rank": 32, "alpha": 64},
            "MLP": {"rank": 32, "alpha": 64},
            "Attention": {"rank": 16, "alpha": 64},
        }

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

        # Audio modules are checked first because internal audio-attention
        # projections may also match the generic Q/V patterns.
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

        parent_name = ".".join(module_name.split(".")[:-1])
        child_name = module_name.split(".")[-1]
        parent_module = model.get_submodule(parent_name)

        config = branch_configs[branch]

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
    print(f"Audio/projector layers  : {injected_counts['Audio']}")
    print(f"MLP layers              : {injected_counts['MLP']}")
    print(f"Attention Q/V layers    : {injected_counts['Attention']}")

    return model

def inject_lora(model: nn.Module) -> nn.Module:
    """Inject the homogeneous LoRA baseline."""

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

        parent_name = ".".join(module_name.split(".")[:-1])
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

    print("LoRA injection completed.")
    print(f"Injected linear layers: {injected_count}")

    return model

# ----------------------------------------------------------------------
# Label extraction and metrics
# ----------------------------------------------------------------------
def extract_multilabels(
    text: str,
    official_labels: Sequence[str],
) -> List[str]:
    """Extract official FSD50K labels from a generated conclusion."""

    if not isinstance(text, str):
        return []

    text_lower = text.lower()

    if "final conclusion" in text_lower:
        conclusion_index = text_lower.rfind(
            "final conclusion"
        )
        text = text[conclusion_index:]
        text_lower = text.lower()

    anchor = "categories are"

    if anchor not in text_lower:
        anchor = "category is"

    if anchor not in text_lower:
        return []

    label_text = text_lower.split(anchor, 1)[-1]
    label_text = label_text.split("\n", 1)[0]
    label_text = re.sub(r"\.\s*$", "", label_text.strip())

    label_lookup = {
        label.lower(): label
        for label in official_labels
    }

    extracted = []

    for raw_label in label_text.split(","):
        candidate = raw_label.strip(
            " .。,:;，；\"'()"
        )
        candidate = candidate.replace(" ", "_")

        if candidate.lower() in label_lookup:
            extracted.append(
                label_lookup[candidate.lower()]
            )

    return list(dict.fromkeys(extracted))

def compute_metrics(eval_predictions) -> Dict[str, float]:
    """Compute strict multi-label metrics for generated outputs."""

    global tokenizer
    global label_space
    global multilabel_binarizer
    global sft_args

    predictions, labels = eval_predictions

    decoded_predictions = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True,
    )

    labels = np.where(
        labels == -100,
        tokenizer.pad_token_id,
        labels,
    )
    decoded_targets = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    predicted_labels = [
        extract_multilabels(text, label_space)
        for text in decoded_predictions
    ]
    target_labels = [
        extract_multilabels(text, label_space)
        for text in decoded_targets
    ]

    valid_indices = [
        index
        for index, targets in enumerate(target_labels)
        if targets
    ]

    if not valid_indices:
        return {
            "f1_macro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_micro": 0.0,
            "precision_micro": 0.0,
            "recall_micro": 0.0,
            "f1_samples": 0.0,
        }

    target_labels = [
        target_labels[index]
        for index in valid_indices
    ]
    predicted_labels = [
        predicted_labels[index]
        for index in valid_indices
    ]

    target_binary = multilabel_binarizer.fit_transform(
        target_labels
    )
    prediction_binary = multilabel_binarizer.transform(
        predicted_labels
    )

    metrics = {
        "f1_macro": f1_score(
            target_binary,
            prediction_binary,
            average="macro",
            zero_division=0,
        ),
        "precision_macro": precision_score(
            target_binary,
            prediction_binary,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            target_binary,
            prediction_binary,
            average="macro",
            zero_division=0,
        ),
        "f1_micro": f1_score(
            target_binary,
            prediction_binary,
            average="micro",
            zero_division=0,
        ),
        "precision_micro": precision_score(
            target_binary,
            prediction_binary,
            average="micro",
            zero_division=0,
        ),
        "recall_micro": recall_score(
            target_binary,
            prediction_binary,
            average="micro",
            zero_division=0,
        ),
        "f1_samples": f1_score(
            target_binary,
            prediction_binary,
            average="samples",
            zero_division=0,
        ),
    }

    try:
        matrices = multilabel_confusion_matrix(
            target_binary,
            prediction_binary,
        )

        summary = {}

        for index, category in enumerate(
            multilabel_binarizer.classes_
        ):
            true_negative, false_positive, false_negative, true_positive = (
                matrices[index].ravel()
            )

            if (
                true_positive
                + false_positive
                + false_negative
                > 0
            ):
                summary[category] = {
                    "TP": int(true_positive),
                    "FP": int(false_positive),
                    "TN": int(true_negative),
                    "FN": int(false_negative),
                }

        output_path = (
            Path(sft_args.output_dir)
            / "multilabel_confusion_matrix.json"
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )

    except Exception as exception:
        print(
            "Failed to save the multi-label confusion summary: "
            f"{exception}"
        )

    return metrics

# ----------------------------------------------------------------------
# Data and trainer utilities
# ----------------------------------------------------------------------
class ClearCacheCallback(TrainerCallback):
    """Release unused memory after each evaluation phase."""

    def on_evaluate(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("Evaluation cache cleared.")

def convert_dataset_items_to_tensors(examples):
    """Convert cached dataset fields to PyTorch tensors."""

    for key in list(examples):
        if key == "audio_info":
            continue

        converted_values = []

        for item in examples[key]:
            if item is None:
                converted_values.append(None)
            elif isinstance(item, torch.Tensor):
                converted_values.append(item)
            elif isinstance(item, (list, np.ndarray)):
                converted_values.append(torch.tensor(item))
            else:
                converted_values.append(torch.tensor(item))

        examples[key] = converted_values

    return examples

def load_cached_datasets(
    training_cache: Path,
    validation_cache: Path,
):
    """Load and validate preprocessed FSD50K datasets."""

    if not training_cache.is_dir():
        raise FileNotFoundError(
            f"Training cache not found: {training_cache}"
        )

    if not validation_cache.is_dir():
        raise FileNotFoundError(
            f"Validation cache not found: {validation_cache}"
        )

    training_dataset = load_from_disk(
        str(training_cache)
    )
    validation_dataset = load_from_disk(
        str(validation_cache)
    )

    required_columns = {
        "input_ids",
        "labels",
        "input_features",
        "feature_attention_mask",
    }

    for split_name, dataset in (
        ("training", training_dataset),
        ("validation", validation_dataset),
    ):
        missing = required_columns - set(dataset.column_names)

        if missing:
            raise ValueError(
                f"The {split_name} cache is missing columns: "
                f"{sorted(missing)}"
            )

    print(
        f"Training cache loaded: {training_cache}, "
        f"samples={len(training_dataset)}"
    )
    print(
        f"Validation cache loaded: {validation_cache}, "
        f"samples={len(validation_dataset)}"
    )

    return training_dataset, validation_dataset

def configure_trainable_parameters(
    model: nn.Module,
    adapter_type: str,
) -> None:
    """Freeze the base model and activate only adapter parameters."""

    model.requires_grad_(False)

    if adapter_type == "amoka":
        trainable_keys = (
            "A_audio",
            "A_mlp",
            "A_attn",
            "B",
            "memory_gate",
            "attn_interaction",
        )
    else:
        trainable_keys = (
            "lora_A",
            "lora_B",
        )

    for parameter_name, parameter in model.named_parameters():
        if any(
            key in parameter_name
            for key in trainable_keys
        ):
            parameter.requires_grad = True

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Total parameters     : {total_parameters:,}")
    print(f"Trainable parameters : {trainable_parameters:,}")
    print(
        "Trainable ratio      : "
        f"{100.0 * trainable_parameters / total_parameters:.4f}%"
    )

# ----------------------------------------------------------------------
# Main training procedure
# ----------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    """Run one FSD50K fine-tuning experiment."""

    global tokenizer
    global sft_args
    global label_space
    global multilabel_binarizer

    label_space = load_official_vocabulary(
        vocabulary_path=args.vocabulary_path,
        label_column=args.vocab_label_column,
    )
    multilabel_binarizer = MultiLabelBinarizer(
        classes=label_space
    )

    run_name = (
        f"FSD50K-{args.adapter_type.upper()}-"
        f"{args.supervision_type.upper()}"
    )

    if (
        args.adapter_type == "amoka"
        and args.amoka_variant != "full"
    ):
        run_name += f"-{args.amoka_variant.upper()}"

    output_directory = args.output_root / run_name
    output_directory.mkdir(parents=True, exist_ok=True)

    dummy_dataset = output_directory / "dummy_dataset.jsonl"

    if not dummy_dataset.exists():
        with dummy_dataset.open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    {
                        "query": "dummy",
                        "response": "dummy",
                    }
                )
                + "\n"
            )

    sft_args = SftArguments(
        model_type=ModelType.qwen2_audio_7b_instruct,
        model_id_or_path=str(args.model_path),
        output_dir=str(output_directory),
        dataset=[str(dummy_dataset)],
        val_dataset=[str(dummy_dataset)],
        dtype="bf16",
        max_length=args.max_length,
        template_type="qwen2-audio",
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if args.enable_swanlab:
        api_key = os.getenv("SWANLAB_API_KEY")

        if api_key:
            swanlab.login(api_key=api_key)

        swanlab.init(
            project=args.swanlab_project,
            experiment_name=run_name,
            config={
                **sft_args.__dict__,
                "adapter_type": args.adapter_type,
                "supervision_type": args.supervision_type,
                "amoka_variant": args.amoka_variant,
            },
        )

    model, tokenizer = get_model_tokenizer(
        sft_args.model_type,
        sft_args.torch_dtype,
        model_id_or_path=sft_args.model_id_or_path,
        model_kwargs={"device_map": {"": args.device}},
    )

    if args.adapter_type == "amoka":
        model = inject_amoka(
            model=model,
            variant=args.amoka_variant,
        )
    else:
        model = inject_lora(model)

    configure_trainable_parameters(
        model=model,
        adapter_type=args.adapter_type,
    )

    model.enable_input_require_grads()
    model.config.use_cache = False

    template = get_template(
        sft_args.template_type,
        tokenizer,
        sft_args.system,
        sft_args.max_length,
    )

    training_dataset, validation_dataset = load_cached_datasets(
        training_cache=args.training_cache,
        validation_cache=args.validation_cache,
    )

    if args.fast_dev_run:
        training_dataset = training_dataset.select(
            range(min(4, len(training_dataset)))
        )
        validation_dataset = validation_dataset.select(
            range(min(2, len(validation_dataset)))
        )

    training_dataset.set_transform(
        convert_dataset_items_to_tensors
    )
    validation_dataset.set_transform(
        convert_dataset_items_to_tensors
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )
    model.generation_config.eos_token_id = (
        tokenizer.eos_token_id
    )
    model.generation_config.max_new_tokens = (
        128 if args.fast_dev_run else args.max_new_tokens
    )

    training_arguments = Seq2SeqTrainingArguments(
        output_dir=str(output_directory),
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=(
            1 if args.fast_dev_run else args.num_train_epochs
        ),
        max_steps=2 if args.fast_dev_run else -1,
        per_device_train_batch_size=(
            1 if args.fast_dev_run else args.train_batch_size
        ),
        gradient_accumulation_steps=(
            1
            if args.fast_dev_run
            else args.gradient_accumulation_steps
        ),
        per_device_eval_batch_size=(
            1 if args.fast_dev_run else args.eval_batch_size
        ),
        dataloader_num_workers=(
            0 if args.fast_dev_run else args.num_workers
        ),
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=False,
        predict_with_generate=True,
        generation_max_length=(
            128 if args.fast_dev_run else args.max_new_tokens
        ),
        eval_accumulation_steps=1,
        eval_strategy=(
            "steps" if args.fast_dev_run else "epoch"
        ),
        eval_steps=1 if args.fast_dev_run else None,
        save_strategy=(
            "steps" if args.fast_dev_run else "epoch"
        ),
        save_steps=1 if args.fast_dev_run else None,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=1,
        remove_unused_columns=False,
    )

    callbacks = [ClearCacheCallback()]

    if args.enable_swanlab:
        callbacks.append(SwanLabCallback())

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=template.data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer._signature_columns = [
        "input_ids",
        "labels",
        "input_features",
        "feature_attention_mask",
        "attention_mask",
        "audio_values",
        "audio_info",
    ]

    print(f"Starting experiment: {run_name}")
    trainer.train()

    if args.enable_swanlab:
        swanlab.finish()

def parse_args() -> argparse.Namespace:
    """Parse training arguments."""

    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2-Audio on FSD50K."
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--vocabulary-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--training-cache",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--validation-cache",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
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
        ],
        default="full",
    )
    parser.add_argument(
        "--vocab-label-column",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
    )
    parser.add_argument(
        "--enable-swanlab",
        action="store_true",
    )
    parser.add_argument(
        "--swanlab-project",
        type=str,
        default="Qwen2-Audio-FSD50K",
    )

    return parser.parse_args()

if __name__ == "__main__":
    train(parse_args())