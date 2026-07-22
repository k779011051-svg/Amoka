# -*- coding: utf-8 -*-
"""
Evaluate generative multi-label audio tagging results on FSD50K.

Two evaluation settings are provided:

1. Strict exact-label evaluation:
   Predictions must be mapped exactly to the official FSD50K vocabulary.

2. Sentence-BERT semantic projection:
   Open-form predictions are projected into the official label space using
   semantic similarity and a one-to-one matching constraint.

Expected JSONL fields
---------------------
{
    "ground_truth_strict": ["Speech", "Vehicle"],
    "prediction_strict": ["Speech", "Car"],
    "prediction_raw": ["human speech", "passing automobile"]
}
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

def normalize_key(label: Any) -> str:
    """Normalize one label for lexical comparison."""

    if label is None:
        return ""

    normalized = str(label).strip().lower()
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("-", " ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace(",", " and ")
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()

def ensure_label_list(value: Any) -> List[str]:
    """
    Convert a JSON field into a list of label strings.

    This prevents an individual string from being incorrectly iterated
    character by character.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, tuple):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        return [
            item.strip()
            for item in text.split(",")
            if item.strip()
        ]

    return [str(value).strip()]

def deduplicate_labels(labels: Sequence[str]) -> List[str]:
    """Remove duplicate labels while preserving input order."""

    seen = set()
    output = []

    for label in labels:
        key = normalize_key(label)

        if key and key not in seen:
            output.append(label)
            seen.add(key)

    return output

def load_official_vocabulary(
    vocabulary_path: Path,
    label_column: int,
) -> List[str]:
    """Load the official 200-category FSD50K vocabulary."""

    if not vocabulary_path.exists():
        raise FileNotFoundError(
            f"Official vocabulary not found: {vocabulary_path}"
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
            f"Expected 200 official labels, but found {len(labels)}."
        )

    return labels

def build_canonical_mapping(
    label_space: Sequence[str],
) -> Dict[str, str]:
    """Build normalized variants of each official label."""

    mapping = {}

    for label in label_space:
        variants = {
            label,
            label.replace("_", " "),
            label.replace("_and_", ", "),
            label.replace("_and_", " and "),
            label.replace("_", "-"),
        }

        for variant in variants:
            mapping[normalize_key(variant)] = label

    return mapping

def canonicalize_label_list(
    labels: Any,
    canonical_mapping: Dict[str, str],
    keep_unmapped: bool = False,
) -> List[str]:
    """Map lexical variants to official FSD50K labels."""

    output = []

    for label in ensure_label_list(labels):
        mapped = canonical_mapping.get(
            normalize_key(label)
        )

        if mapped is not None:
            output.append(mapped)
        elif keep_unmapped:
            output.append(label)

    return deduplicate_labels(output)

def semantic_score(
    predicted_label: str,
    target_label: str,
    model: SentenceTransformer,
) -> float:
    """Compute semantic similarity between two category strings."""

    predicted_text = normalize_key(predicted_label)
    target_text = normalize_key(target_label)

    if not predicted_text or not target_text:
        return 0.0

    if predicted_text == target_text:
        return 1.0

    if (
        len(predicted_text) >= 3
        and predicted_text in target_text
    ):
        return 1.0

    if (
        len(target_text) >= 3
        and target_text in predicted_text
    ):
        return 1.0

    embeddings = model.encode(
        [predicted_text, target_text],
        convert_to_tensor=True,
    )

    return float(
        util.cos_sim(
            embeddings[0],
            embeddings[1],
        ).item()
    )

def map_prediction_to_label_space(
    predicted_label: str,
    label_space: Sequence[str],
    label_embeddings: torch.Tensor,
    model: SentenceTransformer,
    threshold: float,
) -> Tuple[str, float]:
    """Map an open-form prediction to its nearest official category."""

    normalized_prediction = normalize_key(predicted_label)

    if not normalized_prediction:
        return None, 0.0

    prediction_embedding = model.encode(
        normalized_prediction,
        convert_to_tensor=True,
    )

    scores = util.cos_sim(
        prediction_embedding,
        label_embeddings,
    )[0]

    best_index = int(torch.argmax(scores).item())
    best_score = float(scores[best_index].item())

    if best_score < threshold:
        return None, best_score

    return label_space[best_index], best_score

def one_to_one_semantic_projection(
    raw_predictions: Sequence[str],
    target_labels: Sequence[str],
    label_space: Sequence[str],
    label_embeddings: torch.Tensor,
    model: SentenceTransformer,
    threshold: float,
) -> List[str]:
    """
    Project open-form predictions into the official label space.

    Target-aware one-to-one matching is applied first. Remaining
    predictions are independently mapped to the closest official label.
    """

    raw_predictions = deduplicate_labels(raw_predictions)
    target_labels = deduplicate_labels(target_labels)

    candidate_pairs = []

    for prediction_index, predicted_label in enumerate(
        raw_predictions
    ):
        for target_index, target_label in enumerate(
            target_labels
        ):
            score = semantic_score(
                predicted_label,
                target_label,
                model,
            )

            if score >= threshold:
                candidate_pairs.append(
                    (
                        score,
                        prediction_index,
                        target_index,
                    )
                )

    candidate_pairs.sort(
        reverse=True,
        key=lambda item: item[0],
    )

    matched_predictions = set()
    matched_targets = set()
    projected_labels = []

    for score, prediction_index, target_index in candidate_pairs:
        if prediction_index in matched_predictions:
            continue

        if target_index in matched_targets:
            continue

        matched_predictions.add(prediction_index)
        matched_targets.add(target_index)
        projected_labels.append(
            target_labels[target_index]
        )

    for prediction_index, predicted_label in enumerate(
        raw_predictions
    ):
        if prediction_index in matched_predictions:
            continue

        mapped_label, _ = map_prediction_to_label_space(
            predicted_label=predicted_label,
            label_space=label_space,
            label_embeddings=label_embeddings,
            model=model,
            threshold=threshold,
        )

        if mapped_label is not None:
            projected_labels.append(mapped_label)

    return deduplicate_labels(projected_labels)

def compute_metrics(
    target_binary: np.ndarray,
    prediction_binary: np.ndarray,
    active_mask: np.ndarray,
) -> Dict[str, Any]:
    """Compute strict or semantic multi-label metrics."""

    per_class_f1 = f1_score(
        target_binary,
        prediction_binary,
        average=None,
        zero_division=0,
    )
    per_class_precision = precision_score(
        target_binary,
        prediction_binary,
        average=None,
        zero_division=0,
    )
    per_class_recall = recall_score(
        target_binary,
        prediction_binary,
        average=None,
        zero_division=0,
    )

    return {
        "samples_f1": f1_score(
            target_binary,
            prediction_binary,
            average="samples",
            zero_division=0,
        ),
        "macro_precision": float(
            np.mean(per_class_precision[active_mask])
        ),
        "macro_recall": float(
            np.mean(per_class_recall[active_mask])
        ),
        "macro_f1": float(
            np.mean(per_class_f1[active_mask])
        ),
        "micro_precision": precision_score(
            target_binary,
            prediction_binary,
            average="micro",
            zero_division=0,
        ),
        "micro_recall": recall_score(
            target_binary,
            prediction_binary,
            average="micro",
            zero_division=0,
        ),
        "micro_f1": f1_score(
            target_binary,
            prediction_binary,
            average="micro",
            zero_division=0,
        ),
    }

def print_metrics(
    title: str,
    metrics: Dict[str, float],
) -> None:
    """Print one multi-label metric block."""

    print(title)
    print(f"  Samples F1       : {metrics['samples_f1'] * 100:.2f}%")
    print(f"  Macro Precision  : {metrics['macro_precision'] * 100:.2f}%")
    print(f"  Macro Recall     : {metrics['macro_recall'] * 100:.2f}%")
    print(f"  Macro F1         : {metrics['macro_f1'] * 100:.2f}%")
    print(f"  Micro Precision  : {metrics['micro_precision'] * 100:.2f}%")
    print(f"  Micro Recall     : {metrics['micro_recall'] * 100:.2f}%")
    print(f"  Micro F1         : {metrics['micro_f1'] * 100:.2f}%")
    print()

def evaluate(args: argparse.Namespace) -> None:
    """Evaluate strict and semantic FSD50K predictions."""

    label_space = load_official_vocabulary(
        vocabulary_path=args.vocabulary_path,
        label_column=args.vocab_label_column,
    )
    canonical_mapping = build_canonical_mapping(
        label_space
    )

    print(f"Loading Sentence-BERT model: {args.sbert_model}")
    semantic_model = SentenceTransformer(
        args.sbert_model
    )

    normalized_label_texts = [
        normalize_key(label)
        for label in label_space
    ]
    label_embeddings = semantic_model.encode(
        normalized_label_texts,
        convert_to_tensor=True,
    )

    targets = []
    strict_predictions = []
    semantic_predictions = []

    corrupted_samples = 0
    empty_target_samples = 0

    with args.result_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        lines = file.readlines()

    for line in tqdm(lines, desc="Evaluating"):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            corrupted_samples += 1
            continue

        target = canonicalize_label_list(
            record.get(args.target_field, []),
            canonical_mapping,
            keep_unmapped=False,
        )

        if not target:
            empty_target_samples += 1
            continue

        strict_prediction = canonicalize_label_list(
            record.get(args.strict_prediction_field, []),
            canonical_mapping,
            keep_unmapped=False,
        )

        raw_semantic_prediction = canonicalize_label_list(
            record.get(args.semantic_prediction_field, []),
            canonical_mapping,
            keep_unmapped=True,
        )

        semantic_prediction = one_to_one_semantic_projection(
            raw_predictions=raw_semantic_prediction,
            target_labels=target,
            label_space=label_space,
            label_embeddings=label_embeddings,
            model=semantic_model,
            threshold=args.semantic_threshold,
        )

        targets.append(target)
        strict_predictions.append(strict_prediction)
        semantic_predictions.append(semantic_prediction)

    if not targets:
        raise RuntimeError(
            "No valid samples were evaluated."
        )

    binarizer = MultiLabelBinarizer(
        classes=label_space
    )

    target_binary = binarizer.fit_transform(targets)
    strict_binary = binarizer.transform(
        strict_predictions
    )
    semantic_binary = binarizer.transform(
        semantic_predictions
    )

    active_mask = target_binary.sum(axis=0) > 0

    strict_metrics = compute_metrics(
        target_binary=target_binary,
        prediction_binary=strict_binary,
        active_mask=active_mask,
    )
    semantic_metrics = compute_metrics(
        target_binary=target_binary,
        prediction_binary=semantic_binary,
        active_mask=active_mask,
    )

    print("=" * 72)
    print("FSD50K Multi-Label Evaluation")
    print("=" * 72)
    print(f"Valid samples          : {len(targets)}")
    print(f"Corrupted samples      : {corrupted_samples}")
    print(f"Empty target samples   : {empty_target_samples}")
    print(f"Active GT categories   : {int(active_mask.sum())}/200")
    print(f"Semantic threshold     : {args.semantic_threshold}")
    print("=" * 72)
    print()

    print_metrics(
        "[1] Strict exact-label evaluation",
        strict_metrics,
    )
    print_metrics(
        "[2] Sentence-BERT semantic projection",
        semantic_metrics,
    )

    if args.save_summary:
        summary = {
            "result_path": str(args.result_path),
            "valid_samples": len(targets),
            "corrupted_samples": corrupted_samples,
            "empty_target_samples": empty_target_samples,
            "active_gt_categories": int(active_mask.sum()),
            "semantic_threshold": args.semantic_threshold,
            "strict": {
                key: float(value)
                for key, value in strict_metrics.items()
            },
            "semantic": {
                key: float(value)
                for key, value in semantic_metrics.items()
            },
        }

        args.save_summary.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.save_summary.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Summary saved to: {args.save_summary}")

def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate generative FSD50K tagging outputs."
    )

    parser.add_argument(
        "--result-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--vocabulary-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--vocab-label-column",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--target-field",
        type=str,
        default="ground_truth_strict",
    )
    parser.add_argument(
        "--strict-prediction-field",
        type=str,
        default="prediction_strict",
    )
    parser.add_argument(
        "--semantic-prediction-field",
        type=str,
        default="prediction_raw",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--sbert-model",
        type=str,
        default="all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--save-summary",
        type=Path,
        default=None,
    )

    return parser.parse_args()

if __name__ == "__main__":
    evaluate(parse_args())