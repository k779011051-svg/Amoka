# -*- coding: utf-8 -*-
"""
Construct A-CoT supervision targets for the FSD50K development set.

The script performs the following operations:

1. Loads the official FSD50K vocabulary.
2. Loads frozen class-level acoustic templates.
3. Loads the controlled timbre lexicon.
4. Resolves official labels through an explicit template alias mapping.
5. Verifies the expected 160/40 category coverage.
6. Constructs multi-label A-CoT responses.
7. Writes training and validation samples in JSONL format.

The original FSD50K audio files are not redistributed by this script.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

# ----------------------------------------------------------------------
# Resource loading
# ----------------------------------------------------------------------
def load_json(path: Path) -> Any:
    """Load a UTF-8 encoded JSON file."""

    if not path.exists():
        raise FileNotFoundError(f"JSON resource not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def load_official_vocabulary(
    vocabulary_path: Path,
    label_column: int = 1,
) -> List[str]:
    """
    Load the 200 official FSD50K category names from vocabulary.csv.

    Parameters
    ----------
    vocabulary_path:
        Path to the official FSD50K vocabulary.csv file.
    label_column:
        Zero-based index of the column containing category names.
    """

    if not vocabulary_path.exists():
        raise FileNotFoundError(
            f"Official FSD50K vocabulary not found: {vocabulary_path}"
        )

    dataframe = pd.read_csv(
        vocabulary_path,
        header=None,
        encoding="utf-8-sig",
    )

    if label_column >= dataframe.shape[1]:
        raise ValueError(
            f"Label column {label_column} is out of range for "
            f"{vocabulary_path}, which has {dataframe.shape[1]} columns."
        )

    labels = (
        dataframe.iloc[:, label_column]
        .astype(str)
        .str.strip()
        .tolist()
    )
    labels = list(dict.fromkeys(label for label in labels if label))

    if len(labels) != 200:
        raise ValueError(
            f"Expected 200 official FSD50K labels, but found {len(labels)}. "
            "Check the vocabulary file and --vocab-label-column."
        )

    return labels

# ----------------------------------------------------------------------
# Coverage validation
# ----------------------------------------------------------------------
def check_template_coverage(
    official_labels: Sequence[str],
    feature_templates: Dict[str, Dict[str, Any]],
    alias_mapping: Dict[str, str],
    expected_coverage: int = 160,
) -> Dict[str, set]:
    """
    Validate direct and alias-based template coverage.

    Returns
    -------
    dict
        Sets of directly covered, alias-covered, covered, and uncovered
        official categories.
    """

    official_set = {
        str(label).strip()
        for label in official_labels
        if str(label).strip()
    }
    feature_keys = set(feature_templates)

    exact_covered = {
        label
        for label in official_set
        if label in feature_keys
    }

    alias_covered = {
        label
        for label in official_set
        if (
            label not in exact_covered
            and label in alias_mapping
            and alias_mapping[label] in feature_keys
        )
    }

    covered = exact_covered | alias_covered
    uncovered = official_set - covered

    invalid_aliases = {
        source: target
        for source, target in alias_mapping.items()
        if source not in official_set or target not in feature_keys
    }

    print("=" * 72)
    print("FSD50K template coverage")
    print("=" * 72)
    print(f"Official labels       : {len(official_set)}")
    print(f"Feature entries       : {len(feature_keys)}")
    print(f"Exact coverage        : {len(exact_covered)}")
    print(f"Alias coverage        : {len(alias_covered)}")
    print(f"Total coverage        : {len(covered)}")
    print(f"Label-only categories : {len(uncovered)}")

    if invalid_aliases:
        print("\nInvalid alias mappings:")
        for source, target in sorted(invalid_aliases.items()):
            print(f"  {source} -> {target}")

    print("\nAlias-covered categories:")
    for label in sorted(alias_covered):
        print(f"  {label} -> {alias_mapping[label]}")

    print("\nLabel-only categories:")
    for label in sorted(uncovered):
        print(f"  {label}")

    if invalid_aliases:
        raise ValueError(
            f"Found {len(invalid_aliases)} invalid alias mappings."
        )

    if len(covered) != expected_coverage:
        raise ValueError(
            f"Expected {expected_coverage} template-covered categories, "
            f"but obtained {len(covered)}."
        )

    expected_uncovered = len(official_set) - expected_coverage
    if len(uncovered) != expected_uncovered:
        raise ValueError(
            f"Expected {expected_uncovered} label-only categories, "
            f"but obtained {len(uncovered)}."
        )

    return {
        "exact_covered": exact_covered,
        "alias_covered": alias_covered,
        "covered": covered,
        "uncovered": uncovered,
    }

def validate_template_schema(
    feature_templates: Dict[str, Dict[str, Any]],
    timbre_pools: Dict[str, List[str]],
) -> None:
    """Validate the required fields of each acoustic template."""

    required_fields = {
        "onom",
        "pool",
        "mech",
        "analysis",
        "matching",
    }

    errors = []

    for label, template in feature_templates.items():
        missing_fields = required_fields - set(template)

        if missing_fields:
            errors.append(
                f"{label}: missing fields {sorted(missing_fields)}"
            )
            continue

        pool_name = template["pool"]
        if pool_name not in timbre_pools:
            errors.append(
                f"{label}: unknown timbre pool '{pool_name}'"
            )

        if not isinstance(template["onom"], list) or not template["onom"]:
            errors.append(
                f"{label}: 'onom' must be a non-empty list"
            )

    if errors:
        message = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"Invalid template resources:\n{message}")

# ----------------------------------------------------------------------
# Deterministic sample-level randomization
# ----------------------------------------------------------------------
def build_sample_seed(sample_id: str, global_seed: int) -> int:
    """Construct a deterministic random seed for one FSD50K sample."""

    key = f"{global_seed}:{sample_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)

def resolve_feature_template(
    label: str,
    feature_templates: Dict[str, Dict[str, Any]],
    alias_mapping: Dict[str, str],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Resolve an official category to a frozen acoustic template.

    The official label is never replaced in the final target. The alias is
    used only to retrieve an intermediate acoustic-description template.
    """

    official_label = str(label).strip()
    template_key = alias_mapping.get(official_label, official_label)
    template = feature_templates.get(template_key)

    return template_key, template

def jitter_numeric_value(match: re.Match, rng) -> str:
    """Apply a small deterministic perturbation to an acoustic value."""

    value_text = match.group(1)
    unit = match.group(2)

    try:
        value = float(value_text)
    except ValueError:
        return f"{value_text}{unit}"

    jittered = value * rng.uniform(0.9, 1.1)

    if unit.lower() == "hz":
        return f"{int(round(jittered))}{unit}"

    return f"{jittered:.1f}{unit}"

def apply_acoustic_jitter(text: str, rng) -> str:
    """Perturb numeric acoustic descriptors by at most ±10%."""

    pattern = re.compile(
        r"(\d+(?:\.\d+)?)(Hz|kHz|ms|seconds|s)\b",
        flags=re.IGNORECASE,
    )

    return pattern.sub(
        lambda match: jitter_numeric_value(match, rng),
        text,
    )

# ----------------------------------------------------------------------
# A-CoT target construction
# ----------------------------------------------------------------------
def build_multilabel_acot_response(
    labels: Sequence[str],
    sample_id: str,
    feature_templates: Dict[str, Dict[str, Any]],
    timbre_pools: Dict[str, List[str]],
    alias_mapping: Dict[str, str],
    max_detailed_labels: int = 8,
    global_seed: int = 42,
) -> str:
    """
    Construct an A-CoT response for a multi-label FSD50K sample.

    Detailed reasoning blocks are generated for at most
    ``max_detailed_labels`` template-covered categories. All official
    ground-truth categories are retained in the final conclusion,
    including categories without an intermediate template.
    """

    import random

    normalized_labels = [
        str(label).strip()
        for label in labels
        if str(label).strip()
    ]

    if not normalized_labels:
        raise ValueError(
            f"Sample {sample_id} does not contain any valid labels."
        )

    rng = random.Random(
        build_sample_seed(sample_id, global_seed)
    )

    covered_items = []

    for official_label in normalized_labels:
        template_key, template = resolve_feature_template(
            label=official_label,
            feature_templates=feature_templates,
            alias_mapping=alias_mapping,
        )

        if template is None:
            continue

        covered_items.append(
            {
                "official_label": official_label,
                "template_key": template_key,
                "template": template,
            }
        )

    detailed_items = covered_items[:max_detailed_labels]
    reasoning_blocks = []

    for source_index, item in enumerate(detailed_items, start=1):
        official_label = item["official_label"]
        template = item["template"]

        pool_name = template["pool"]
        timbre = rng.choice(timbre_pools[pool_name])
        onomatopoeia = rng.choice(template["onom"])

        acoustic_analysis = apply_acoustic_jitter(
            template["analysis"],
            rng,
        )

        perception = (
            f"Auditory features highlight a {timbre} quality, "
            f"strongly resembling '{onomatopoeia}'."
        )

        mechanism = (
            f"This sound is typically produced by {template['mech']}."
        )

        reasoning_blocks.append(
            f"--- Sound Source {source_index}: {official_label} ---\n"
            f"1. Perceptual Description: {perception}\n"
            f"2. Acoustic Analysis: {acoustic_analysis}\n"
            f"3. Mechanism Reasoning: {mechanism}\n"
            f"4. Matching and Exclusion: {template['matching']}"
        )

    conclusion = (
        "The sound event categories are "
        f"{', '.join(normalized_labels)}."
    )

    if not reasoning_blocks:
        return f"Final Conclusion: {conclusion}"

    return (
        "Reasoning Steps:\n"
        + "\n\n".join(reasoning_blocks)
        + f"\n\nFinal Conclusion: {conclusion}"
    )

# ----------------------------------------------------------------------
# Dataset generation
# ----------------------------------------------------------------------
def parse_label_string(value: Any) -> List[str]:
    """Parse the comma-separated label field in FSD50K metadata."""

    return [
        label.strip()
        for label in str(value).split(",")
        if label.strip()
    ]

def validate_metadata_labels(
    dataframe: pd.DataFrame,
    official_labels: Sequence[str],
) -> None:
    """Ensure that all metadata labels belong to the official vocabulary."""

    official_set = set(official_labels)
    observed_labels = set()

    for value in dataframe["labels"]:
        observed_labels.update(parse_label_string(value))

    unknown_labels = observed_labels - official_set

    if unknown_labels:
        raise ValueError(
            "The metadata contains labels outside the official vocabulary:\n"
            + "\n".join(f"  {label}" for label in sorted(unknown_labels))
        )

def write_jsonl_split(
    dataframe: pd.DataFrame,
    output_path: Path,
    feature_templates: Dict[str, Dict[str, Any]],
    timbre_pools: Dict[str, List[str]],
    alias_mapping: Dict[str, str],
    max_detailed_labels: int,
    global_seed: int,
) -> None:
    """Write one FSD50K split as an A-CoT JSONL file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    instruction = (
        "Identify all sound events and provide a detailed acoustic "
        "analysis for each template-covered source."
    )

    with output_path.open("w", encoding="utf-8") as file:
        for _, row in tqdm(
            dataframe.iterrows(),
            total=len(dataframe),
            desc=f"Writing {output_path.name}",
        ):
            sample_id = str(row["fname"])
            labels = parse_label_string(row["labels"])

            response = build_multilabel_acot_response(
                labels=labels,
                sample_id=sample_id,
                feature_templates=feature_templates,
                timbre_pools=timbre_pools,
                alias_mapping=alias_mapping,
                max_detailed_labels=max_detailed_labels,
                global_seed=global_seed,
            )

            sample = {
                "id": sample_id,
                "query": instruction,
                "response": response,
                "audios": [str(row["audio_path"])],
            }

            file.write(
                json.dumps(sample, ensure_ascii=False) + "\n"
            )

def run_generation(args: argparse.Namespace) -> None:
    """Execute the complete FSD50K A-CoT construction pipeline."""

    dataset_root = args.fsd50k_root.resolve()

    metadata_path = (
        dataset_root
        / "FSD50K.metadata"
        / "FSD50K.ground_truth"
        / "dev.csv"
    )
    vocabulary_path = (
        dataset_root
        / "FSD50K.metadata"
        / "vocabulary.csv"
    )
    audio_directory = dataset_root / "FSD50K.dev_audio"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"FSD50K development metadata not found: {metadata_path}"
        )

    if not audio_directory.is_dir():
        raise FileNotFoundError(
            f"FSD50K development audio directory not found: "
            f"{audio_directory}"
        )

    feature_templates = load_json(args.feature_templates)
    timbre_pools = load_json(args.timbre_pools)
    alias_mapping = load_json(args.alias_mapping)

    validate_template_schema(
        feature_templates=feature_templates,
        timbre_pools=timbre_pools,
    )

    official_labels = load_official_vocabulary(
        vocabulary_path=vocabulary_path,
        label_column=args.vocab_label_column,
    )

    coverage = check_template_coverage(
        official_labels=official_labels,
        feature_templates=feature_templates,
        alias_mapping=alias_mapping,
        expected_coverage=args.expected_coverage,
    )

    print(
        "\nConfirmed supervision composition: "
        f"{len(coverage['covered'])} template-covered categories and "
        f"{len(coverage['uncovered'])} label-only categories."
    )

    dataframe = pd.read_csv(
        metadata_path,
        skipinitialspace=True,
    )

    required_columns = {"fname", "labels", "split"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing_columns)}"
        )

    validate_metadata_labels(dataframe, official_labels)

    dataframe["audio_path"] = dataframe["fname"].apply(
        lambda sample_id: audio_directory / f"{sample_id}.wav"
    )

    initial_count = len(dataframe)
    dataframe = dataframe[
        dataframe["audio_path"].apply(Path.exists)
    ].copy()

    print(
        f"Audio validation completed: {len(dataframe)}/"
        f"{initial_count} files are available."
    )

    train_dataframe = dataframe[
        dataframe["split"] == "train"
    ].copy()

    validation_dataframe = dataframe[
        dataframe["split"] == "val"
    ].copy()

    print(f"Training samples   : {len(train_dataframe)}")
    print(f"Validation samples : {len(validation_dataframe)}")

    write_jsonl_split(
        dataframe=train_dataframe,
        output_path=args.output_dir / "fsd50k_train_acot.jsonl",
        feature_templates=feature_templates,
        timbre_pools=timbre_pools,
        alias_mapping=alias_mapping,
        max_detailed_labels=args.max_detailed_labels,
        global_seed=args.seed,
    )

    write_jsonl_split(
        dataframe=validation_dataframe,
        output_path=args.output_dir / "fsd50k_val_acot.jsonl",
        feature_templates=feature_templates,
        timbre_pools=timbre_pools,
        alias_mapping=alias_mapping,
        max_detailed_labels=args.max_detailed_labels,
        global_seed=args.seed,
    )

    print("FSD50K A-CoT dataset construction completed.")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Construct A-CoT supervision targets for FSD50K."
    )

    parser.add_argument(
        "--fsd50k-root",
        type=Path,
        required=True,
        help="Root directory of the extracted FSD50K dataset.",
    )
    parser.add_argument(
        "--feature-templates",
        type=Path,
        default=Path("resources/fsd50k_features.json"),
        help="Path to the frozen class-level acoustic templates.",
    )
    parser.add_argument(
        "--timbre-pools",
        type=Path,
        default=Path("resources/timbre_pools.json"),
        help="Path to the controlled timbre lexicon.",
    )
    parser.add_argument(
        "--alias-mapping",
        type=Path,
        default=Path("resources/fsd50k_feature_aliases.json"),
        help="Path to the official-label-to-template mapping.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory in which generated JSONL files are stored.",
    )
    parser.add_argument(
        "--vocab-label-column",
        type=int,
        default=1,
        help="Column index containing label names in vocabulary.csv.",
    )
    parser.add_argument(
        "--expected-coverage",
        type=int,
        default=160,
        help="Expected number of template-covered official categories.",
    )
    parser.add_argument(
        "--max-detailed-labels",
        type=int,
        default=4,
        help="Maximum number of detailed A-CoT blocks per sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for deterministic target construction.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    run_generation(parse_args())