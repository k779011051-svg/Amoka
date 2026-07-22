# -*- coding: utf-8 -*-
import argparse
import csv
import json
import re
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# =========================
# 1. 默认配置
# =========================
RESULT_PATH = "F:\dataset\FSD50K\\amoka\inference_results_FSD50K-AMOKA-COT-ALL_R16.jsonl"

# custom: 使用下方 CUSTOM_FSD50K_LABELS
# official: 使用 VOCAB_PATH 指向的官方 vocabulary.csv
LABEL_SOURCE = "custom"

VOCAB_PATH = "F:/dataset/FSD50K/FSD50K.metadata/vocabulary.csv"

# 官方 vocabulary.csv 中标签列的索引。
# 如果你的 csv 是: index,label,mids，则用 1。
# 如果你的 csv 是: label,mids,...，则改成 0。
VOCAB_LABEL_COLUMN = 1

SEMANTIC_THRESHOLD = 0.5

# prediction_raw: 开放式生成结果，用于语义匹配
# prediction_strict: 已经被解析进标签空间的预测
SEMANTIC_PRED_FIELD = "prediction_raw"

# =========================
# 2. 你的 FSD50K active-200 标签
# =========================
CUSTOM_FSD50K_LABELS = [
    "Accelerating_and_revving_and_vroom",
    "Accordion",
    "Acoustic_guitar",
    "Aircraft",
    "Alarm",
    "Animal",
    "Applause",
    "Bark",
    "Bass_drum",
    "Bass_guitar",
    "Bathtub_(filling_or_washing)",
    "Bell",
    "Bicycle",
    "Bicycle_bell",
    "Bird",
    "Bird_vocalization_and_bird_call_and_bird_song",
    "Boat_and_Water_vehicle",
    "Boiling",
    "Boom",
    "Bowed_string_instrument",
    "Brass_instrument",
    "Breathing",
    "Burping_and_eructation",
    "Bus",
    "Buzz",
    "Camera",
    "Car",
    "Car_passing_by",
    "Cat",
    "Chatter",
    "Cheering",
    "Chewing_and_mastication",
    "Chicken_and_rooster",
    "Child_speech_and_kid_speaking",
    "Chime",
    "Chink_and_clink",
    "Chirp_and_tweet",
    "Chuckle_and_chortle",
    "Church_bell",
    "Clapping",
    "Clock",
    "Coin_(dropping)",
    "Computer_keyboard",
    "Conversation",
    "Cough",
    "Cowbell",
    "Crack",
    "Crackle",
    "Crash_cymbal",
    "Cricket",
    "Crow",
    "Crowd",
    "Crumpling_and_crinkling",
    "Crushing",
    "Crying_and_sobbing",
    "Cupboard_open_or_close",
    "Cutlery_and_silverware",
    "Cymbal",
    "Dishes_and_pots_and_pans",
    "Dog",
    "Domestic_animals_and_pets",
    "Domestic_sounds_and_home_sounds",
    "Door",
    "Doorbell",
    "Drawer_open_or_close",
    "Drill",
    "Drip",
    "Drum",
    "Drum_kit",
    "Electric_guitar",
    "Engine",
    "Engine_starting",
    "Explosion",
    "Fart",
    "Female_singing",
    "Female_speech_and_woman_speaking",
    "Fill_(with_liquid)",
    "Finger_snapping",
    "Fire",
    "Fireworks",
    "Fixed-wing_aircraft_and_airplane",
    "Fowl",
    "Frog",
    "Frying_(food)",
    "Gasp",
    "Giggle",
    "Glass",
    "Glockenspiel",
    "Gong",
    "Growling",
    "Guitar",
    "Gull_and_seagull",
    "Gunshot_and_gunfire",
    "Gurgling",
    "Hammer",
    "Hands",
    "Harmonica",
    "Harp",
    "Hi-hat",
    "Hiss",
    "Human_group_actions",
    "Human_voice",
    "Idling",
    "Insect",
    "Keyboard_(musical)",
    "Keys_jangling",
    "Knock",
    "Laughter",
    "Liquid",
    "Livestock_and_farm_animals_and_working_animals",
    "Male_singing",
    "Male_speech_and_man_speaking",
    "Mallet_percussion",
    "Marimba_and_xylophone",
    "Mechanical_fan",
    "Mechanisms",
    "Meow",
    "Microwave_oven",
    "Motor_vehicle_(road)",
    "Motorcycle",
    "Music",
    "Musical_instrument",
    "Ocean",
    "Organ",
    "Packing_tape_and_duct_tape",
    "Percussion",
    "Piano",
    "Plucked_string_instrument",
    "Pour",
    "Power_tool",
    "Printer",
    "Purr",
    "Race_car_and_auto_racing",
    "Rail_transport",
    "Rain",
    "Raindrop",
    "Ratchet_and_pawl",
    "Rattle",
    "Rattle_(instrument)",
    "Respiratory_sounds",
    "Ringtone",
    "Run",
    "Sawing",
    "Scissors",
    "Scratching_(performance_technique)",
    "Screaming",
    "Screech",
    "Shatter",
    "Shout",
    "Sigh",
    "Singing",
    "Sink_(filling_or_washing)",
    "Siren",
    "Skateboard",
    "Slam",
    "Sliding_door",
    "Snare_drum",
    "Sneeze",
    "Speech",
    "Speech_synthesizer",
    "Splash_and_splatter",
    "Squeak",
    "Stream",
    "Strum",
    "Subway_and_metro_and_underground",
    "Tabla",
    "Tambourine",
    "Tap",
    "Tearing",
    "Telephone",
    "Thump_and_thud",
    "Thunder",
    "Thunderstorm",
    "Tick",
    "Tick-tock",
    "Toilet_flush",
    "Tools",
    "Traffic_noise_and_roadway_noise",
    "Train",
    "Trickle_and_dribble",
    "Truck",
    "Trumpet",
    "Typewriter",
    "Typing",
    "Vehicle",
    "Vehicle_horn_and_car_horn_and_honking",
    "Walk_and_footsteps",
    "Water",
    "Water_tap_and_faucet",
    "Waves_and_surf",
    "Whispering",
    "Whoosh_and_swoosh_and_swish",
    "Wild_animals",
    "Wind",
    "Wind_chime",
    "Wind_instrument_and_woodwind_instrument",
    "Wood",
    "Writing",
    "Yell",
    "Zipper_(clothing)",
]

# =========================
# 3. 标签读取与规范化
# =========================
def normalize_key(label):
    if label is None:
        return ""

    label = str(label).strip().lower()
    label = label.replace("_", " ")
    label = label.replace("-", " ")
    label = label.replace("/", " ")
    label = label.replace("&", " and ")
    label = label.replace(",", " and ")
    label = re.sub(r"\s+", " ", label)
    return label.strip()

def normalize_for_embedding(label):
    return normalize_key(label)

def load_official_vocabulary(vocab_path, label_column=1):
    vocab_path = Path(vocab_path)

    if not vocab_path.exists():
        raise FileNotFoundError(f"Official vocabulary.csv not found: {vocab_path}")

    labels = []

    with open(vocab_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        for row_idx, row in enumerate(reader):
            if not row:
                continue

            row = [x.strip() for x in row]
            if label_column >= len(row):
                continue

            candidate = row[label_column].strip().strip("\"'")

            if not candidate:
                continue

            # 跳过常见表头
            if row_idx == 0 and normalize_key(candidate) in {
                "label",
                "labels",
                "display name",
                "display_name",
                "name",
                "category",
            }:
                continue

            labels.append(candidate)

    labels = list(dict.fromkeys(labels))

    if len(labels) != 200:
        raise ValueError(
            f"Official vocabulary label count should be 200, got {len(labels)}. "
            f"Please check VOCAB_PATH={vocab_path} and VOCAB_LABEL_COLUMN={label_column}."
        )

    return labels

def load_label_space(label_source, vocab_path=None, vocab_label_column=1):
    label_source = str(label_source).lower().strip()

    if label_source == "custom":
        labels = list(CUSTOM_FSD50K_LABELS)
    elif label_source == "official":
        labels = load_official_vocabulary(vocab_path, vocab_label_column)
    else:
        raise ValueError(f"Unsupported label_source: {label_source}. Use custom or official.")

    if len(labels) != 200:
        raise ValueError(f"Label space must contain 200 labels, got {len(labels)}.")

    return labels

def build_label_canonicalizer(label_space):
    """
    将不同写法映射回当前评估标签空间中的标准标签。
    例如：
        Electric_guitar
        electric guitar
        electric-guitar
    都映射到 label_space 中对应的 canonical label。
    """
    mapping = {}

    for label in label_space:
        variants = {
            label,
            label.replace("_", " "),
            label.replace("_and_", ", "),
            label.replace("_and_", " and "),
            label.replace("_", "-"),
        }

        for v in variants:
            mapping[normalize_key(v)] = label

    return mapping

def canonicalize_label(label, canonical_map):
    key = normalize_key(label)
    if not key:
        return None
    return canonical_map.get(key, None)

def canonicalize_label_list(labels, canonical_map, keep_unmapped=False):
    out = []

    for x in labels:
        mapped = canonicalize_label(x, canonical_map)

        if mapped is not None:
            out.append(mapped)
        elif keep_unmapped:
            out.append(x)

    return deduplicate_labels(out)

def deduplicate_labels(labels):
    seen = set()
    out = []

    for x in labels:
        key = normalize_key(x)

        if key and key not in seen:
            out.append(x)
            seen.add(key)

    return out

# =========================
# 4. 语义匹配
# =========================
def semantic_score(pred_label, true_label, sbert_model):
    p = normalize_for_embedding(pred_label)
    t = normalize_for_embedding(true_label)

    if not p or not t:
        return 0.0

    if p == t:
        return 1.0

    if len(p) >= 3 and p in t:
        return 1.0

    if len(t) >= 3 and t in p:
        return 1.0

    emb_p = sbert_model.encode(p, convert_to_tensor=True)
    emb_t = sbert_model.encode(t, convert_to_tensor=True)
    return float(util.cos_sim(emb_p, emb_t).item())

def map_pred_to_label_space(
    pred_label,
    label_space,
    label_embs,
    sbert_model,
    threshold=0.50,
):
    p = normalize_for_embedding(pred_label)

    if not p:
        return None, 0.0

    emb_p = sbert_model.encode(p, convert_to_tensor=True)
    scores = util.cos_sim(emb_p, label_embs)[0]

    best_idx = int(torch.argmax(scores).item())
    best_score = float(scores[best_idx].item())

    if best_score < threshold:
        return None, best_score

    return label_space[best_idx], best_score

def one_to_one_semantic_projection(
    y_pred_raw,
    y_true,
    label_space,
    label_embs,
    sbert_model,
    threshold=0.50,
):
    y_true = deduplicate_labels(y_true)
    y_pred_raw = deduplicate_labels(y_pred_raw)

    candidate_pairs = []

    for pi, pred_label in enumerate(y_pred_raw):
        for ti, true_label in enumerate(y_true):
            score = semantic_score(pred_label, true_label, sbert_model)

            if score >= threshold:
                candidate_pairs.append((score, pi, ti))

    candidate_pairs.sort(reverse=True, key=lambda x: x[0])

    matched_pred = set()
    matched_true = set()
    projected_pred_labels = []

    for score, pi, ti in candidate_pairs:
        if pi in matched_pred or ti in matched_true:
            continue

        matched_pred.add(pi)
        matched_true.add(ti)
        projected_pred_labels.append(y_true[ti])

    for pi, pred_label in enumerate(y_pred_raw):
        if pi in matched_pred:
            continue

        mapped_label, _ = map_pred_to_label_space(
            pred_label=pred_label,
            label_space=label_space,
            label_embs=label_embs,
            sbert_model=sbert_model,
            threshold=threshold,
        )

        if mapped_label is not None:
            projected_pred_labels.append(mapped_label)

    return list(set(projected_pred_labels))

# =========================
# 5. 指标计算
# =========================
def compute_multilabel_metrics(yt_bin, yp_bin, active_mask):
    f1_per_class = f1_score(yt_bin, yp_bin, average=None, zero_division=0)
    p_per_class = precision_score(yt_bin, yp_bin, average=None, zero_division=0)
    r_per_class = recall_score(yt_bin, yp_bin, average=None, zero_division=0)

    macro_f1 = float(np.mean(f1_per_class[active_mask])) if np.any(active_mask) else 0.0
    macro_p = float(np.mean(p_per_class[active_mask])) if np.any(active_mask) else 0.0
    macro_r = float(np.mean(r_per_class[active_mask])) if np.any(active_mask) else 0.0

    samples_f1 = f1_score(yt_bin, yp_bin, average="samples", zero_division=0)

    micro_f1 = f1_score(yt_bin, yp_bin, average="micro", zero_division=0)
    micro_p = precision_score(yt_bin, yp_bin, average="micro", zero_division=0)
    micro_r = recall_score(yt_bin, yp_bin, average="micro", zero_division=0)

    return {
        "samples_f1": samples_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "f1_per_class": f1_per_class,
        "precision_per_class": p_per_class,
        "recall_per_class": r_per_class,
    }

def print_metric_block(title, metrics):
    print(title)
    print(f"  - Samples F1       : {metrics['samples_f1'] * 100:.2f}%")
    print(f"  - Macro Precision  : {metrics['macro_precision'] * 100:.2f}%")
    print(f"  - Macro Recall     : {metrics['macro_recall'] * 100:.2f}%")
    print(f"  - Macro F1         : {metrics['macro_f1'] * 100:.2f}%")
    print(f"  - Micro Precision  : {metrics['micro_precision'] * 100:.2f}%")
    print(f"  - Micro Recall     : {metrics['micro_recall'] * 100:.2f}%")
    print(f"  - Micro F1         : {metrics['micro_f1'] * 100:.2f}%\n")

# =========================
# 6. 主评估逻辑
# =========================
def evaluate(args):
    label_space = load_label_space(
        label_source=args.label_source,
        vocab_path=args.vocab_path,
        vocab_label_column=args.vocab_label_column,
    )

    canonical_map = build_label_canonicalizer(label_space)

    print("Loading Sentence-BERT model...")
    sbert_model = SentenceTransformer(args.sbert_model)

    label_texts = [normalize_for_embedding(x) for x in label_space]
    label_embs = sbert_model.encode(label_texts, convert_to_tensor=True)

    y_true_all = []
    y_pred_strict_all = []
    y_pred_semantic_all = []

    valid_samples = 0
    corrupted_samples = 0
    empty_gt_samples = 0

    result_path = Path(args.result_path)

    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    with open(result_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Evaluating"):
        if not line.strip():
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            corrupted_samples += 1
            continue

        gt_raw = data.get("ground_truth_strict", [])
        pred_strict_raw = data.get("prediction_strict", [])
        pred_semantic_raw = data.get(args.semantic_pred_field, [])

        gt = canonicalize_label_list(gt_raw, canonical_map, keep_unmapped=False)

        if not gt:
            empty_gt_samples += 1
            continue

        pred_strict = canonicalize_label_list(
            pred_strict_raw,
            canonical_map,
            keep_unmapped=False,
        )

        # semantic raw 要保留未映射标签，因为后续需要语义投影。
        pred_semantic_input = canonicalize_label_list(
            pred_semantic_raw,
            canonical_map,
            keep_unmapped=True,
        )

        semantic_projected = one_to_one_semantic_projection(
            y_pred_raw=pred_semantic_input,
            y_true=gt,
            label_space=label_space,
            label_embs=label_embs,
            sbert_model=sbert_model,
            threshold=args.semantic_threshold,
        )

        valid_samples += 1

        y_true_all.append(gt)
        y_pred_strict_all.append(pred_strict)
        y_pred_semantic_all.append(semantic_projected)

    print(f"\nValid evaluated samples : {valid_samples}")
    print(f"Corrupted samples       : {corrupted_samples}")
    print(f"Empty/unmapped GT       : {empty_gt_samples}")
    print(f"Result file             : {result_path}")
    print(f"Label source            : {args.label_source}")
    print(f"Label space size        : {len(label_space)}")

    if args.label_source == "official":
        print(f"Official vocabulary     : {args.vocab_path}")
        print(f"Vocab label column      : {args.vocab_label_column}")

    print(f"Semantic pred field     : {args.semantic_pred_field}")
    print(f"Semantic threshold      : {args.semantic_threshold}\n")

    if valid_samples == 0:
        raise RuntimeError("No valid samples were evaluated. Please check ground_truth_strict and label space.")

    mlb = MultiLabelBinarizer(classes=label_space)

    yt_bin = mlb.fit_transform(y_true_all)
    yp_strict_bin = mlb.transform(y_pred_strict_all)
    yp_semantic_bin = mlb.transform(y_pred_semantic_all)

    active_mask = np.sum(yt_bin, axis=0) > 0
    active_classes = np.array(label_space)[active_mask]

    strict_metrics = compute_multilabel_metrics(
        yt_bin=yt_bin,
        yp_bin=yp_strict_bin,
        active_mask=active_mask,
    )

    semantic_metrics = compute_multilabel_metrics(
        yt_bin=yt_bin,
        yp_bin=yp_semantic_bin,
        active_mask=active_mask,
    )

    label_set = set(label_space)

    pred_strict_active = len(set(
        label for labels in y_pred_strict_all
        for label in labels
        if label in label_set
    ))

    pred_semantic_active = len(set(
        label for labels in y_pred_semantic_all
        for label in labels
        if label in label_set
    ))

    print("=" * 72)
    print("FSD50K Multi-Label Evaluation Results".center(72))
    print("=" * 72)
    print(f"Total evaluated samples     : {valid_samples}")
    print(f"Active GT categories        : {len(active_classes)} / {len(label_space)}")
    print(f"Strict pred categories      : {pred_strict_active} / {len(label_space)}")
    print(f"Semantic pred categories    : {pred_semantic_active} / {len(label_space)}")
    print("=" * 72 + "\n")

    print_metric_block("[1] Strict Exact Match", strict_metrics)
    print_metric_block("[2] SBERT Semantic Soft-Match with One-to-One Constraint", semantic_metrics)

    if args.save_summary:
        summary = {
            "result_path": str(result_path),
            "label_source": args.label_source,
            "vocab_path": str(args.vocab_path) if args.vocab_path else None,
            "vocab_label_column": args.vocab_label_column,
            "label_space_size": len(label_space),
            "valid_samples": valid_samples,
            "corrupted_samples": corrupted_samples,
            "empty_or_unmapped_gt_samples": empty_gt_samples,
            "active_gt_categories": int(len(active_classes)),
            "strict_pred_categories": int(pred_strict_active),
            "semantic_pred_categories": int(pred_semantic_active),
            "semantic_pred_field": args.semantic_pred_field,
            "semantic_threshold": args.semantic_threshold,
            "strict": {
                k: float(v)
                for k, v in strict_metrics.items()
                if not k.endswith("_per_class")
            },
            "semantic": {
                k: float(v)
                for k, v in semantic_metrics.items()
                if not k.endswith("_per_class")
            },
        }

        save_path = Path(args.save_summary)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"Summary saved to: {save_path}")

# =========================
# 7. 参数入口
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result_path",
        type=str,
        default=RESULT_PATH,
        help="Inference result jsonl path.",
    )

    parser.add_argument(
        "--label_source",
        type=str,
        default=LABEL_SOURCE,
        choices=["custom", "official"],
        help="custom uses built-in FSD labels; official uses vocabulary.csv.",
    )

    parser.add_argument(
        "--vocab_path",
        type=str,
        default=VOCAB_PATH,
        help="Official FSD50K vocabulary.csv path. Only used when label_source=official.",
    )

    parser.add_argument(
        "--vocab_label_column",
        type=int,
        default=VOCAB_LABEL_COLUMN,
        help="Column index of label names in official vocabulary.csv.",
    )

    parser.add_argument(
        "--semantic_threshold",
        type=float,
        default=SEMANTIC_THRESHOLD,
        help="Threshold for SBERT semantic one-to-one matching.",
    )

    parser.add_argument(
        "--semantic_pred_field",
        type=str,
        default=SEMANTIC_PRED_FIELD,
        choices=["prediction_raw", "prediction_strict"],
        help="Prediction field used for semantic evaluation.",
    )

    parser.add_argument(
        "--sbert_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-BERT model name or local path.",
    )

    parser.add_argument(
        "--save_summary",
        type=str,
        default="",
        help="Optional json path to save metric summary.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    evaluate(parse_args())