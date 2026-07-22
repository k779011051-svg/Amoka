# -*- coding: utf-8 -*-
import os
import gc
import json
import re
import warnings

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    multilabel_confusion_matrix,
)

from swift.llm import (
    SftArguments,
    get_model_tokenizer,
    get_template,
    ModelType,
)
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

import swanlab
from swanlab.integration.huggingface import SwanLabCallback

warnings.filterwarnings("ignore")

# ============================================================
# 0. 实验开关
# ============================================================
FAST_DEV_RUN = os.getenv("FAST_DEV_RUN", "0") == "1"

# ADAPTER_TYPE: amoka / lora
# SUPERVISION_TYPE: cot / direct
ADAPTER_TYPE = os.getenv("ADAPTER_TYPE", "amoka").lower()
SUPERVISION_TYPE = os.getenv("SUPERVISION_TYPE", "cot").lower()

assert ADAPTER_TYPE in ["amoka", "lora"], f"Invalid ADAPTER_TYPE: {ADAPTER_TYPE}"
assert SUPERVISION_TYPE in ["cot", "direct"], f"Invalid SUPERVISION_TYPE: {SUPERVISION_TYPE}"

# AMOKA_VARIANT:
# full      : 完整 A-MokA，audio + mlp + broad q/v attention
# wo_audio  : 去掉 audio_tower / multi_modal_projector 分支
# wo_mlp    : 去掉 gate_proj / up_proj / down_proj 分支
# wo_attn   : 去掉 broad q/v attention routing 分支
# all_r32   : 三分支全部 r=32
AMOKA_VARIANT = os.getenv("AMOKA_VARIANT", "full").lower()

assert AMOKA_VARIANT in [
    "full",
    "wo_audio",
    "wo_mlp",
    "wo_attn",
    "all_r32",
    "all_r16",
], f"Invalid AMOKA_VARIANT: {AMOKA_VARIANT}"

RUN_NAME = f"FSD50K-{ADAPTER_TYPE.upper()}-{SUPERVISION_TYPE.upper()}"
if ADAPTER_TYPE == "amoka" and AMOKA_VARIANT != "full":
    RUN_NAME = f"{RUN_NAME}-{AMOKA_VARIANT.upper()}"

# ============================================================
# 1. 路径配置
# ============================================================
BASE_MODEL_PATH = "/workspace/qwen/Qwen2-Audio-7B-Instruct"
CACHE_ROOT = "/workspace/cache"
OUTPUT_ROOT = "/workspace/swift-FSD/output/FSD50K-BROAD-QV-ABLATION"

if SUPERVISION_TYPE == "cot":
    PROCESSED_TRAIN_DIR = os.path.join(CACHE_ROOT, "fsd50k_processed_train_cache_cot")
    PROCESSED_VAL_DIR = os.path.join(CACHE_ROOT, "fsd50k_processed_val_cache_cot")
else:
    PROCESSED_TRAIN_DIR = os.path.join(CACHE_ROOT, "fsd50k_processed_train_cache_direct")
    PROCESSED_VAL_DIR = os.path.join(CACHE_ROOT, "fsd50k_processed_val_cache_direct")

OUTPUT_DIR = os.path.join(OUTPUT_ROOT, RUN_NAME)

# dummy dataset 是为了满足 SftArguments 的 dataset 字段要求
DUMMY_DATASET_PATH = "/workspace/cache/dummy_dataset.jsonl"
os.makedirs(os.path.dirname(DUMMY_DATASET_PATH), exist_ok=True)
if not os.path.exists(DUMMY_DATASET_PATH):
    with open(DUMMY_DATASET_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps({"query": "dummy", "response": "dummy"}, ensure_ascii=False) + "\n")

# ============================================================
# 2. FSD50K active-200 标签
# ============================================================
FSD50K_LABELS = [
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

assert len(FSD50K_LABELS) == 200, f"Expected 200 FSD50K labels, got {len(FSD50K_LABELS)}"
mlb = MultiLabelBinarizer(classes=FSD50K_LABELS)

tokenizer = None
sft_args = None

# ============================================================
# 3. A-MokA 模块
# ============================================================
class MokALinear(nn.Module):
    def __init__(self, base_layer, r=32, alpha=64, module_name=""):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.layer_type = "attention"
        if "audio_tower" in module_name or "multi_modal_projector" in module_name:
            self.layer_type = "audio"
        elif any(k in module_name for k in ["gate_proj", "up_proj", "down_proj"]):
            self.layer_type = "mlp"

        if self.layer_type == "audio":
            self.A_audio = nn.Linear(in_features, r, bias=False)
            nn.init.kaiming_uniform_(self.A_audio.weight, a=np.sqrt(5))
        elif self.layer_type == "mlp":
            self.A_mlp = nn.Linear(in_features, r, bias=False)
            nn.init.kaiming_uniform_(self.A_mlp.weight, a=np.sqrt(5))
            self.memory_gate = nn.Parameter(torch.ones(1, r) * 0.5)
        else:
            self.A_attn = nn.Linear(in_features, r, bias=False)
            nn.init.kaiming_uniform_(self.A_attn.weight, a=np.sqrt(5))
            self.attn_interaction = nn.Parameter(torch.zeros(1, r))

        self.B = nn.Linear(r, out_features, bias=False)
        nn.init.normal_(self.B.weight, std=0.01)

    def forward(self, x):
        result = self.base_layer(x)

        dtype = self.B.weight.dtype
        x_calc = x.to(dtype)

        if self.layer_type == "audio":
            adapter_offset = self.B(self.A_audio(x_calc))
        elif self.layer_type == "mlp":
            gate = self.memory_gate.to(x.device).to(dtype)
            adapter_offset = self.B(self.A_mlp(x_calc) * (1 + gate))
        else:
            inter = self.attn_interaction.to(x.device).to(dtype)
            adapter_offset = self.B(self.A_attn(x_calc) * (1 + inter))

        return result + (adapter_offset * self.scaling).to(result.dtype)

def inject_moka(model):
    """
    A-MokA 注入，支持分支消融。

    注意：这里恢复 early-best 的 broad q/v 覆盖方式：
        attn_keys = ["self_attn.q_proj", "self_attn.v_proj", "q_proj", "v_proj"]

    AMOKA_VARIANT:
        full      : audio + mlp + broad q/v attention
        wo_audio  : 去掉 audio/projector branch
        wo_mlp    : 去掉 MLP branch
        wo_attn   : 去掉 q/v attention branch
        all_r32   : audio/mlp/attn 全部 r=32
    """
    print(f"Injecting A-MokA adapters. Variant = {AMOKA_VARIANT}")

    audio_keys = ["audio_tower.layers", "multi_modal_projector"]
    mlp_keys = ["gate_proj", "up_proj", "down_proj"]

    # 关键改动：恢复 broad q/v 覆盖
    attn_keys = ["self_attn.q_proj", "self_attn.v_proj", "q_proj", "v_proj"]

    use_audio = AMOKA_VARIANT != "wo_audio"
    use_mlp = AMOKA_VARIANT != "wo_mlp"
    use_attn = AMOKA_VARIANT != "wo_attn"

    if AMOKA_VARIANT == "all_r32":
        audio_config = {"r": 32, "alpha": 64}
        mlp_config = {"r": 32, "alpha": 64}
        attn_config = {"r": 32, "alpha": 64}
    elif AMOKA_VARIANT == "all_r16":
        audio_config = {"r": 16, "alpha": 64}
        mlp_config = {"r": 16, "alpha": 64}
        attn_config = {"r": 16, "alpha": 64}
    else:
        audio_config = {"r": 32, "alpha": 64}
        mlp_config = {"r": 32, "alpha": 64}
        attn_config = {"r": 16, "alpha": 64}

    target_keys = []
    if use_audio:
        target_keys += audio_keys
    if use_mlp:
        target_keys += mlp_keys
    if use_attn:
        target_keys += attn_keys

    memo = set()
    injected_count = {"Audio": 0, "MLP": 0, "Attention": 0}
    injected_names = {"Audio": [], "MLP": [], "Attention": []}

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if not any(key in name for key in target_keys):
            continue

        if module in memo:
            continue

        is_audio_module = any(key in name for key in audio_keys)
        is_mlp_module = any(key in name for key in mlp_keys)
        is_attn_qv_module = any(key in name for key in attn_keys)

        branch_name = None
        config = None

        # 注意：这里的判断顺序非常重要。
        # 1) audio_tower / projector 内部即使包含 q_proj/v_proj，也优先视为 Audio branch。
        # 2) 如果 AMOKA_VARIANT=wo_audio，则这些 audio 模块直接跳过，不会被 broad q/v 误归入 Attention。
        if is_audio_module:
            if not use_audio:
                continue
            branch_name = "Audio"
            config = audio_config

        elif is_mlp_module:
            if not use_mlp:
                continue
            branch_name = "MLP"
            config = mlp_config

        elif is_attn_qv_module:
            if not use_attn:
                continue
            branch_name = "Attention"
            config = attn_config

        else:
            continue

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name)

        new_module = MokALinear(
            module,
            r=config["r"],
            alpha=config["alpha"],
            module_name=name,
        )
        new_module.to(module.weight.device).to(module.weight.dtype)

        setattr(parent, child_name, new_module)
        memo.add(module)

        injected_count[branch_name] += 1
        injected_names[branch_name].append(name)

    print("A-MokA injection finished.")
    print(f"Variant                  : {AMOKA_VARIANT}")
    print(f"Use audio/projector       : {use_audio}")
    print(f"Use MLP                   : {use_mlp}")
    print(f"Use broad q/v attention   : {use_attn}")
    print(f"Audio / projector layers  : {injected_count['Audio']} | r={audio_config['r'] if use_audio else '-'}")
    print(f"MLP layers                : {injected_count['MLP']} | r={mlp_config['r'] if use_mlp else '-'}")
    print(f"Attention q/v layers      : {injected_count['Attention']} | r={attn_config['r'] if use_attn else '-'}")

    for branch in ["Audio", "MLP", "Attention"]:
        print(f"\n[{branch}] injected layer examples:")
        if not injected_names[branch]:
            print("  None")
        else:
            for layer_name in injected_names[branch][:30]:
                print(f"  {layer_name}")
            if len(injected_names[branch]) > 30:
                print(f"  ... total {len(injected_names[branch])} layers")

    return model

# ============================================================
# 4. LoRA baseline 模块
# ============================================================
class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=32, alpha=64, module_name=""):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        result = self.base_layer(x)
        dtype = self.lora_B.weight.dtype
        x_calc = x.to(dtype)
        adapter_offset = self.lora_B(self.lora_A(x_calc))
        return result + (adapter_offset * self.scaling).to(result.dtype)

def inject_lora(model):
    """
    LoRA baseline。
    这里保持你原先的同质化 LoRA 设置：统一 r=32, alpha=64。
    注意：为了不影响你已经完成的 LoRA 实验，这里不改成 broad q/v。
    """
    print("Injecting homogeneous LoRA adapters.")

    lora_config = {"r": 32, "alpha": 64}

    target_keys = [
        "audio_tower.layers",
        "multi_modal_projector",
        "gate_proj",
        "up_proj",
        "down_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    ]

    memo = set()
    injected_count = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(key in name for key in target_keys):
            continue
        if module in memo:
            continue

        # Attention 只处理 language_model 内部 q/v
        if any(key in name for key in ["self_attn.q_proj", "self_attn.v_proj"]):
            if "language_model" not in name:
                continue

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name)

        new_module = LoRALinear(
            module,
            r=lora_config["r"],
            alpha=lora_config["alpha"],
            module_name=name,
        )
        new_module.to(module.weight.device).to(module.weight.dtype)

        setattr(parent, child_name, new_module)
        memo.add(module)
        injected_count += 1

    print("LoRA injection finished.")
    print(f"Injected LoRA linear layers: {injected_count}")
    return model

# ============================================================
# 5. 数据与评估函数
# ============================================================
class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Evaluation cache cleared.")

def extract_multilabels(text):
    if not isinstance(text, str):
        return []

    text_lower = text.lower()

    # 优先解析最后一个 Final Conclusion，减少 CoT 中间文本干扰
    if "final conclusion" in text_lower:
        idx = text_lower.rfind("final conclusion")
        text = text[idx:]
        text_lower = text.lower()

    anchor_str = "categories are"
    if anchor_str not in text_lower:
        anchor_str = "category is"
    if anchor_str not in text_lower:
        return []

    labels_part = text_lower.split(anchor_str, 1)[-1]
    labels_part = labels_part.replace("(additional sounds detected:", ",")
    labels_part = labels_part.split("\n", 1)[0]
    labels_part = labels_part.strip()
    labels_part = re.sub(r"\.\s*$", "", labels_part)

    raw_labels = [l.strip() for l in labels_part.split(",") if l.strip()]
    found_labels = []
    label_map = {sl.lower(): sl for sl in FSD50K_LABELS}

    for rl in raw_labels:
        rl_standardized = rl.strip(" .。,:;，；\"'")
        rl_standardized = rl_standardized.replace(" ", "_")

        if rl_standardized.endswith(")") and rl_standardized.lower() not in label_map:
            candidate = rl_standardized[:-1]
            if candidate.lower() in label_map:
                rl_standardized = candidate

        key = rl_standardized.lower()
        if key in label_map:
            found_labels.append(label_map[key])

    return list(set(found_labels))

def compute_metrics(eval_preds):
    global tokenizer, sft_args

    preds, labels = eval_preds

    preds_decode = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    labels_decode = tokenizer.batch_decode(labels, skip_special_tokens=True)

    y_pred = [extract_multilabels(x) for x in preds_decode]
    y_true = [extract_multilabels(x) for x in labels_decode]

    valid_indices = [i for i, labels_i in enumerate(y_true) if labels_i]
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

    y_true = [y_true[i] for i in valid_indices]
    y_pred = [y_pred[i] for i in valid_indices]

    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    f1_macro = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    precision_macro = precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    recall_macro = recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)

    f1_micro = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    precision_micro = precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    recall_micro = recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)

    f1_samples = f1_score(y_true_bin, y_pred_bin, average="samples", zero_division=0)

    try:
        mcm = multilabel_confusion_matrix(y_true_bin, y_pred_bin)
        confusion_dict = {}

        for i, label_name in enumerate(mlb.classes_):
            tn, fp, fn, tp = mcm[i].ravel()
            if tp + fp + fn > 0:
                confusion_dict[label_name] = {
                    "TP": int(tp),
                    "FP": int(fp),
                    "TN": int(tn),
                    "FN": int(fn),
                }

        os.makedirs(sft_args.output_dir, exist_ok=True)
        cm_path = os.path.join(sft_args.output_dir, "multilabel_confusion_matrix.json")
        with open(cm_path, "w", encoding="utf-8") as f:
            json.dump(confusion_dict, f, ensure_ascii=False, indent=4)

        md_table = "| Category | TP | FP | TN | FN |\n|---|---:|---:|---:|---:|\n"
        for label_name, values in confusion_dict.items():
            md_table += (
                f"| {label_name} | {values['TP']} | {values['FP']} | "
                f"{values['TN']} | {values['FN']} |\n"
            )
        swanlab.log({"Multi-Label Confusion Matrix": swanlab.Text(md_table)})
    except Exception as exc:
        print(f"Failed to save/log confusion matrix: {exc}")

    return {
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_micro": f1_micro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_samples": f1_samples,
    }

def to_torch(examples):
    for key in list(examples.keys()):
        if key == "audio_info":
            continue

        values = []
        for item in examples[key]:
            if item is None:
                values.append(None)
            elif isinstance(item, torch.Tensor):
                values.append(item)
            elif isinstance(item, (list, np.ndarray)):
                values.append(torch.tensor(item))
            else:
                values.append(torch.tensor(item))

        examples[key] = values

    return examples

def load_cached_dataset(train_cache_dir, val_cache_dir):
    if not os.path.isdir(train_cache_dir):
        raise FileNotFoundError(f"Training cache not found: {train_cache_dir}")
    if not os.path.isdir(val_cache_dir):
        raise FileNotFoundError(f"Validation cache not found: {val_cache_dir}")

    train_dataset = load_from_disk(train_cache_dir)
    val_dataset = load_from_disk(val_cache_dir)

    required_columns = {
        "input_ids",
        "labels",
        "input_features",
        "feature_attention_mask",
    }

    for name, dataset in [("train", train_dataset), ("validation", val_dataset)]:
        missing = required_columns - set(dataset.column_names)
        if missing:
            raise ValueError(f"{name} cache is missing columns: {sorted(missing)}")

    print(f"Loaded train cache: {train_cache_dir}, samples={len(train_dataset)}")
    print(f"Loaded val cache  : {val_cache_dir}, samples={len(val_dataset)}")
    print(f"Cache columns     : {train_dataset.column_names}")

    return train_dataset, val_dataset

def print_trainable_summary(model, adapter_type):
    total_count = 0
    trainable_count = 0

    if adapter_type == "amoka":
        branch_stats = {
            "A_audio": 0,
            "A_mlp": 0,
            "A_attn": 0,
            "B": 0,
            "memory_gate": 0,
            "attn_interaction": 0,
        }
    else:
        branch_stats = {
            "lora_A": 0,
            "lora_B": 0,
        }

    for name, param in model.named_parameters():
        total_count += param.numel()
        if param.requires_grad:
            trainable_count += param.numel()
            for key in branch_stats:
                if key in name:
                    branch_stats[key] += param.numel()

    if adapter_type == "amoka":
        print(f"\n{adapter_type.upper()} trainable parameter summary | AMOKA_VARIANT={AMOKA_VARIANT}")
    else:
        print(f"\n{adapter_type.upper()} trainable parameter summary")

    print(f"Total parameters     : {total_count:,}")
    print(f"Trainable parameters : {trainable_count:,}")
    print(f"Trainable ratio      : {100 * trainable_count / total_count:.4f}%")
    for key, value in branch_stats.items():
        print(f"{key:20s}: {value:,}")
    print()

# ============================================================
# 6. 主训练逻辑
# ============================================================
def main():
    global tokenizer, sft_args

    print("\n================ Experiment Configuration ================")
    print(f"RUN_NAME          : {RUN_NAME}")
    print(f"ADAPTER_TYPE      : {ADAPTER_TYPE}")
    print(f"SUPERVISION_TYPE  : {SUPERVISION_TYPE}")
    print(f"AMOKA_VARIANT     : {AMOKA_VARIANT}")
    print(f"BASE_MODEL_PATH   : {BASE_MODEL_PATH}")
    print(f"TRAIN_CACHE       : {PROCESSED_TRAIN_DIR}")
    print(f"VAL_CACHE         : {PROCESSED_VAL_DIR}")
    print(f"OUTPUT_DIR        : {OUTPUT_DIR}")
    print(f"FAST_DEV_RUN      : {FAST_DEV_RUN}")
    print("==========================================================\n")

    swanlab_api_key = os.getenv("SWANLAB_API_KEY")
    if swanlab_api_key:
        swanlab.login(api_key=swanlab_api_key)

    sft_args = SftArguments(
        model_type=ModelType.qwen2_audio_7b_instruct,
        model_id_or_path=BASE_MODEL_PATH,
        output_dir=OUTPUT_DIR,
        dataset=[DUMMY_DATASET_PATH],
        val_dataset=[DUMMY_DATASET_PATH],
        dtype="bf16",
        max_length=2048,
        template_type="qwen2-audio",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
    )

    swanlab.init(
        project="Qwen2-Audio-FSD50K-SCI-Ablation",
        experiment_name=RUN_NAME,
        config={
            **sft_args.__dict__,
            "adapter_type": ADAPTER_TYPE,
            "supervision_type": SUPERVISION_TYPE,
            "amoka_variant": AMOKA_VARIANT,
            "processed_train_dir": PROCESSED_TRAIN_DIR,
            "processed_val_dir": PROCESSED_VAL_DIR,
        },
    )

    # ------------------------------------------------------------
    # 6.1 加载模型
    # ------------------------------------------------------------
    model, tokenizer = get_model_tokenizer(
        sft_args.model_type,
        sft_args.torch_dtype,
        model_id_or_path=sft_args.model_id_or_path,
        model_kwargs={
            "device_map": {"": 0}
        },
    )

    if hasattr(model, "hf_device_map"):
        print("Model hf_device_map:", model.hf_device_map)
        offloaded_modules = {
            name: device
            for name, device in model.hf_device_map.items()
            if str(device) in ["cpu", "disk"]
        }
        if offloaded_modules:
            raise RuntimeError(
                "Model has modules offloaded to CPU/disk, which is incompatible with Trainer training. "
                f"Offloaded modules: {offloaded_modules}"
            )

    # ------------------------------------------------------------
    # 6.2 注入 adapter
    # ------------------------------------------------------------
    if ADAPTER_TYPE == "amoka":
        model = inject_moka(model)
    elif ADAPTER_TYPE == "lora":
        model = inject_lora(model)
    else:
        raise ValueError(f"Unsupported ADAPTER_TYPE: {ADAPTER_TYPE}")

    # 冻结基座，只训练 adapter
    model.requires_grad_(False)

    if ADAPTER_TYPE == "amoka":
        trainable_keys = [
            "A_audio",
            "A_mlp",
            "A_attn",
            "B",
            "memory_gate",
            "attn_interaction",
        ]
    else:
        trainable_keys = [
            "lora_A",
            "lora_B",
        ]

    for name, param in model.named_parameters():
        if any(key in name for key in trainable_keys):
            param.requires_grad = True

    print_trainable_summary(model, ADAPTER_TYPE)

    model.enable_input_require_grads()
    model.config.use_cache = False

    # ------------------------------------------------------------
    # 6.3 模板和数据
    # ------------------------------------------------------------
    template = get_template(
        sft_args.template_type,
        tokenizer,
        sft_args.system,
        sft_args.max_length,
    )

    train_dataset, val_dataset = load_cached_dataset(
        PROCESSED_TRAIN_DIR,
        PROCESSED_VAL_DIR,
    )

    if FAST_DEV_RUN:
        train_n = min(4, len(train_dataset))
        val_n = min(2, len(val_dataset))
        train_dataset = train_dataset.select(range(train_n))
        val_dataset = val_dataset.select(range(val_n))
        print("FAST_DEV_RUN is enabled.")
        print(f"Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    train_dataset.set_transform(to_torch)
    val_dataset.set_transform(to_torch)

    # ------------------------------------------------------------
    # 6.4 generation 设置
    # ------------------------------------------------------------
    model.generation_config.max_new_tokens = 128 if FAST_DEV_RUN else 600
    model.generation_config.max_length = 2048 if FAST_DEV_RUN else 4096

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------
    # 6.5 Trainer
    # ------------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR if not FAST_DEV_RUN else os.path.join(OUTPUT_DIR, "fast_dev_run"),

        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        num_train_epochs=1 if FAST_DEV_RUN else 5,
        max_steps=2 if FAST_DEV_RUN else -1,

        per_device_train_batch_size=1 if FAST_DEV_RUN else 2,
        gradient_accumulation_steps=1 if FAST_DEV_RUN else 8,
        per_device_eval_batch_size=1 if FAST_DEV_RUN else 2,

        dataloader_num_workers=0 if FAST_DEV_RUN else 4,

        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=False,

        predict_with_generate=True,
        generation_max_length=128 if FAST_DEV_RUN else 600,

        eval_accumulation_steps=1,
        eval_strategy="steps" if FAST_DEV_RUN else "epoch",
        eval_steps=1 if FAST_DEV_RUN else None,

        save_strategy="steps" if FAST_DEV_RUN else "epoch",
        save_steps=1 if FAST_DEV_RUN else None,
        save_total_limit=1 if FAST_DEV_RUN else None,

        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,

        logging_steps=1,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=template.data_collator,
        compute_metrics=compute_metrics,
        callbacks=[SwanLabCallback(), ClearCacheCallback()],
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

    print(f"Starting FSD50K SFT ablation: {RUN_NAME}")
    trainer.train()

    swanlab.finish()

if __name__ == "__main__":
    main()