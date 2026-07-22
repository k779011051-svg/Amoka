# -*- coding: utf-8 -*-
import os
import json
import re
import gc
import glob
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm
import transformers
from safetensors.torch import load_file
from datasets import load_from_disk, concatenate_datasets
from swift.llm import ModelType, get_model_tokenizer, get_template

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# ============================================================
# 0. 实验开关：与训练脚本保持一致
# ============================================================
# ADAPTER_TYPE: amoka / lora
# SUPERVISION_TYPE: cot / direct
# 注意：
#   SUPERVISION_TYPE 表示 checkpoint 的训练监督格式；
#   TEST_DATA_PATH 可以是 cot cache/direct cache/jsonl，不强制和 SUPERVISION_TYPE 一致。
ADAPTER_TYPE = os.getenv("ADAPTER_TYPE", "amoka").lower()
SUPERVISION_TYPE = os.getenv("SUPERVISION_TYPE", "cot").lower()

assert ADAPTER_TYPE in ["amoka", "lora"], f"Invalid ADAPTER_TYPE: {ADAPTER_TYPE}"
assert SUPERVISION_TYPE in ["cot", "direct"], f"Invalid SUPERVISION_TYPE: {SUPERVISION_TYPE}"

# A-MokA 三分支消融开关
# full      : audio + mlp + broad q/v attention
# wo_audio  : 去掉 audio_tower / multi_modal_projector branch
# wo_mlp    : 去掉 gate_proj / up_proj / down_proj branch
# wo_attn   : 去掉 broad q/v attention branch
# all_r32   : audio/mlp/attn 全部 r=32
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
BASE_MODEL_PATH = os.getenv(
    "BASE_MODEL_PATH",
    "/workspace/qwen/Qwen2-Audio-7B-Instruct",
)

# 这里需要和三分支消融训练的 OUTPUT_ROOT 对齐
OUTPUT_ROOT = os.getenv(
    "OUTPUT_ROOT",
    "/workspace/swift-FSD/output/FSD50K-BROAD-QV-ABLATION",
)

# 可手动指定 checkpoint；不指定则自动从 OUTPUT_ROOT/RUN_NAME 下寻找最新 checkpoint
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "").strip()

# 支持两种输入：
# 1) JSONL 文件；
# 2) HuggingFace save_to_disk cache 目录，例如：
#    /workspace/cache/fsd50k_processed_val_cache_direct
#    /workspace/cache/fsd50k_processed_val_cache_cot
TEST_DATA_PATH = os.getenv(
    "TEST_DATA_PATH",
    os.getenv(
        "TEST_JSONL_PATH",
        "/workspace/cache/fsd50k_processed_val_cache_direct",
    ),
)

OUTPUT_RESULT_PATH = os.getenv("OUTPUT_RESULT_PATH", "").strip()
if not OUTPUT_RESULT_PATH:
    OUTPUT_RESULT_PATH = os.path.join(
        OUTPUT_ROOT,
        RUN_NAME,
        f"inference_results_{RUN_NAME}.jsonl",
    )

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "3"))

_max_samples = os.getenv("MAX_INFER_SAMPLES", "").strip()
MAX_INFER_SAMPLES = int(_max_samples) if _max_samples else None

# 生成长度和 response prefix 由 checkpoint 的训练监督格式决定
if SUPERVISION_TYPE == "direct":
    DEFAULT_MAX_NEW_TOKENS = 128
    DEFAULT_USE_RESPONSE_PREFIX = False
else:
    DEFAULT_MAX_NEW_TOKENS = 800
    DEFAULT_USE_RESPONSE_PREFIX = True

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS)))

USE_RESPONSE_PREFIX = os.getenv(
    "USE_RESPONSE_PREFIX",
    "1" if DEFAULT_USE_RESPONSE_PREFIX else "0",
) == "1"

RESPONSE_PREFIX = os.getenv("RESPONSE_PREFIX", "Reasoning Steps:\n")

# ============================================================
# 2. Active-200 标签表：必须和训练脚本完全一致
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

assert len(FSD50K_LABELS) == 200, f"Expected 200 labels, got {len(FSD50K_LABELS)}"

# ============================================================
# 3. A-MokA 模块：与训练脚本保持一致
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
        elif self.layer_type == "mlp":
            self.A_mlp = nn.Linear(in_features, r, bias=False)
            self.memory_gate = nn.Parameter(torch.ones(1, r) * 0.5)
        else:
            self.A_attn = nn.Linear(in_features, r, bias=False)
            self.attn_interaction = nn.Parameter(torch.zeros(1, r))

        self.B = nn.Linear(r, out_features, bias=False)

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
    A-MokA 推理注入逻辑。
    必须和三分支消融训练脚本完全一致。

    AMOKA_VARIANT:
        full      : audio + mlp + broad q/v attention
        wo_audio  : 去掉 audio/projector branch
        wo_mlp    : 去掉 MLP branch
        wo_attn   : 去掉 broad q/v attention branch
        all_r32   : audio/mlp/attn 全部 r=32
        all_r16   : audio/mlp/attn 全部 r=16
    """
    print(f"Injecting A-MokA adapters for inference. Variant = {AMOKA_VARIANT}")

    audio_keys = ["audio_tower.layers", "multi_modal_projector"]
    mlp_keys = ["gate_proj", "up_proj", "down_proj"]

    # 关键：broad q/v 覆盖方式，必须和训练一致
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
    injected_count = {
        "Audio": 0,
        "MLP": 0,
        "Attention": 0,
    }
    injected_names = {
        "Audio": [],
        "MLP": [],
        "Attention": [],
    }

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

        # 注意优先级：
        # audio/projector 优先于 broad q/v，避免 audio tower 内 q/v 被误归到 attention branch。
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

    print("A-MokA inference injection finished.")
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
    保持和训练 baseline 一致。
    """
    print("Injecting homogeneous LoRA adapters for inference.")

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
# 5. checkpoint 自动查找与加载
# ============================================================
def find_latest_checkpoint(run_name, output_root):
    """
    自动寻找最新 checkpoint。
    兼容：
    1. OUTPUT_ROOT/RUN_NAME/checkpoint-xxxx
    2. OUTPUT_ROOT/RUN_NAME/RUN_NAME/checkpoint-xxxx
    3. OUTPUT_ROOT/checkpoint-xxxx
    """
    candidates = []

    root1 = Path(output_root) / run_name
    root2 = Path(output_root) / run_name / run_name
    root3 = Path(output_root)

    for root in [root1, root2, root3]:
        if root.exists():
            candidates.extend(glob.glob(str(root / "checkpoint-*")))

    valid = []
    for p in candidates:
        pp = Path(p)
        if not pp.is_dir():
            continue
        weight_files = list(pp.glob("*.safetensors")) + list(pp.glob("*.bin"))
        if weight_files:
            valid.append(pp)

    if not valid:
        raise FileNotFoundError(
            "No valid checkpoint found. Please set CHECKPOINT_PATH explicitly.\n"
            f"Searched under:\n"
            f"  {root1}\n"
            f"  {root2}\n"
            f"  {root3}"
        )

    def step_id(path):
        m = re.search(r"checkpoint-(\d+)$", str(path))
        return int(m.group(1)) if m else -1

    valid = sorted(valid, key=step_id)
    return str(valid[-1])

def get_trainable_markers(adapter_type):
    if adapter_type == "amoka":
        return [
            "A_audio",
            "A_mlp",
            "A_attn",
            "B",
            "memory_gate",
            "attn_interaction",
        ]
    else:
        return [
            "lora_A",
            "lora_B",
        ]

def load_checkpoint_with_report(model, checkpoint_path, adapter_type):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    weight_files = sorted(glob.glob(str(checkpoint_path / "*.safetensors")))
    if not weight_files:
        weight_files = sorted(glob.glob(str(checkpoint_path / "*.bin")))

    if not weight_files:
        raise FileNotFoundError(f"No .safetensors or .bin files found in: {checkpoint_path}")

    print("\nLoading checkpoint files:")
    for f in weight_files:
        print(f"  {f}")

    state_dict_all = {}
    for f in weight_files:
        if f.endswith(".safetensors"):
            sd = load_file(f)
        else:
            sd = torch.load(f, map_location="cpu")
        state_dict_all.update(sd)

    trainable_markers = get_trainable_markers(adapter_type)
    model_param_names = set(name for name, _ in model.named_parameters())

    loaded_keys = []
    loaded_adapter_keys = []
    missing_trainable_keys = []
    unexpected_keys = []
    shape_mismatch_keys = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict_all:
                src = state_dict_all[name]
                if tuple(src.shape) != tuple(param.shape):
                    shape_mismatch_keys.append((name, tuple(src.shape), tuple(param.shape)))
                    continue

                param.copy_(src.to(device=param.device, dtype=param.dtype))
                loaded_keys.append(name)

                if any(k in name for k in trainable_markers):
                    loaded_adapter_keys.append(name)
            else:
                if any(k in name for k in trainable_markers):
                    missing_trainable_keys.append(name)

    for key in state_dict_all.keys():
        if key not in model_param_names:
            unexpected_keys.append(key)

    print("\n================ Checkpoint Loading Report ================")
    print(f"Adapter type                     : {adapter_type}")
    print(f"AMOKA_VARIANT                    : {AMOKA_VARIANT}")
    print(f"Checkpoint path                  : {checkpoint_path}")
    print(f"Total checkpoint tensors          : {len(state_dict_all)}")
    print(f"Loaded tensors                    : {len(loaded_keys)}")
    print(f"Loaded adapter trainable tensors  : {len(loaded_adapter_keys)}")
    print(f"Missing adapter trainable tensors : {len(missing_trainable_keys)}")
    print(f"Unexpected checkpoint tensors     : {len(unexpected_keys)}")
    print(f"Shape mismatch tensors            : {len(shape_mismatch_keys)}")

    if missing_trainable_keys:
        print("\n[Warning] Missing adapter tensors, first 40:")
        for k in missing_trainable_keys[:40]:
            print("  ", k)

    if unexpected_keys:
        print("\n[Info] Unexpected checkpoint tensors, first 20:")
        for k in unexpected_keys[:20]:
            print("  ", k)

    if shape_mismatch_keys:
        print("\n[Warning] Shape mismatch tensors, first 20:")
        for name, ckpt_shape, model_shape in shape_mismatch_keys[:20]:
            print(f"  {name}: ckpt={ckpt_shape}, model={model_shape}")

    print("===========================================================\n")

    if len(loaded_adapter_keys) == 0:
        raise RuntimeError(
            "No adapter parameters were loaded. "
            "Please check ADAPTER_TYPE, AMOKA_VARIANT, CHECKPOINT_PATH and injection function."
        )

    return model

# ============================================================
# 6. 标签解析与指标
# ============================================================
def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def extract_labels(text):
    """
    从模型输出或 GT response 中提取 active-200 标签。
    兼容 CoT 和 Direct：
    - CoT: 优先从最后一个 Final Conclusion 之后解析；
    - Direct: 没有 Final Conclusion 时从全文解析。
    """
    if not text:
        return [], []

    label_map = {lab.lower(): lab for lab in FSD50K_LABELS}
    text_lower = text.lower()

    if "final conclusion" in text_lower:
        idx = text_lower.rfind("final conclusion")
        parse_lower = text_lower[idx:]
    else:
        parse_lower = text_lower

    if "categories are" in parse_lower:
        labels_part = parse_lower.split("categories are", 1)[-1]
    elif "category is" in parse_lower:
        labels_part = parse_lower.split("category is", 1)[-1]
    else:
        return [], []

    labels_part = labels_part.replace("(additional sounds detected:", ",")
    labels_part = labels_part.split("\n", 1)[0]
    labels_part = labels_part.strip()
    labels_part = re.sub(r"\.\s*$", "", labels_part)

    raw_items = [x.strip() for x in labels_part.split(",") if x.strip()]

    strict_labels = []
    raw_labels = []

    for item in raw_items:
        raw = item.strip(" .。,:;，；\"'")
        raw = raw.replace(" ", "_")

        if raw.endswith(")") and raw.lower() not in label_map:
            candidate = raw[:-1]
            if candidate.lower() in label_map:
                raw = candidate

        raw_labels.append(raw)

        key = raw.lower()
        if key in label_map:
            strict_labels.append(label_map[key])

    return unique_keep_order(strict_labels), unique_keep_order(raw_labels)

def compute_sample_metrics(y_true, y_pred):
    true_set = set(y_true)
    pred_set = set(y_pred)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1, tp, fp, fn

# ============================================================
# 7. 数据读取：同时支持 JSONL 和 HF cache
# ============================================================
def load_eval_dataset(data_path):
    """
    同时支持两种验证集格式：

    1. JSONL:
       每行包含 query/response/audios。

    2. HuggingFace save_to_disk cache:
       包含 input_ids/labels/input_features/feature_attention_mask 等。
       推理时会根据 labels != -100 找到 response 起点，只保留 prompt 部分，避免 GT 泄漏。
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Eval data path does not exist: {data_path}")

    # ------------------------------------------------------------
    # 7.1 HF cache directory
    # ------------------------------------------------------------
    if data_path.is_dir():
        manifest_path = data_path / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            datasets = []
            for shard in manifest["shards"]:
                shard_idx = shard["shard_idx"]
                shard_dir = data_path / f"shard_{shard_idx:05d}"

                if not shard_dir.exists():
                    raise FileNotFoundError(f"Shard directory not found: {shard_dir}")

                print(f"Loading eval cache shard: {shard_dir}")
                datasets.append(load_from_disk(str(shard_dir)))

            dataset = concatenate_datasets(datasets)
        else:
            dataset = load_from_disk(str(data_path))

        required_columns = {
            "input_ids",
            "labels",
            "input_features",
            "feature_attention_mask",
        }

        missing = required_columns - set(dataset.column_names)
        if missing:
            raise ValueError(f"Eval cache is missing columns: {sorted(missing)}")

        if MAX_INFER_SAMPLES is not None:
            dataset = dataset.select(range(min(MAX_INFER_SAMPLES, len(dataset))))

        print(f"Loaded HF eval cache: {data_path}")
        print(f"Samples: {len(dataset)}")
        print(f"Columns: {dataset.column_names}")

        return {
            "type": "cache",
            "data": dataset,
        }

    # ------------------------------------------------------------
    # 7.2 JSONL file
    # ------------------------------------------------------------
    data = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            for key in ["query", "response", "audios"]:
                if key not in obj:
                    raise KeyError(f"line {line_idx}: missing key `{key}`")

            audios = obj["audios"]
            if isinstance(audios, str):
                audios = [audios]

            for audio_path in audios:
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(
                        f"line {line_idx}: audio file not found: {audio_path}"
                    )

            data.append({
                "query": obj["query"],
                "response": obj["response"],
                "audios": audios,
            })

    if MAX_INFER_SAMPLES is not None:
        data = data[:MAX_INFER_SAMPLES]

    print(f"Loaded JSONL eval data: {data_path}")
    print(f"Samples: {len(data)}")

    return {
        "type": "jsonl",
        "data": data,
    }

def decode_gt_from_cached_labels(labels, tokenizer):
    """
    从 SFT cache 的 labels 中恢复 GT response。
    labels 中 prompt 部分通常为 -100，response 部分是真实 token。
    """
    labels = np.array(labels)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    return tokenizer.decode(labels.tolist(), skip_special_tokens=True)

def find_response_start(labels):
    """
    找到 response 起点。
    labels != -100 的第一个位置就是模型需要学习生成的 response 起点。
    """
    labels = np.array(labels)
    valid_pos = np.where(labels != -100)[0]

    if len(valid_pos) == 0:
        return None

    return int(valid_pos[0])

def maybe_to_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.tensor(value)
    if isinstance(value, list):
        return torch.tensor(value)
    return value

def build_generation_item_from_cache(item, prefix_ids):
    """
    将 SFT cache 中的 full input_ids 截断为 prompt-only。
    不能直接使用完整 input_ids，否则会把 GT response 泄漏给模型。
    """
    if "labels" not in item:
        raise KeyError("Cached item does not contain labels; cannot find response boundary.")

    response_start = find_response_start(item["labels"])

    if response_start is None:
        raise ValueError("Cannot find response start because all labels are -100.")

    gen_item = {}

    for key, value in item.items():
        if key == "labels":
            continue

        if key == "input_ids":
            prompt_ids = list(value[:response_start])
            if USE_RESPONSE_PREFIX:
                prompt_ids = prompt_ids + list(prefix_ids)
            gen_item[key] = torch.tensor(prompt_ids, dtype=torch.long)

        elif key == "attention_mask":
            prompt_mask = list(value[:response_start])
            if USE_RESPONSE_PREFIX:
                prompt_mask = prompt_mask + [1] * len(prefix_ids)
            gen_item[key] = torch.tensor(prompt_mask, dtype=torch.long)

        elif key == "position_ids":
            prompt_pos = list(value[:response_start])
            if USE_RESPONSE_PREFIX:
                start_pos = prompt_pos[-1] + 1 if len(prompt_pos) > 0 else 0
                prompt_pos = prompt_pos + list(range(start_pos, start_pos + len(prefix_ids)))
            gen_item[key] = torch.tensor(prompt_pos, dtype=torch.long)

        elif key == "audio_info":
            gen_item[key] = value

        else:
            gen_item[key] = maybe_to_tensor(value)

    return gen_item

def encode_generation_item_from_jsonl(template, item, prefix_ids):
    """
    JSONL 模式下构造 prompt-only 输入。
    只使用 query 和 audios，response 置空，避免 GT 泄漏。
    """
    encode_obj = {
        "query": item["query"],
        "response": "",
        "audios": item["audios"],
    }

    encoded = template.encode(encode_obj)

    if isinstance(encoded, tuple):
        encoded = encoded[0]

    if not isinstance(encoded, dict):
        raise TypeError(f"template.encode() should return dict, got {type(encoded)}")

    encoded.pop("labels", None)

    if "input_ids" not in encoded or encoded["input_ids"] is None:
        raise KeyError("encoded does not contain input_ids")

    encoded["input_ids"] = list(encoded["input_ids"])

    if USE_RESPONSE_PREFIX and len(prefix_ids) > 0:
        encoded["input_ids"] = encoded["input_ids"] + list(prefix_ids)

        if "attention_mask" in encoded and encoded["attention_mask"] is not None:
            encoded["attention_mask"] = list(encoded["attention_mask"]) + [1] * len(prefix_ids)

        if "position_ids" in encoded and encoded["position_ids"] is not None:
            pos = list(encoded["position_ids"])
            start_pos = pos[-1] + 1 if len(pos) > 0 else 0
            encoded["position_ids"] = pos + list(range(start_pos, start_pos + len(prefix_ids)))

    for key in ["input_ids", "attention_mask", "position_ids"]:
        if key in encoded and encoded[key] is not None:
            encoded[key] = torch.tensor(encoded[key], dtype=torch.long)

    return encoded

# ============================================================
# 8. 推理主函数
# ============================================================
def run_inference():
    global CHECKPOINT_PATH

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if not CHECKPOINT_PATH:
        CHECKPOINT_PATH = find_latest_checkpoint(RUN_NAME, OUTPUT_ROOT)

    print("\n================ Inference Configuration ================")
    print(f"RUN_NAME            : {RUN_NAME}")
    print(f"ADAPTER_TYPE        : {ADAPTER_TYPE}")
    print(f"SUPERVISION_TYPE    : {SUPERVISION_TYPE}")
    print(f"AMOKA_VARIANT       : {AMOKA_VARIANT}")
    print(f"Base model          : {BASE_MODEL_PATH}")
    print(f"Output root         : {OUTPUT_ROOT}")
    print(f"Checkpoint          : {CHECKPOINT_PATH}")
    print(f"Eval data path      : {TEST_DATA_PATH}")
    print(f"Output path         : {OUTPUT_RESULT_PATH}")
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Max samples         : {MAX_INFER_SAMPLES}")
    print(f"Use response prefix : {USE_RESPONSE_PREFIX}")
    print(f"Response prefix     : {repr(RESPONSE_PREFIX)}")
    print(f"Max new tokens      : {MAX_NEW_TOKENS}")
    print(f"Device              : {device}")
    print("=========================================================\n")

    # ------------------------------------------------------------
    # 8.1 加载 base model
    # ------------------------------------------------------------
    print("Loading Qwen2-Audio base model...")

    model, tokenizer = get_model_tokenizer(
        ModelType.qwen2_audio_7b_instruct,
        torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        model_id_or_path=BASE_MODEL_PATH,
        device_map="auto" if torch.cuda.is_available() else None,
        model_kwargs={
            "attn_implementation": "sdpa",
        },
    )

    # ------------------------------------------------------------
    # 8.2 注入 adapter
    # ------------------------------------------------------------
    if ADAPTER_TYPE == "amoka":
        model = inject_moka(model)
    elif ADAPTER_TYPE == "lora":
        model = inject_lora(model)
    else:
        raise ValueError(f"Unsupported ADAPTER_TYPE: {ADAPTER_TYPE}")

    # ------------------------------------------------------------
    # 8.3 加载 checkpoint
    # ------------------------------------------------------------
    model = load_checkpoint_with_report(model, CHECKPOINT_PATH, ADAPTER_TYPE)
    model.eval()

    # ------------------------------------------------------------
    # 8.4 tokenizer / generation 设置
    # ------------------------------------------------------------
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    model.config.use_cache = True
    model.generation_config.use_cache = True

    model.generation_config.max_new_tokens = MAX_NEW_TOKENS
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.generation_config.repetition_penalty = 1.0

    template = get_template("qwen2-audio", tokenizer, max_length=2048)

    # ------------------------------------------------------------
    # 8.5 读取验证数据：cache 或 jsonl
    # ------------------------------------------------------------
    eval_pack = load_eval_dataset(TEST_DATA_PATH)
    eval_type = eval_pack["type"]
    dataset = eval_pack["data"]

    print(f"Loaded inference samples: {len(dataset)}")
    print(f"Eval data type: {eval_type}")

    prefix_ids = (
        tokenizer.encode(RESPONSE_PREFIX, add_special_tokens=False)
        if USE_RESPONSE_PREFIX
        else []
    )

    output_dir = os.path.dirname(OUTPUT_RESULT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(OUTPUT_RESULT_PATH):
        os.remove(OUTPUT_RESULT_PATH)

    print(f"\nStart inference: Batch={BATCH_SIZE}, max_new_tokens={MAX_NEW_TOKENS}\n")

    label_set_lower = {x.lower() for x in FSD50K_LABELS}

    total_tp = 0
    total_fp = 0
    total_fn = 0
    sample_f1_list = []

    # ------------------------------------------------------------
    # 8.6 批量推理
    # ------------------------------------------------------------
    for start in tqdm.tqdm(range(0, len(dataset), BATCH_SIZE)):
        end = min(start + BATCH_SIZE, len(dataset))

        if eval_type == "cache":
            batch_items = [dataset[i] for i in range(start, end)]
        else:
            batch_items = dataset[start:end]

        encoded_list = []
        gt_text_list = []

        for item in batch_items:
            if eval_type == "cache":
                gt_text = decode_gt_from_cached_labels(item["labels"], tokenizer)
                encoded = build_generation_item_from_cache(
                    item=item,
                    prefix_ids=prefix_ids,
                )
            else:
                gt_text = item["response"]
                encoded = encode_generation_item_from_jsonl(
                    template=template,
                    item=item,
                    prefix_ids=prefix_ids,
                )

            gt_text_list.append(gt_text)
            encoded_list.append(encoded)

        batch = template.data_collator(encoded_list)

        for k in list(batch.keys()):
            if k == "audio_info":
                continue
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **batch,
                generation_config=model.generation_config,
            )

        input_token_len = batch["input_ids"].shape[1]

        with open(OUTPUT_RESULT_PATH, "a", encoding="utf-8") as f_out:
            for j, item in enumerate(batch_items):
                gen_ids = output_ids[j][input_token_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                if USE_RESPONSE_PREFIX:
                    full_response = RESPONSE_PREFIX + gen_text
                else:
                    full_response = gen_text

                gt_strict, gt_raw = extract_labels(gt_text_list[j])
                pred_strict, pred_raw = extract_labels(full_response)

                invalid_pred_raw = [
                    x for x in pred_raw
                    if x.lower() not in label_set_lower
                ]

                p, r, f1, tp, fp, fn = compute_sample_metrics(gt_strict, pred_strict)

                total_tp += tp
                total_fp += fp
                total_fn += fn
                sample_f1_list.append(f1)

                if eval_type == "jsonl":
                    audio_info = item.get("audios", None)
                    query_text = item.get("query", None)
                else:
                    audio_info = item.get("audios", None) if isinstance(item, dict) else None
                    query_text = item.get("query", None) if isinstance(item, dict) else None

                record = {
                    "sample_index": start + j,
                    "run_name": RUN_NAME,
                    "adapter_type": ADAPTER_TYPE,
                    "supervision_type": SUPERVISION_TYPE,
                    "amoka_variant": AMOKA_VARIANT,
                    "eval_type": eval_type,
                    "checkpoint_path": CHECKPOINT_PATH,
                    "audio": audio_info,
                    "query": query_text,
                    "ground_truth_strict": gt_strict,
                    "ground_truth_raw": gt_raw,
                    "prediction_strict": pred_strict,
                    "prediction_raw": pred_raw,
                    "invalid_prediction_raw": invalid_pred_raw,
                    "num_invalid_prediction_raw": len(invalid_pred_raw),
                    "sample_precision": p,
                    "sample_recall": r,
                    "sample_f1": f1,
                    "ground_truth_text": gt_text_list[j],
                    "full_response": full_response,
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        del batch, output_ids, encoded_list, batch_items
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 8.7 汇总指标
    # ------------------------------------------------------------
    micro_precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )

    micro_recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )

    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    samples_f1 = float(np.mean(sample_f1_list)) if sample_f1_list else 0.0

    summary = {
        "run_name": RUN_NAME,
        "adapter_type": ADAPTER_TYPE,
        "supervision_type": SUPERVISION_TYPE,
        "amoka_variant": AMOKA_VARIANT,
        "eval_type": eval_type,
        "checkpoint_path": CHECKPOINT_PATH,
        "test_data_path": TEST_DATA_PATH,
        "output_result_path": OUTPUT_RESULT_PATH,
        "num_samples": len(dataset),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "samples_f1": samples_f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }

    summary_path = OUTPUT_RESULT_PATH.replace(".jsonl", "_summary.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print("\n================ Inference Summary ================")
    print(json.dumps(summary, ensure_ascii=False, indent=4))
    print("===================================================\n")
    print(f"Inference finished. Results saved to: {OUTPUT_RESULT_PATH}")
    print(f"Summary saved to: {summary_path}\n")

if __name__ == "__main__":
    run_inference()