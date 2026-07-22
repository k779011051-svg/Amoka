# -*- coding: utf-8 -*-
import os
import json
import gc
import shutil
import warnings
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk, concatenate_datasets
from swift.llm import ModelType, get_model_tokenizer, get_template

warnings.filterwarnings("ignore")

# ============================================================
# 1. 路径配置
# ============================================================
BASE_MODEL_PATH = "F:\Qwen2-Audio-main\qwen2-audio-7b-local\Qwen\Qwen2-Audio-7B-Instruct"

TRAIN_JSONL_PATH = r"F:\dataset\FSD50K\amoka\fsd50k_train_direct.jsonl"
VAL_JSONL_PATH = r"F:\dataset\FSD50K\amoka\fsd50k_val_direct.jsonl"

# Ubuntu 上真实音频目录。jsonl 里的 F:\xxx 会被丢弃，只取文件名。
AUDIO_ROOT = "F:\dataset\FSD50K\FSD50K.dev_audio"

TRAIN_CACHE_DIR = r"F:\dataset\FSD50K\fsd50k_processed_train_cache_direct"
VAL_CACHE_DIR = r"F:\dataset\FSD50K\fsd50k_processed_val_cache_direct"

CHUNK_SIZE = 500

TEMPLATE_TYPE = "qwen2-audio"
MAX_LENGTH = 2048

# 测试时可以设成 1000，正式跑设为 None
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

# ============================================================
# 2. 只从原始路径提取 wav 文件名
# ============================================================
def resolve_audio_path(raw_path):
    """
    丢弃 jsonl 中的位置信息，只保留文件名。

    输入示例：
        F:\\dataset\\FSD50K\\FSD50K.dev_audio\\64760.wav

    输出示例：
        /workspace/dataset/FSD50K/FSD50K.dev_audio/64760.wav
    """
    raw_path = str(raw_path).strip()
    filename = os.path.basename(raw_path.replace("\\", "/"))
    return os.path.join(AUDIO_ROOT, filename)

# ============================================================
# 3. 逐行读取 jsonl，不一次性读入内存
# ============================================================
def iter_jsonl(jsonl_path, max_samples=None):
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    sample_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            if "query" not in obj:
                raise KeyError(f"line {line_idx}: missing query")
            if "response" not in obj:
                raise KeyError(f"line {line_idx}: missing response")
            if "audios" not in obj:
                raise KeyError(f"line {line_idx}: missing audios")

            audios = obj["audios"]
            if isinstance(audios, str):
                audios = [audios]

            resolved_audios = [resolve_audio_path(x) for x in audios]

            for audio_path in resolved_audios:
                if line_idx < 3:
                    print(
                        f"[Check audio] line={line_idx}, "
                        f"path={audio_path}, exists={os.path.exists(audio_path)}"
                    )

                if not os.path.exists(audio_path):
                    raise FileNotFoundError(
                        f"Audio file not found at line {line_idx}: {audio_path}\n"
                        f"请检查 AUDIO_ROOT 是否正确：{AUDIO_ROOT}"
                    )

            yield {
                "query": obj["query"],
                "response": obj["response"],
                "audios": resolved_audios,
            }

            sample_count += 1
            if max_samples is not None and sample_count >= max_samples:
                break

# ============================================================
# 4. 删除所有位置信息，只保留训练特征
# ============================================================
def strip_to_training_features(encoded):
    """
    最终 cache 只保留训练/预测需要的张量特征。
    不保留：
        audios
        audio
        audio_info
        audio_path
        path
        query
        response
        raw text
    """
    keep_keys = [
        "input_ids",
        "labels",
        "attention_mask",
        "input_features",
        "feature_attention_mask",
    ]

    cleaned = {}

    for key in keep_keys:
        if key in encoded:
            cleaned[key] = encoded[key]

    required_keys = [
        "input_ids",
        "labels",
        "input_features",
        "feature_attention_mask",
    ]

    missing = [key for key in required_keys if key not in cleaned]
    if missing:
        raise KeyError(
            f"Encoded sample missing required feature keys: {missing}. "
            f"Existing keys: {list(encoded.keys())}"
        )

    return cleaned

# ============================================================
# 5. 单条编码
# ============================================================
def encode_one_sample(template, item, sample_idx):
    try:
        encoded = template.encode({
            "query": item["query"],
            "response": item["response"],
            "audios": item["audios"],
        })

        if isinstance(encoded, tuple):
            encoded = encoded[0]

        if not isinstance(encoded, dict):
            raise TypeError(f"template.encode returned {type(encoded)}, expected dict")

        encoded = strip_to_training_features(encoded)
        return encoded

    except Exception as exc:
        raise RuntimeError(
            f"Failed to encode sample {sample_idx}, "
            f"err={repr(exc)}"
        )

# ============================================================
# 6. 保存单个 shard
# ============================================================
def save_shard(samples, shard_dir):
    shard_dir = Path(shard_dir)

    if shard_dir.exists():
        shutil.rmtree(shard_dir)

    dataset = Dataset.from_list(samples)

    # 再保险：保存前强制去掉非训练字段
    allowed_columns = {
        "input_ids",
        "labels",
        "attention_mask",
        "input_features",
        "feature_attention_mask",
    }

    remove_columns = [
        col for col in dataset.column_names
        if col not in allowed_columns
    ]

    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)

    dataset.save_to_disk(str(shard_dir))

    print(f"[Saved shard] {shard_dir}")
    print(f"  samples : {len(dataset)}")
    print(f"  columns : {dataset.column_names}")

    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# 7. 构建 cache：每 500 条保存一次，最终合并
# ============================================================
def build_cache(jsonl_path, output_cache_dir, template, max_samples=None):
    output_cache_dir = Path(output_cache_dir)
    shard_root = output_cache_dir.with_name(output_cache_dir.name + "_shards")

    if shard_root.exists():
        print(f"Removing old shard cache: {shard_root}")
        shutil.rmtree(shard_root)

    if output_cache_dir.exists():
        print(f"Removing old final cache: {output_cache_dir}")
        shutil.rmtree(output_cache_dir)

    shard_root.mkdir(parents=True, exist_ok=True)

    buffer = []
    shard_paths = []

    total = 0
    shard_idx = 0

    for item in iter_jsonl(jsonl_path, max_samples=max_samples):
        encoded = encode_one_sample(template, item, sample_idx=total)
        buffer.append(encoded)
        total += 1

        if len(buffer) >= CHUNK_SIZE:
            shard_dir = shard_root / f"shard_{shard_idx:05d}"
            save_shard(buffer, shard_dir)

            shard_paths.append(str(shard_dir))
            buffer = []
            shard_idx += 1

            print(f"[Progress] encoded={total}")

    if buffer:
        shard_dir = shard_root / f"shard_{shard_idx:05d}"
        save_shard(buffer, shard_dir)

        shard_paths.append(str(shard_dir))
        buffer = []

    print(f"All shards saved. total_samples={total}, num_shards={len(shard_paths)}")

    manifest = {
        "jsonl_path": str(jsonl_path),
        "output_cache_dir": str(output_cache_dir),
        "shard_root": str(shard_root),
        "chunk_size": CHUNK_SIZE,
        "total_samples": total,
        "num_shards": len(shard_paths),
        "shards": shard_paths,
        "kept_columns": [
            "input_ids",
            "labels",
            "attention_mask",
            "input_features",
            "feature_attention_mask",
        ],
    }

    with open(shard_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Merging shards into final cache: {output_cache_dir}")

    datasets = [load_from_disk(path) for path in shard_paths]
    merged = concatenate_datasets(datasets)

    allowed_columns = {
        "input_ids",
        "labels",
        "attention_mask",
        "input_features",
        "feature_attention_mask",
    }

    remove_columns = [
        col for col in merged.column_names
        if col not in allowed_columns
    ]

    if remove_columns:
        merged = merged.remove_columns(remove_columns)

    merged.save_to_disk(str(output_cache_dir))

    print("=" * 80)
    print(f"Final cache saved : {output_cache_dir}")
    print(f"Samples           : {len(merged)}")
    print(f"Columns           : {merged.column_names}")
    print("=" * 80)

    del merged
    del datasets
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# 8. 主函数
# ============================================================
def main():
    print("\n================ Build Direct Feature-Only Cache ================")
    print(f"BASE_MODEL_PATH  : {BASE_MODEL_PATH}")
    print(f"AUDIO_ROOT       : {AUDIO_ROOT}")
    print(f"TRAIN_JSONL_PATH : {TRAIN_JSONL_PATH}")
    print(f"VAL_JSONL_PATH   : {VAL_JSONL_PATH}")
    print(f"TRAIN_CACHE_DIR  : {TRAIN_CACHE_DIR}")
    print(f"VAL_CACHE_DIR    : {VAL_CACHE_DIR}")
    print(f"CHUNK_SIZE       : {CHUNK_SIZE}")
    print("================================================================\n")

    print("Loading tokenizer and template...")

    model, tokenizer = get_model_tokenizer(
        ModelType.qwen2_audio_7b_instruct,
        torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        model_id_or_path=BASE_MODEL_PATH,
        model_kwargs={
            "device_map": {"": 0} if torch.cuda.is_available() else None,
        },
    )

    template = get_template(
        TEMPLATE_TYPE,
        tokenizer,
        None,
        MAX_LENGTH,
    )

    # 只需要 tokenizer / processor / template，编码完成前不需要模型前向
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nBuilding train cache...")
    build_cache(
        jsonl_path=TRAIN_JSONL_PATH,
        output_cache_dir=TRAIN_CACHE_DIR,
        template=template,
        max_samples=MAX_TRAIN_SAMPLES,
    )

    print("\nBuilding validation cache...")
    build_cache(
        jsonl_path=VAL_JSONL_PATH,
        output_cache_dir=VAL_CACHE_DIR,
        template=template,
        max_samples=MAX_VAL_SAMPLES,
    )

    print("\nDirect feature-only cache building finished.")

if __name__ == "__main__":
    main()