

import json
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel


# ========== 1. 路径配置 ==========

BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_FILE = (
    BASE_DIR / "data/data_processed/image_manifest.jsonl"
)

OUTPUT_DIR = BASE_DIR / "data/features/image"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ========== 2. 模型配置 ==========

MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16
CHUNK_SIZE = 1000

print("运行设备：", DEVICE)


# ========== 3. 加载 CLIP 模型 ==========

# cast 用于兼容部分 Transformers 版本的类型声明
image_processor: Any = cast(
    Any,
    CLIPImageProcessor.from_pretrained(MODEL_NAME)
)

model: Any = cast(
    Any,
    CLIPModel.from_pretrained(MODEL_NAME)
)

model.to(DEVICE)
model.eval()

print("CLIP 模型加载完成")


# ========== 4. 初始化变量 ==========

feature_ids = []
feature_vectors = []

chunk_index = 0
total = 0
failed = 0


# ========== 5. 保存特征 ==========

def save_chunk():
    global chunk_index

    if not feature_vectors:
        return

    embeddings = torch.cat(feature_vectors, dim=0)

    output_path = (
        OUTPUT_DIR / f"product_image_{chunk_index:04d}.pt"
    )

    torch.save(
        {
            "ids": feature_ids.copy(),
            "embeddings": embeddings
        },
        output_path
    )

    print("保存特征：", output_path)

    feature_ids.clear()
    feature_vectors.clear()

    chunk_index += 1


# ========== 6. 批量提取图像特征 ==========

@torch.inference_mode()
def extract_batch(items):

    global total, failed

    images = []
    ids = []

    # 读取图片
    for product_id, image_path in items:

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            images.append(image)
            ids.append(product_id)

        except (OSError, UnidentifiedImageError) as e:
            failed += 1
            print(f"图片读取失败：{image_path}，原因：{e}")

    if not images:
        return

    # CLIP 图像预处理
    inputs = image_processor.preprocess(
        images=images,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(DEVICE)

    # 获取 CLIP 图像编码器的池化特征
    outputs = model.vision_model(
        pixel_values=pixel_values
    )

    # 获取池化后的图像表示
    pooled_output = outputs.pooler_output

    # 通过 CLIP 投影层，得到 512 维图像特征
    embeddings = model.visual_projection(pooled_output)

    # L2 归一化
    mbeddings = torch.nn.functional.normalize(
        embeddings.float(),
        p=2,
        dim=-1
    ).cpu()


    # 保存 ID 和特征的对应关系
    for i, product_id in enumerate(ids):
        feature_ids.append(product_id)
        feature_vectors.append(embeddings[i:i + 1])

    total += len(ids)

    # 达到指定数量就保存
    if len(feature_ids) >= CHUNK_SIZE:
        save_chunk()


# ========== 7. 主程序 ==========

def main():

    global failed

    batch = []

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in tqdm(f, desc="CLIP图像特征提取"):

            record = json.loads(line)

            product_id = record["parent_asin"]

            image_path = (
                BASE_DIR / record["image_path"]
            )

            # 跳过不存在的图片
            if not image_path.is_file():
                failed += 1
                continue

            batch.append((product_id, image_path))

            # 批量处理
            if len(batch) >= BATCH_SIZE:
                extract_batch(batch)
                batch.clear()

    # 处理最后不足一个 batch 的图片
    if batch:
        extract_batch(batch)

    # 保存剩余特征
    save_chunk()

    print("\n========== 提取完成 ==========")
    print("成功提取商品数：", total)
    print("失败图片数：", failed)
    print("特征文件数：", chunk_index)


if __name__ == "__main__":
    main()
