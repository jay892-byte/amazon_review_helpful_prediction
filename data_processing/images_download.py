import json
import time
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image, UnidentifiedImageError


# =========================
# 1. 路径配置
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data/data_processed/clean_products.jsonl"
IMAGE_DIR = BASE_DIR / "data/images"
MANIFEST_FILE = BASE_DIR / "data/processed/image_manifest.jsonl"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 下载配置
# =========================

MAX_IMAGES = 1000
TIMEOUT = 15
MAX_BYTES = 10 * 1024 * 1024

session = requests.Session()

session.headers.update({
    "User-Agent": "Mozilla/5.0"
})


# =========================
# 3. 统计变量
# =========================

downloaded = 0
skipped = 0
failed = 0
no_image = 0
processed = 0


# =========================
# 4. 读取已有图片映射
# =========================

manifest_dict = {}

if MANIFEST_FILE.exists():

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            manifest_dict[record["parent_asin"]] = record


# =========================
# 5. 图片下载函数
# =========================

def download_image(url, save_path):

    try:

        with session.get(
            url,
            timeout=TIMEOUT,
            stream=True
        ) as response:

            response.raise_for_status()

            chunks = []
            total_size = 0

            for chunk in response.iter_content(8192):

                if not chunk:
                    continue

                total_size += len(chunk)

                if total_size > MAX_BYTES:
                    print("图片超过大小限制")
                    return False

                chunks.append(chunk)

        image = Image.open(
            BytesIO(b"".join(chunks))
        )

        image.load()
        image = image.convert("RGB")

        # 先保存临时文件，成功后再替换正式文件
        temp_path = save_path.with_suffix(".tmp")

        try:
            image.save(
                temp_path,
                format="JPEG",
                quality=90
            )

            temp_path.replace(save_path)

        finally:
            temp_path.unlink(missing_ok=True)

        return True

    except (
        requests.RequestException,
        OSError,
        ValueError,
        UnidentifiedImageError
    ) as e:

        print(f"下载失败：{url}")
        print(f"原因：{e}")

        return False


# =========================
# 6. 遍历商品并下载
# =========================

seen_products = set()

try:

    with open(INPUT_FILE, "r", encoding="utf-8") as fin:

        for line in fin:

            if not line.strip():
                continue

            product = json.loads(line)

            product_id = product.get("parent_asin")

            if not product_id:
                failed += 1
                continue

            if product_id in seen_products:
                continue

            seen_products.add(product_id)

            # 限制本次处理的商品数量
            if MAX_IMAGES is not None and processed >= MAX_IMAGES:
                break

            processed += 1

            image_urls = product.get("image_urls") or []

            safe_id = "".join(
                c for c in product_id
                if c.isalnum() or c in "-_"
            )

            if not safe_id:
                failed += 1
                continue

            save_path = IMAGE_DIR / f"{safe_id}.jpg"

            # 读取已有图片
            success = False

            if save_path.exists():

                try:
                    with Image.open(save_path) as image:
                        image.verify()

                    skipped += 1
                    success = True

                except (OSError, ValueError):
                    save_path.unlink(missing_ok=True)

            # 下载新图片
            if not success:

                if not isinstance(image_urls, list):
                    image_urls = []

                valid_urls = [
                    url for url in image_urls
                    if isinstance(url, str)
                    and url.startswith(("http://", "https://"))
                ]

                if not valid_urls:
                    no_image += 1
                    continue

                for url in valid_urls:

                    if download_image(url, save_path):

                        downloaded += 1
                        success = True
                        break

                    time.sleep(0.2)

            # 记录成功商品
            if success:

                manifest_dict[product_id] = {
                    "parent_asin": product_id,
                    "image_path": str(
                        save_path.relative_to(BASE_DIR)
                    ).replace("\\", "/")
                }

            else:
                failed += 1

            if processed % 100 == 0:
                print(f"已处理 {processed} 个商品")

finally:
    session.close()


# =========================
# 7. 写入图片映射文件
# =========================

with open(MANIFEST_FILE, "w", encoding="utf-8") as fout:

    for record in manifest_dict.values():

        fout.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


# =========================
# 8. 输出统计结果
# =========================

print("\n图片下载完成")

print("本次处理商品数：", processed)
print("新下载图片数：", downloaded)
print("已有图片数：", skipped)
print("下载失败商品数：", failed)
print("没有可用图片 URL 的商品数：", no_image)
print("累计映射商品数：", len(manifest_dict))