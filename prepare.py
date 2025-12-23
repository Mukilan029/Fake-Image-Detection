import os
from PIL import Image, ImageChops, ImageEnhance
from io import BytesIO

# -------- CONFIG --------
INPUT_REAL = r"D:\FakeImageDetection\dataset\real"
INPUT_FAKE = r"D:\FakeImageDetection\dataset\fake"

OUTPUT_REAL = r"D:\FakeImageDetection\dataset_ela\real"
OUTPUT_FAKE = r"D:\FakeImageDetection\dataset_ela\fake"

JPEG_QUALITY = 90
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif")

# ------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_to_ela(input_path, output_path):
    try:
        original = Image.open(input_path).convert("RGB")

        buffer = BytesIO()
        original.save(buffer, "JPEG", quality=JPEG_QUALITY)
        buffer.seek(0)
        compressed = Image.open(buffer)

        ela = ImageChops.difference(original, compressed)

        extrema = ela.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1
        scale = 255.0 / max_diff

        ela = ImageEnhance.Brightness(ela).enhance(scale)
        ela.save(output_path)

        return True
    except Exception as e:
        print(f"❌ Failed: {input_path} | {e}")
        return False

def process_folder(input_dir, output_dir):
    ensure_dir(output_dir)
    files = os.listdir(input_dir)
    print(f"📂 Processing {input_dir} ({len(files)} files)")

    processed = 0
    for file in files:
        if file.lower().endswith(IMAGE_EXTENSIONS):
            src = os.path.join(input_dir, file)
            dst = os.path.join(output_dir, os.path.splitext(file)[0] + ".jpg")

            if not os.path.exists(dst):
                if convert_to_ela(src, dst):
                    processed += 1

        if processed % 500 == 0 and processed > 0:
            print(f"  ✔ {processed} images processed")

    print(f"✅ Finished {output_dir}")

# -------- RUN --------
if __name__ == "__main__":
    process_folder(INPUT_REAL, OUTPUT_REAL)
    process_folder(INPUT_FAKE, OUTPUT_FAKE)
    print("🎉 ELA preprocessing completed")
