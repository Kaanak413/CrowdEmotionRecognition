import os
from pathlib import Path
from PIL import Image

path = '/home/kaan/Desktop/Projects-ML/EmotionRecognition/super/emotion_images/'
image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp"]  # supported extensions
img_type_accepted_by_tf = {"jpeg", "png", "gif", "bmp"}  # TensorFlow-accepted formats
totalFiledeleted = 0

for filepath in Path(path).rglob("*"):
    # Skip non-image files and hidden files
    if filepath.suffix.lower() in image_extensions and not filepath.name.startswith("."):
        try:
            # Verify that the file is a readable image
            with Image.open(filepath) as img:
                img_type = img.format.lower()
                
                # Remove file if not in TensorFlow-accepted formats
                if img_type not in img_type_accepted_by_tf:
                    totalFiledeleted += 1
                    os.remove(filepath)
                    print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
        except (IOError, SyntaxError) as e:
            print(f"{filepath} is not a valid image or is corrupted. Deleting...")
            os.remove(filepath)
            totalFiledeleted += 1

print(f"Total files deleted: {totalFiledeleted}")