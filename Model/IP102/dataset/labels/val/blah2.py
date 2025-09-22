import os
import shutil

image_root = "PlantYOLO/images"
label_root = "PlantYOLO/labels"

for split in ["train", "val"]:
    split_image_path = os.path.join(image_root, split)
    split_label_path = os.path.join(label_root, split)
    
    for class_id in os.listdir(split_image_path):
        class_folder = os.path.join(split_image_path, class_id)
        if not os.path.isdir(class_folder):
            continue

        for img_file in os.listdir(class_folder):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            # Move image
            src_img = os.path.join(class_folder, img_file)
            dst_img = os.path.join(split_image_path, img_file)
            shutil.move(src_img, dst_img)
            
            # Move label
            src_label = os.path.join(split_label_path, os.path.splitext(img_file)[0]+".txt")
            dst_label = os.path.join(split_label_path, os.path.splitext(img_file)[0]+".txt")
            shutil.move(src_label, dst_label)
