import os
import shutil
import random

# --- YOLLAR ---
image_dir = r"C:\Users\MSI\Desktop\Python\Turk-Isaret-Dili-Algilama\TID_data"
label_dir = r"C:\Users\MSI\Desktop\Python\YOLO_dataset\yoloData"
output_dir = r"C:\Users\MSI\Desktop\YOLO_Final_Dataset"

# Klasör yapısını oluştur
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

# Tüm resimleri listele (Alt klasörlerden topla)
all_images = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            all_images.append(os.path.join(root, file))

# Veriyi karıştır ve %80 Train, %20 Val olarak böl
random.shuffle(all_images)
split_idx = int(len(all_images) * 0.8)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

def move_files(image_list, split):
    for img_path in image_list:
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(label_dir, name + ".txt")
        
        if os.path.exists(lbl_path):
            # Resmi ve Etiketi kopyala
            shutil.copy(img_path, os.path.join(output_dir, split, 'images', os.path.basename(img_path)))
            shutil.copy(lbl_path, os.path.join(output_dir, split, 'labels', name + ".txt"))

move_files(train_images, 'train')
move_files(val_images, 'val')
print(f"✅ Bitti! Veriler {output_dir} klasöründe toplandı.")