import cv2
import mediapipe as mp
import os

# --- 1. YOLLARI AYARLA ---
# Resimlerinin olduğu ana klasör (TID_data)
IMAGE_DIR = r"C:\Users\MSI\Desktop\Python\Turk-Isaret-Dili-Algilama\TID_data"
# Etiketlerin (.txt) gideceği klasör
LABEL_DIR = r"C:\Users\MSI\Desktop\Python\YOLO_dataset\yoloData"

# --- 2. GÜVENLİ SINIF LİSTESİ (Sıralama Çok Önemli!) ---
classes = [
    "i", "s", "o", "c", "u", "A", "B", "C", "D", "E", 
    "F", "G", "g", "H", "I", "J", "K", "L", "M", "N", 
    "O", "P", "R", "S", "T", "U", "V", "Y", "Z"
]

if not os.path.exists(LABEL_DIR):
    os.makedirs(label_dir)

# MediaPipe Hands Kurulumu
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

print("🚀 Otomatik etiketleme başliyor... Lütfen bekle.")

# Klasörleri tek tek tara
for folder_name in os.listdir(IMAGE_DIR):
    folder_path = os.path.join(IMAGE_DIR, folder_name)
    if not os.path.isdir(folder_path): continue

    # --- 3. SEMBOL EŞLEŞTİRME (KRİTİK KISIM) ---
    class_name = folder_name
    if folder_name == "!": class_name = "i"
    elif folder_name == ";": class_name = "s"
    elif folder_name == "_": class_name = "g"
    elif folder_name == "+": class_name = "u"
    elif folder_name == "=": class_name = "o"
    # Eğer klasör adı 'C' ise ama sen listede 'c' (küçük) yaptıysan eşleştir:
    elif folder_name == "C": class_name = "C" # Listende büyük C de var.
    
    try:
        class_id = classes.index(class_name)
    except ValueError:
        print(f"⚠️ {folder_name} (Hedef: {class_name}) sinif listesinde bulunamadi, atlaniyor.")
        continue

    print(f"📂 {folder_name} klasörü işleniyor (Sınıf ID: {class_id})...")

    # Klasör içindeki resimleri tara
    for img_name in os.listdir(folder_path):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')): continue
        
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        if image is None: continue
        
        h, w, _ = image.shape
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_max, y_max = 0, 0
                x_min, y_min = w, h
                
                # Elin 21 noktasını tara, en uçları bul
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max: x_max = x
                    if x < x_min: x_min = x
                    if y > y_max: y_max = y
                    if y < y_min: y_min = y

                # YOLO Formatına Dönüştürme (0-1 arası normalize)
                x_center = (x_min + x_max) / 2 / w
                y_center = (y_min + y_max) / 2 / h
                width = (x_max - x_min) / w
                height = (y_max - y_min) / h

                # .txt dosyasını oluştur
                txt_name = os.path.splitext(img_name)[0] + ".txt"
                with open(os.path.join(LABEL_DIR, txt_name), "w") as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("✅ Tüm işlem başarıyla tamamlandı! 'yoloData' klasörünü kontrol edebilirsin.")