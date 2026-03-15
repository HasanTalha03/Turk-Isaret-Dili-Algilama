import cv2
import mediapipe as mp
import csv
import os


ana_yol = r'C:\Users\MSI\Desktop\Python\BitirmeProjesi.py\TID_data'
output_csv = 'isaret_dili_koordinat_veriseti.csv'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1) # Mediapipe el algılama ayarları

# CSV Başlığı (42 nokta * 3 eksen = 126 sütun + label)
header = [f'pt{i}_{axis}' for i in range(42) for axis in ['x', 'y', 'z']] + ['label']
with open(output_csv, 'w', newline='') as f:
    csv.writer(f).writerow(header)

print(f"--- TÜM VERİ SETİ TARAMASI BAŞLADI ---")

# Ana yolun içindeki harf klasörlerini listele
harf_klasorleri = [d for d in os.listdir(ana_yol) if os.path.isdir(os.path.join(ana_yol, d))]

toplam_basarili = 0

for harf in harf_klasorleri:
    ana_klasör = os.path.join(ana_yol, harf)
    print(f"\nKlasör İşleniyor: {harf}")
    
    # JPG, JPEG ve PNG dosyalarının tamamını kapsayacak şekilde filtreleme işlemi
    dosyalar = [f for f in os.listdir(ana_klasör) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for dosya in dosyalar:
        tam_yol = os.path.join(ana_klasör, dosya)
        
        image = cv2.imread(tam_yol)
        if image is not None:
            # PNG dosyalarındaki şeffaflık varsa temizle
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_img)

            if results.multi_hand_landmarks:
                coords = [0.0] * 126
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    if idx >= 2: break
                    
                    
                    base_x = hand_lms.landmark[0].x
                    base_y = hand_lms.landmark[0].y
                    base_z = hand_lms.landmark[0].z
                    
                    for i, lm in enumerate(hand_lms.landmark):
                        base_idx = idx * 63 + i * 3
                        # Her noktayı bilekten olan farkına göre kaydet (Normalizasyon)
                        coords[base_idx] = lm.x - base_x
                        coords[base_idx+1] = lm.y - base_y
                        coords[base_idx+2] = lm.z - base_z
                
                # Veriyi CSV'ye ekle
                with open(output_csv, 'a', newline='') as f:
                    csv.writer(f).writerow(coords + [harf])
                
                toplam_basarili += 1
                if toplam_basarili % 10 == 0:
                    print(f"   > Toplam Kayıt Sayısı: {toplam_basarili}")

print(f"\n--- İŞLEM TAMAMLANDI ---")
print(f"Başarıyla Kaydedilen Toplam Veri: {toplam_basarili}")
