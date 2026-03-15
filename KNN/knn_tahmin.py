import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque, Counter
import time
import os


model_path = r'C:\Users\MSI\Desktop\Python\Turk-Isaret-Dili-Algilama\model_kayit\KNN\isaret_dili_knn.pkl'
encoder_path = r'C:\Users\MSI\Desktop\Python\Turk-Isaret-Dili-Algilama\model_kayit\KNN\label_encoder_knn.pkl'

knn = joblib.load(model_path)
le = joblib.load(encoder_path)


tahmin_gecmisi = deque(maxlen=20)
olusturulan_metin = ""
son_eklenen_harf = ""
son_ekleme_zamani = time.time()
bekleme_suresi = 4.5  
mod = "MENU" 


BG_COLOR = (20, 20, 20)
ACCENT_COLOR = (0, 255, 127) 
TEXT_COLOR = (240, 240, 240)
BAR_COLOR = (255, 191, 0) 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) # detection kısmı elleri tespit etmedeki sınır değer
mp_draw = mp.solutions.drawing_utils  # el üzerindeki çizgileri çizmemize yarıyor

cap = cv2.VideoCapture(0) # kamerayı kendi bilgisayar kameramız olarak ayarladık



def draw_menu(img, h, w):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.putText(img, "KNN TID TERCUMAN MENU", (w//2 - 250, h//2 - 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, ACCENT_COLOR, 2)
    cv2.rectangle(img, (w//2 - 250, h//2 - 40), (w//2 + 250, h//2 + 20), (40, 40, 40), -1)
    cv2.putText(img, "[1] TEST MODU ", (w//2 - 210, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
    cv2.rectangle(img, (w//2 - 250, h//2 + 40), (w//2 + 250, h//2 + 100), (40, 40, 40), -1)
    cv2.putText(img, "[2] METIN MODU ", (w//2 - 210, h//2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if mod == "MENU":
        draw_menu(frame, h, w)
        cv2.imshow('TID Tercuman KNN', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'): mod = "TEST"
        elif key == ord('2'): mod = "METIN"
        elif key == 27: break
        continue

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # renk uzayını mediapipe ın anlayacağı şekilde değiştiriyoruz
    results = hands.process(rgb_img)   # aldığımız görüntüleri mediapipe a göndermek için kullanıyoruz
    coords = [0.0] * 126  # iki el ile oluşturduğumuz karakterler olduğu için 126 seçtik (2 el x 21 nokta x 3 eksen)
    current_prediction = ""   # her döngü başında tahmini sıfırlıyoruz
    max_prob = 0  # modelin tahmin değerini de sıfırlıyoruz

    if results.multi_hand_landmarks:  # elin olup olmadığını kontrol ediyoruz
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):    
            if idx >= 2: break    # sadece elleri al gerisini alma 
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            base_x, base_y, base_z = hand_lms.landmark[0].x, hand_lms.landmark[0].y, hand_lms.landmark[0].z  # elimizi ekranın her yerinde tanıyabilmek için gerekli
            for i, lm in enumerate(hand_lms.landmark):
                start_idx = idx * 63 + i * 3
                coords[start_idx] = lm.x - base_x
                coords[start_idx+1] = lm.y - base_y
                coords[start_idx+2] = lm.z - base_z
        # parmakların bilek ile arasındaki mesafeye göre işlem yapıyoruz bu sayede elin ekranda nerede olduğu önemli değil 

        probs = knn.predict_proba([coords])[0]
        max_prob = np.max(probs)
        raw_val = knn.classes_[np.argmax(probs)]
        
        if max_prob > 0.70:
            tahmin_gecmisi.append(le.inverse_transform([raw_val])[0])
        
        if tahmin_gecmisi:
            current_prediction = Counter(tahmin_gecmisi).most_common(1)[0][0]
    else:
        tahmin_gecmisi.clear()
        son_eklenen_harf = ""

    # --- ARAYÜZ ÇİZİMLERİ ---
    if mod == "TEST":
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"TEST: {current_prediction} (%{int(max_prob*100)})", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, ACCENT_COLOR, 2)
        cv2.putText(frame, "Menu: [M]", (w-120, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

    elif mod == "METIN":
        panel_w = 400
        combined_img = np.full((h, w + panel_w, 3), BG_COLOR, dtype=np.uint8)
        combined_img[0:h, 0:w] = frame
        
        # Sol üstte anlık harf ve yüzde kutusu
        if current_prediction:
            cv2.rectangle(combined_img, (10, 10), (220, 90), (0, 0, 0), -1)
            cv2.putText(combined_img, f"{current_prediction} %{int(max_prob*100)}", (25, 65), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, ACCENT_COLOR, 3)

            simdi = time.time()
            if current_prediction == son_eklenen_harf:
                gecen_sure = simdi - son_ekleme_zamani
                progress = min(gecen_sure / bekleme_suresi, 1.0)
                # İlerleme Barı
                cv2.rectangle(combined_img, (w + 40, 180), (w + panel_w - 40, 190), (50, 50, 50), -1)
                cv2.rectangle(combined_img, (w + 40, 180), (w + 40 + int(progress * (panel_w - 80)), 190), BAR_COLOR, -1)
                
                if gecen_sure >= bekleme_suresi:
                    olusturulan_metin += current_prediction
                    son_ekleme_zamani = simdi
                    tahmin_gecmisi.clear()
            else:
                son_eklenen_harf = current_prediction
                son_ekleme_zamani = simdi

        
        px = w + 30
        cv2.putText(combined_img, "KNN TERCUMAN", (px, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, ACCENT_COLOR, 2)
        cv2.putText(combined_img, "OLUSTURULAN METIN", (px, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        y_start = 290
        for i in range(0, len(olusturulan_metin), 12):
            cv2.putText(combined_img, olusturulan_metin[i:i+12], (px, y_start), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, TEXT_COLOR, 2)
            y_start += 50

        
        cv2.rectangle(combined_img, (w + 20, h - 140), (w + panel_w - 20, h - 20), (30, 30, 30), -1)
        cv2.putText(combined_img, "[M] Menu | [G] Geri | [B] Bosluk", (px, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
        cv2.putText(combined_img, "[S] Sil  | [ESC] Cikis", (px, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        frame = combined_img

    cv2.imshow('TID Tercuman KNN', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'): mod = "MENU"
    elif key == ord('g') and mod == "METIN": olusturulan_metin = olusturulan_metin[:-1]
    elif key == ord('s') and mod == "METIN": olusturulan_metin = ""
    elif key == ord('b') and mod == "METIN": olusturulan_metin += " "
    elif key == 27: break

cap.release()
cv2.destroyAllWindows()