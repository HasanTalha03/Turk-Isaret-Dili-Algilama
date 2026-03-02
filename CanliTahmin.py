import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque, Counter
import time


model = joblib.load('tid_model_svm.pkl') # Daha önce kaydettiğimiz svm modelini ve scaler i çağırıyoruz
scaler = joblib.load('scaler.pkl')


tahmin_gecmisi = deque(maxlen=20)
olusturulan_metin = ""
son_eklenen_harf = ""
son_ekleme_zamani = time.time()
bekleme_suresi = 4.5  
mod = "MENU" 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Renk Paleti 
BG_COLOR = (20, 20, 20)      
ACCENT_COLOR = (0, 255, 127) 
TEXT_COLOR = (240, 240, 240) 
BAR_COLOR = (255, 191, 0)   

def draw_menu(img, h, w):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    
    cv2.putText(img, "TID TERCUMAN ANA MENU", (w//2 - 220, h//2 - 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, ACCENT_COLOR, 2)
    
    # Buton Alanları
    cv2.rectangle(img, (w//2 - 250, h//2 - 40), (w//2 + 250, h//2 + 20), (40, 40, 40), -1)
    cv2.putText(img, "[1] TEST MODU ", (w//2 - 210, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
    
    cv2.rectangle(img, (w//2 - 250, h//2 + 40), (w//2 + 250, h//2 + 100), (40, 40, 40), -1)
    cv2.putText(img, "[2] METIN MODU ", (w//2 - 210, h//2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
    
    cv2.putText(img, "Cikis: [ESC]", (w//2 - 60, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if mod == "MENU":
        draw_menu(frame, h, w)
        cv2.imshow('TID Tercuman', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'): mod = "TEST"
        elif key == ord('2'): mod = "METIN"
        elif key == 27: break
        continue

   
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)
    coords = [0.0] * 126
    current_prediction = ""
    max_prob = 0

    if results.multi_hand_landmarks:
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            if idx >= 2: break
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            base_x, base_y, base_z = hand_lms.landmark[0].x, hand_lms.landmark[0].y, hand_lms.landmark[0].z
            for i, lm in enumerate(hand_lms.landmark):
                start_idx = idx * 63 + i * 3
                coords[start_idx] = lm.x - base_x
                coords[start_idx+1] = lm.y - base_y
                coords[start_idx+2] = lm.z - base_z
        
        coords_scaled = scaler.transform([coords])
        probs = model.predict_proba(coords_scaled)[0]   #predict_proba modelin her harf için verdiği olasılığı hesaplar 
        max_prob = np.max(probs)
        raw_val = model.classes_[np.argmax(probs)]
        
        if max_prob > 0.75:
            tahmin_gecmisi.append(raw_val)
        if tahmin_gecmisi:
            current_prediction = Counter(tahmin_gecmisi).most_common(1)[0][0]
    else:
        # Eller sahnede değilse geçmişi ve zamanlayıcıyı sıfırlıyoruz
        tahmin_gecmisi.clear()
        son_eklenen_harf = ""

    
    if mod == "TEST":
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"TEST MODU: {current_prediction} (%{int(max_prob*100)})", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, ACCENT_COLOR, 2)
        cv2.putText(frame, "Menü: [M]", (w-120, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

    elif mod == "METIN":
     
        panel_w = 400
        combined_img = np.full((h, w + panel_w, 3), BG_COLOR, dtype=np.uint8)
        combined_img[0:h, 0:w] = frame
        
        # Kelime Mantığı
        if current_prediction:
            simdi = time.time()
            if current_prediction == son_eklenen_harf:
                gecen_sure = simdi - son_ekleme_zamani
                progress = min(gecen_sure / bekleme_suresi, 1.0)
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
        cv2.putText(combined_img, "TID TRANSLATOR", (px, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, ACCENT_COLOR, 2)
        cv2.line(combined_img, (px, 75), (w + panel_w - 30, 75), (100, 100, 100), 1)
        
       
        status_color = ACCENT_COLOR if results.multi_hand_landmarks else (0, 0, 255)
        cv2.circle(combined_img, (px + 10, 110), 7, status_color, -1)
        cv2.putText(combined_img, "SISTEM AKTIF" if results.multi_hand_landmarks else "EL BEKLENIYOR", 
                    (px + 30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

       
        cv2.putText(combined_img, "OLUSTURULAN METIN", (px, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        y_start = 290
        for i in range(0, len(olusturulan_metin), 12):
            cv2.putText(combined_img, olusturulan_metin[i:i+12], (px, y_start), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, TEXT_COLOR, 2)
            y_start += 50

       
        cv2.rectangle(combined_img, (w + 20, h - 140), (w + panel_w - 20, h - 20), (30, 30, 30), -1)
        cv2.putText(combined_img, "[M] Ana Menu", (px, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(combined_img, "[G] Geri Al", (px, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(combined_img, "[B] Bosluk", (px, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(combined_img, "[S] Temizle", (px, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
       
        if current_prediction:
            cv2.rectangle(combined_img, (20, 20), (180, 80), (0, 0, 0), -1)
            cv2.putText(combined_img, current_prediction, (40, 70), cv2.FONT_HERSHEY_DUPLEX, 1.8, ACCENT_COLOR, 3)

        frame = combined_img

    cv2.imshow('TID Tercuman', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'): mod = "MENU"
    elif key == ord('g') and mod == "METIN": olusturulan_metin = olusturulan_metin[:-1]
    elif key == ord('s') and mod == "METIN": olusturulan_metin = ""
    elif key == ord('b') and mod == "METIN": olusturulan_metin += " "
    elif key == 27: break

cap.release()
cv2.destroyAllWindows()