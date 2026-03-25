import cv2
from ultralytics import YOLO
import numpy as np

# Google Colab'de oluşturup indirdiğimiz modeli kullandık
model_path = r'C:\Users\MSI\Desktop\Python\Turk-Isaret-Dili-Algilama\YOLO\best.pt'
model = YOLO(model_path)


cap = cv2.VideoCapture(0)

print("Kamera açiliyor... Kapatmak için 'q' tuşuna basin.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    
    results = model(frame, conf=0.25, device='cpu', verbose=False) # Kullanılan bilgisayarın ekran kartı Python kütüphanesi ile uyumlu olmadığından Cpu kullandık

    
    panel_width = 300
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
    
    
    combined_frame = np.hstack((frame, panel))


    main_res = "-"
    main_conf = 0.0

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # En yüksek olasılıklı olanı aldık
        box = results[0].boxes[0]
        main_res = model.names[int(box.cls)].upper()
        main_conf = float(box.conf)
        
        # YOLO'nun kendi kutularını kamera görüntüsü üzerine çizdik
        frame = results[0].plot()
        
        combined_frame = np.hstack((frame, panel))

    
    start_x = w + 20 # Siyah panelin başladığı yer
    cv2.putText(combined_frame, "TAHMIN PANELI", (start_x, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(combined_frame, "Harf:", (start_x, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.putText(combined_frame, main_res, (start_x + 20, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5) 
    
    cv2.putText(combined_frame, "Olasilik:", (start_x, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.putText(combined_frame, f"%{main_conf*100:.1f}", (start_x + 10, 330), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    cv2.putText(combined_frame, "Hasan Talha", (start_x, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

   
    cv2.imshow("Test Ekrani", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()