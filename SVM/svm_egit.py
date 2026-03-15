import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler 
import joblib


print("Veri yükleniyor...")
df = pd.read_csv('isaret_dili_koordinat_veriseti.csv')


df = df.dropna()  #Mediapipe ın datasetimizde el bulamadığı için boş bıraktığı satırları temizledik.


X = df.drop('label', axis=1) #X: Modelin görmesi gereken el şekli.
y = df['label']   # y: Modelin tahmin etmesi gereken harf ismi.


# Standartlaştırma : Bu SVM algoritmasının harfler arasındaki mesafeyi çok daha hassas hesaplamasını sağlar 
scaler = StandardScaler()   # Ham koordinat verileri ortalaması 0 ,standart sapması 1 olacak şekilde dönüştürdük.
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
# veri setimizi %90 eğitim %10 test olarak böldük.
# ilk önce %20 eğitim %10 test olarak denedik.İkinci durumda daha çok verim aldık.


print("Model eğitiliyor...")
model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True) 
model.fit(X_train, y_train)

# Kernel=rbf veriyi yüksek boyutlu uzaya taşıyarak karmaşık el şekillerini birbirinden ayırır.
# C= 10.0 Modelin hata yapma toleransı
# probability gerçek tahminde yüzde kaç emin olunduğunu görmemizi sağlıyor.


y_pred = model.predict(X_test)    #model test verileri içinden tahminlerde bulunuyor 
dogruluk = accuracy_score(y_test, y_pred) # gerçek harflerle karşılaştırarak başarı yüzdesini bulduk.

print("-" * 30)
print(f"Yeni Model Başari Orani: %{dogruluk * 100:.2f}")
print("-" * 30)
print("Siniflandirma Raporu:")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)

plt.title('TİD SVM Modeli Karişiklik Matrisi')
plt.xlabel('Tahmin Edilen Harf')
plt.ylabel('Gerçek Harf')
plt.show() # Grafiği ekranda gösterir


joblib.dump(model, 'tid_model_svm.pkl')   # Modeli ve Scaler i kaydettik joblib kütüphanesi ile
joblib.dump(scaler, 'scaler.pkl')         # Canlı tahmin sırasında veriler ilk önce Scalerden geçiyor. Normalizasyon yapar.
print("\nModel ve Scaler başariyla kaydedildi!")
