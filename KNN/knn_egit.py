import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# verimizi yukluyoruz
csv_path = r'C:\Users\MSI\Desktop\Python\Turk-Isaret-Dili-Algilama\isaret_dili_koordinat_veriseti.csv'
df = pd.read_csv(csv_path)   # pandas kutuphanesi ile okuyoruz

# verilerimizde etiketin sonda olduğunu belirtiyoruz ki knn duzgun calıssın
X_df = df.iloc[:, :-1] 
y = df.iloc[:, -1].values 

# veri temizliği
X_df = X_df.apply(pd.to_numeric, errors='coerce')  # bir veride geçersiz karakter varsa onu NaN hala getirir
mask = X_df.notna().all(axis=1)   
X = X_df[mask].values                              # yalnızca tamamen sağlam olan kısımları dahil ediyoruz
y = y[mask]


le = LabelEncoder()                  # burada karakterlerimizi 'A','B' sayısal formata çeviriyoruz çünkü knn matematiksel bir algoritma
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)  # verilerimizi %80 eğitim %20 test olarak ayırıyoruz


knn = KNeighborsClassifier(n_neighbors=3)   # en iyi sonucu k=3 değerinde aldık 
knn.fit(X_train, y_train)                   # modeli eğitiyoruz


y_pred = knn.predict(X_test)

print("\n  --- Siniflandirma Raporu ---")

# Recall ve benzeri değerleri yazdiriyoruz

print(classification_report(y_test, y_pred, target_names=le.classes_))

# Karışıklık Matrisi Görselleştirme
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Greens')
plt.title('KNN - Turk Isaret Dili Karisiklik Matrisi')
plt.xlabel('Tahmin Edilen Harf')
plt.ylabel('Gercek Harf')
plt.show()

  # Modelimizi kaydettiğimiz kısım 

klasor_adi = r"C:\Users\MSI\Desktop\Python\Turk-Isaret-Dili-Algilama\model_kayit"
if not os.path.exists(klasor_adi):
    os.makedirs(klasor_adi)

knn_yolu = os.path.join(klasor_adi, 'isaret_dili_knn.pkl')
encoder_yolu = os.path.join(klasor_adi, 'label_encoder_knn.pkl')

joblib.dump(knn, knn_yolu)
joblib.dump(le, encoder_yolu)

print(f"\n Genel Başari Orani: %{knn.score(X_test, y_test) * 100:.2f}")      # en son modelin başarısının ne kadar olduğunu görmek için ekrana yazdırıyoruz
print(f" Modeller '{klasor_adi}' klasörüne tertemiz kaydedildi.")