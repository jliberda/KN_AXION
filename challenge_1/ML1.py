# ============================================================
# IMPORTY
# ============================================================
import pandas as pd

# Funkcja, która losowo miesza wiersze i dzieli je na 80% danych treningowych oraz 20% danych testowych.
from sklearn.model_selection import train_test_split

# Import modelu Logistic Regression do przewidywania, czy pacjent ma chorobę serca.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# do EDA
import matplotlib.pyplot as plt
import seaborn as sns # heatmapa

# ============================================================
# 1. EDA - EXPLORATORY DATA ANALYSIS
# ============================================================

df = pd.read_csv("heart.csv") # df - pandas Data Frame

# print("(Ilość wierszy, ilość kolumn): ", df.shape)

# print("\ntypy danych, czy gdzieś jest null?, kolumny, wiersze:")
# print(df.info())

# print("\nstatystyki:")
# print(df.describe())

# print("\nbrakujące wartości:")
# print(df.isnull().sum())

# print("\nduplikaty:")
# print(df.duplicated().sum()) # 723 DUPLIKATY!!!
df = df.drop_duplicates() # usuwa duplikaty
# print(df.shape)
# print(df.duplicated().sum())

# print("\nhistogramy:")
# print(df.hist())

# print("\nheatmapa:")
# sns.heatmap(df.corr(numeric_only=True), annot=True)
# plt.figure(figsize=(12, 10))
# sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.1f')
# plt.savefig("heatmap.png")
# print("zapisana w folderze jako 'heatmap.png'")

# ============================================================
# 2. PRZYGOTOWANIE DANYCH
# ============================================================

#PYTANIE: Czy usunięcie trestbps (resting blood pressure) pogorszy przewidywania modelu AI?
print('HIPOTEZA: Model będzie gorzej przewidywał czy dany pacjent ma chorobę serca gdy usunę kolumnę trestbps.')

features = df.drop('target', axis=1) # tabela bez kolumny 'target', axis=0 (wiersze), axis=1 (kolumny). DataFrame.
target = df['target'] # tylko kolumna 'target'. Series.
features_bez = df.drop(['target', 'trestbps'], axis=1) # bez (kolumny trestbps)
target_bez = df['target']

f_train, f_test, t_train, t_test = train_test_split(features, target, test_size=0.2, random_state=2137) # hehe
f_train_bez, f_test_bez, t_train_bez, t_test_bez = train_test_split(features_bez, target_bez, test_size=0.2, random_state=2137)

# ============================================================
# 3. TRENING MODELI
# ============================================================
model = LogisticRegression(max_iter=1000) # max_iter - maksymalna ilość popraw wag przez model.
model_bez = LogisticRegression(max_iter=1000)

model.fit(f_train, t_train)
model_bez.fit(f_train_bez, t_train_bez)
# convergence - zbieżność, model znalazł dobre dopasowania wag.

# ============================================================
# 4. TESTOWANIE I METRYKI (OCENA MODELU)
# ============================================================
predictions = model.predict(f_test)
predictions_bez = model_bez.predict(f_test_bez)

accuracy = accuracy_score(t_test, predictions) # ile % pacjentów zostało zdiagnozowanych poprawnie, odsetek wszystkich przewidywań, które były poprawne.
accuracy_bez = accuracy_score(t_test_bez, predictions_bez)

print('\n=== MODEL ZE WSZYSTKIMI CECHAMI ===')
print('Accuracy:', accuracy)
print(classification_report(t_test, predictions))

print('\n=== MODEL BEZ CECHY trestbps ===')
print('Accuracy:', round(accuracy_bez, 4))
print(classification_report(t_test_bez, predictions_bez))

# Precission - jaki odsetek przewidywań konkretnych klas był poprawny.
# Recall - jaki odsetek rzeczywistych przypadków danej klasy model wykrył.
# F1-score - połączenie Precission i Recall. Przydaje się do porównywania modeli.
# Support - liczba przykładów w zbiorze, wierszy.

# Accuracy - Accuracy ale zaokrąglone
# Macro avg – średnia metryk dla obu klas, każda klasa ma taką samą wagę.
# Weighted avg – średnia metryk dla obu klas, ważona liczbą przykładów w każdej klasie.


# ============================================================
# 5. PORÓWNANIE WYNIKÓW
# ============================================================

roznica = accuracy - accuracy_bez

print('\n=== PORÓWNANIE ===\n')
print('Accuracy z trestbps:', round(accuracy*100,2))
print('Accuracy bez trestbps:', round(accuracy_bez*100,2))
print('Różnica:', round(roznica*100,2), '\n')

if accuracy_bez < accuracy:
    print('Po usunięciu trestbps accuracy spadło. Hipoteza potwierdzona.')
elif accuracy_bez > accuracy:
    print('Po usunięciu trestbps accuracy wzrosło. Hipoteza niepotwierdzona.')
else:
    print('Accuracy nie zmieniło się po usunięciu trestbps.')



# Materiały pomocnicze:
# Dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# EDA Tutorial: https://www.geeksforgeeks.org/data-analysis/exploratory-data-analysis-in-python/
# Tutorial: https://pytutorial.com/how-to-make-an-ai-in-python-step-by-step/
