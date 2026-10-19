# ============================================================
# IMPORTY
# ============================================================
import pandas as pd

# Funkcja, która losowo miesza wiersze i dzieli je na 80% danych treningowych oraz 20% danych testowych.
from sklearn.model_selection import train_test_split

# Import modelu Logistic Regression do przewidywania, czy pacjent ma chorobę serca.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# 1. EDA - EXPLORATORY DATA ANALYSIS
# ============================================================
with open('heart.csv') as plik:
    heart = [x.strip().split(',') for x in plik]

# zlicz ile jest osób z chorobą serca (c), a ile jest osób zdrowych (z).
z = c = 0

# sprawdź czy dane są kompletne.
puste = 0

# oblicz średni wiek dla osób chorych i zdrowych.
c_wiek = z_wiek = 0

# najmłodszy i najstarszy wiek osoby
c_wiek_max = z_wiek_max = 0
c_wiek_min = z_wiek_min = float('inf')

for age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target in heart[1:]:
    if '' in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target]:
        puste += 1
    if target == '1':
        c += 1
        c_wiek += int(age)
        c_wiek_min = min(c_wiek_min, int(age))
        c_wiek_max = max(c_wiek_max, int(age))
    else:
        z += 1
        z_wiek += int(age)
        z_wiek_min = min(z_wiek_min, int(age))
        z_wiek_max = max(z_wiek_max, int(age))

print('=== EDA ===')
print('Liczba osób chorych:', c)
print('Liczba osób zdrowych:', z)
print('Wiersze z brakami danych:', puste)
print('Średni wiek osób chorych:', round(c_wiek/c, 2))
print('Średni wiek osób zdrowych:', round(z_wiek/z, 2))
print('Wiek osób chorych:', str(c_wiek_min)+'-'+str(c_wiek_max))
print('Wiek osób zdrowych:', str(z_wiek_min)+'-'+str(z_wiek_max))


# ============================================================
# 2. PRZYGOTOWANIE DANYCH
# ============================================================
df = pd.read_csv("heart.csv") # df - pandas Data Frame

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
print('Accuracy z trestbps:', accuracy)
print('Accuracy bez trestbps:', accuracy_bez)
print('Różnica:', roznica, '\n')

if accuracy_bez < accuracy:
    print('Po usunięciu trestbps accuracy spadło. Hipoteza potwierdzona.')
elif accuracy_bez > accuracy:
    print('Po usunięciu trestbps accuracy wzrosło. Hipoteza niepotwierdzona.')
else:
    print('Accuracy nie zmieniło się po usunięciu trestbps.')



# Materiały pomocnicze:
# Dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# Tutorial: https://pytutorial.com/how-to-make-an-ai-in-python-step-by-step/