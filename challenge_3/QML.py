# ============================================================
# IMPORTY
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from  qiskit import QuantumCircuit 
from  qiskit.quantum_info import SparsePauliOp 
from  qiskit.primitives import StatevectorEstimator 
from qiskit.circuit import ParameterVector

# do trenowania
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ile trening zajmuje?
from tqdm import tqdm

# ==============================================================
# 0. Ustawienia
# ==============================================================
n = 8 # num of features
layers = 2 # do ansatzu
epochs = 10 # epochs
lr = 0.05 # learning rate
batch_size = 32 # do dzielenia wierszy na grupy

# ==============================================================
# 1. EDA - EXPLORATORY DATA ANALYSIS
# ==============================================================

df = pd.read_csv("energy_efficiency_data.csv") # df - pandas Data Frame

# print("(Ilość wierszy, ilość kolumn): ", df.shape)

# print("\n\ntypy danych, czy gdzieś jest null?, kolumny, wiersze:")
# print(df.info())

# print("\n\nstatystyki:")
# print(df.describe())

# print("\n\nbrakujące wartości:")
# print(df.isnull().sum())

# print("\n\nduplikaty:")
# print(df.duplicated().sum())

# print("\nheatmapa:")
# plt.figure(figsize=(12, 16)) # obszar, figsize(width, height) rozmiar w calach.
# sns.heatmap(df.corr(), annot=True, fmt=".2f")
# # corr() - tabela, jak silnie ane 2 cechy są z sobą skorelowane
# # annot=True - annotations, adnotacje, liczby określające korelacje.
# # fmt - format
# plt.subplots_adjust(top=0.90, bottom=0.25, left=0.20, right=1.0) # adjust boarders
# plt.show()

# ==============================================================
# 2. Wybór cech mocno skorelowanych.
# ==============================================================

# Features:
# Relative_Compactness - jak zbity jest budynek. Im bardziej jak kostka, tym bardziej zbity.
# Surface_Area - powierzchnia na zewnątrz budynku (ściany + dach).
# Wall_Area, Roof_Area - wiadomo.
# Overall_Height - (3,5 m) albo (7 m).
# Orientation - w którą stronę budynek jest obrócony. Praktycznie nie ma wpływu na wynik.
# Glazing_Area - ile jest okien.
# Glazing_Area_Distribution - gdzie są okna, równomiercie, od północy, itd...

# Dwa targety:
# Heating_Load - ile energii budynek potrzebuje na ogrzewanie.
# Cooling_Load - ile energii potrzebuje na chłodzenie.

# Na podstawie heatmapy z EDA, możemy stwierdzić, że cechy:
# -> Relative_Compactness i Surface_Area   : -0.99
# -> Roof_Area i Overall_Height            : -0.97
# -> Surface_Area i Roof_Area              : +0.88
# -> Relative_Compactness i Roof_Area      : -0.87
# -> Surface_Area i Overall_Height         : -0.86
# -> Relative_Compactness i Overall_Height : +0.83
# są mocno skorelowane.

# Wybieram: Relative_Compactness+Surface_Area -> Surface_Area+Roof_Area  -> Roof_Area+Overall_Height

# ==============================================================
# 3. Funkcje pomocnicze
# ==============================================================

# 3.1 Entanglement ---------------------------------------------
def entanglement(topology, qubits):
    if topology == "linear":
        return [(i, i+1) for i in range(qubits-1)]
    elif topology == "ring":
        return [(i, i+1) for i in range(qubits-1)] + [(0, qubits-1)]
    elif topology == "star":
        return [(0, i) for i in range(1,qubits)]
    else: #topology = "full"
        return [(i, j) for i in range(qubits-1) for j in range(i+1, qubits)]

def build_base_circuit(topology, x_param, th_param):
    qc = QuantumCircuit(n)

    # Angle Encoding
    for i in range(n):
        qc.ry(x_param[i], i)

    # Ansatz
    for layer in range(layers):

        for i in range(n):
            qc.ry(th_param[layer*n+i], i)

        for control, target in entanglement(topology, n):
            qc.cx(control, target)

    return qc

def predict(features, topology, th):

    estimator = StatevectorEstimator()
    observable = SparsePauliOp.from_sparse_list(
        [("Z", [0], 1.0)], num_qubits=n
    )

    x_param = ParameterVector('x', n)
    th_param = ParameterVector('th', len(th))

    qc = build_base_circuit(topology, x_param, th_param)

    # Podstawiamy wyuczone parametry theta
    bound_circuit = qc.assign_parameters(dict(zip(th_param, th)))

    # Przekazujemy wszystkie budynki jako jedną tablicę
    parameter_values = features.to_numpy()
    
    job = estimator.run([(bound_circuit, observable, parameter_values)])

    return np.asarray(job.result()[0].data.evs).flatten()

def train(f_train, t_train, topology, initial_th, batch_size=32):
    th = initial_th.copy() # thety

    #skalowanie wartości do [-1,1]
    mn = t_train.min()
    mx = t_train.max()
    
    losses = []
    
    for epoch in (epoch_progress := tqdm(range(epochs), desc="Training " + topology)):
        
        # Losowa kolejność danych w każdej epoce
        indices = np.random.permutation(len(f_train))
        epoch_loss = 0

        # Podział danych na batche
        for start in range(0, len(f_train), batch_size):
            batch_indices = indices[start:start + batch_size]

            f_batch = f_train.iloc[batch_indices]
            t_batch = t_train.iloc[batch_indices]

            t_scaled = 2*(t_batch.values-mn) / (mx-mn)-1

            predictions = predict(f_batch, topology, th)
            gradients = []

            for j in range(len(th)):

                # Parametr w górę
                th_plus = th.copy()
                th_plus[j] += np.pi / 2
                pred_plus = predict(f_batch, topology, th_plus)

                # parametr w dół
                th_minus = th.copy()
                th_minus[j] -= np.pi / 2
                pred_minus = predict(f_batch, topology, th_minus)

                # Obliczamy gradient
                derivative = (pred_plus - pred_minus) / 2
                gradient = np.mean(2*(predictions-t_scaled)*derivative) # jak zmiana parametru wpłynie na MSE

                gradients.append(gradient)

            # Poprawiamy parametry
            th -= lr * np.array(gradients)

            # Obliczamy błąd dla tego batcha (na zaktualizowanych parametrach)
            predictions_after = predict(f_batch, topology, th)
            loss = np.mean((predictions_after-t_scaled) ** 2)
            epoch_loss += loss * len(batch_indices)

        # Średnie MSE z całej epoki
        epoch_loss /= len(f_train)
        losses.append(epoch_loss)   

        epoch_progress.set_postfix(MSE=round(epoch_loss, 6))

    return th, mn, mx, losses


# ==============================================================
# 4. Przygotowanie danych
# ==============================================================

f = df[["Relative_Compactness", "Surface_Area", "Roof_Area", "Overall_Height"]+[x for x in df.columns if x not in ["Relative_Compactness", "Surface_Area", "Roof_Area", "Overall_Height", "Heating_Load", "Cooling_Load"]]].copy()
t = df["Heating_Load"] # w datasecie mamy 2 targety, wybieram Heating_Load do testów i treningu
f_scrambled = f.sample(frac=1, axis=1, random_state=2137).copy()

f_train, f_test, t_train, t_test = train_test_split(f, t, test_size=0.2, random_state=2137)
f_train_scrambled, f_test_scrambled, t_train_scrambled, t_test_scrambled = train_test_split(f_scrambled, t, test_size=0.2, random_state=2137)

# Skalowanie danych do przedziału [0, pi] ----------------------
for col in f_train.columns:
    mx = max(f_train[col])
    mn = min(f_train[col])

    f_train[col] = (f_train[col]-mn) / (mx-mn) * np.pi
    f_test[col] = (f_test[col]-mn) / (mx-mn) * np.pi

for col in f_train_scrambled.columns:
    mx = max(f_train_scrambled[col])
    mn = min(f_train_scrambled[col])

    f_train_scrambled[col] = (f_train_scrambled[col]-mn) / (mx-mn) * np.pi
    f_test_scrambled[col] = (f_test_scrambled[col]-mn) / (mx-mn) * np.pi

# ==============================================================
# 5. Trenowanie i porównanie topologii
# ==============================================================

topologies = ["linear", "ring", "star", "full"]
results = {}
results_scrambled = {}
all_losses = {}
np.random.seed(2137)
initial_th = np.random.uniform(-np.pi, np.pi, layers * n)

for topology in topologies:

    print("\nTrenowanie: ", topology)
    th, mn, mx, losses = train(f_train, t_train, topology, initial_th)
    all_losses[topology] = losses

    predictions = predict(f_test, topology, th)
    # Skala
    predictions = (predictions+1) / 2*(mx-mn) + mn

    # Metryki
    mse = mean_squared_error(t_test, predictions)
    mae = mean_absolute_error(t_test, predictions)
    r2 = r2_score(t_test, predictions)

    results[topology] = [mse, mae, r2]

    print("MSE:", mse)
    print("MAE:", mae)
    print("R2:", r2)

print("\n\n===== WYNIKI - CECHY UPORZĄDKOWANE =====")

for topology, metrics in results.items():
    print("\n", topology)
    print("MSE:", metrics[0]) # Mean Squared Error (pred-poprawny_wynik)**2
    print("MAE:", metrics[1]) # Mean Absolute Error (pred-poprawny_wynik)
    print("R2:", metrics[2]) # R squared 
# R2 = 1   -> model przewiduje idealnie
# R2 = 0   -> model jest lepszy niż śrepe zgadywanie 50/50
# R2 < 0   -> model jest gorszy od 50/50

# Drugi eksperyment - losowa kolejność cech
np.random.seed(2137)
initial_th = np.random.uniform(-np.pi, np.pi, layers * n)

for topology in topologies:

    print("\nTrenowanie - scrambled: ", topology)
    th, mn, mx, losses = train(f_train_scrambled, t_train_scrambled, topology, initial_th)

    predictions = predict(f_test_scrambled, topology, th)
    # Skala
    predictions = (predictions+1) / 2*(mx-mn) + mn

    # Metryki
    mse = mean_squared_error(t_test_scrambled, predictions)
    mae = mean_absolute_error(t_test_scrambled, predictions)
    r2 = r2_score(t_test_scrambled, predictions)

    results_scrambled[topology] = [mse, mae, r2]

    print("MSE:", mse)
    print("MAE:", mae)
    print("R2:", r2)

print("\n\n===== WYNIKI - CECHY SCRAMBLED =====")

for topology, metrics in results_scrambled.items():
    print("\n", topology)
    print("MSE:", metrics[0])
    print("MAE:", metrics[1])
    print("R2:", metrics[2])

# Wykres uczenia
plt.figure(figsize=(8, 5))

for topology, losses in all_losses.items():
    plt.plot(range(1, len(losses)+1), losses, label=topology)

plt.xlabel("Epoka")
plt.ylabel("MSE")
plt.title("Proces uczenia - cechy uporządkowane")
plt.legend()
plt.grid()
plt.show()

# ==============================================================
# 6. WNIOSKI
# ==============================================================
# R2 (10 epok):
#                   linear   ring    star    full
#   uporządkowane    0.47   -0.05    0.38    0.37
#   scrambled        0.71    0.65    0.67    0.67
#
# Hipoteza obalona częściowo.
#
# Linear ma najlepszy wynik w obu układach, a zużywa mniej zasobów
# niż ring i full.
#
# Gdyby liczyło się to, że skorelowane cechy stoją obok siebie, uporządkowane 
# powinny wypaść lepiej niż scrambled. Wyszło odwrotnie: scrambled jest lepszy
# dla każdej topologii, a linear wygrywa też w nim.