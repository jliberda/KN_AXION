import numpy as np
import random

# ===============================================================
# GATES
# ===============================================================
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
def Ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]])
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

# ===============================================================
# INPUT
# ===============================================================
while True:
    custom = int(input("Do you want to imput your own custom qubit?: (1 - yes, 0 - no)\n"))
    if custom != 0 and custom != 1:
        print("Choose 1 or 0.\n")
    else:
        break
print()

while custom:
    qubits = int(input("Choose num of qubits: 1 or 2: "))

    if qubits == 1:
        print("Enter a state. (float num)")
        a0 = float(input("0 amplitude: "))
        a1 = float(input("1 amplitude: "))
        state = np.array([a0, a1])
        break

    elif qubits == 2:
        print("Enter a state. (float num)")
        a00 = float(input("00 amplitude: "))
        a01 = float(input("01 amplitude: "))
        a10 = float(input("10 amplitude: "))
        a11 = float(input("11 amplitude: "))
        state = np.array([a00, a01, a10, a11])
        break

    else:
        print("You must choose 1 or 2.")

# ===============================================================
# FUNCTIONS
# ===============================================================
def apply_gate(gate, state):
    return np.dot(gate, state)

def normalise(state):
    total = np.sum(np.abs(state)**2)

    if np.isclose(total, 1):
        return state
    elif total != 0:
        # |A alfa|^2 + |A beta|^2 = 11
        # |A(alfa+beta)|^2 = 1
        # A = 1/sqrt(alfa+beta)
        return state / np.sqrt(total)
    else:
        print("Zero state cannot be normalised.")
        return state

def measure(state, qubit_num):
    if len(state) == 2:
        p0 = abs(state[0])**2
        r = random.random()
        if r <= p0:
            state[1] = 0
            return 0, normalise(state)
        state[0] = 0
        return 1, normalise(state)

    if qubit_num == 1:
        p0 = abs(state[0])**2 + abs(state[1])**2
        r = random.random()
        if r <= p0:
            state[2:] = 0
            return 0, normalise(state)
        state[:2] = 0
        return 1, normalise(state)

    if qubit_num == 2:
        p0 = abs(state[0])**2 + abs(state[2])**2
        r = random.random()
        if r <= p0:
            state[[1, 3]] = 0
            return 0, normalise(state)
        state[[0, 2]] = 0
        return 1, normalise(state)

# ===============================================================
# TESTING
# ===============================================================

# bramka Hadamarda
print("testing")
zero = np.array([1.0, 0.0])
print("H·H|0⟩ = ", apply_gate(H, apply_gate(H, zero)), " = |0⟩", "\n")



# Bella Circiut
qubit1 = qubit2 = np.array([1.0, 0.0]) # |00⟩

# H na pierwszym kubicie
qubit1 = apply_gate(H, qubit1)

# after H: 1/sqrt(2)(|00⟩ + |10⟩)
after_H = np.array([
    qubit1[0] * qubit2[0],
    qubit1[0] * qubit2[1],
    qubit1[1] * qubit2[0],
    qubit1[1] * qubit2[1]])

# CNOT
bell = apply_gate(CNOT, after_H)
print("|00⟩ --- Bella cuirciut ---> ", bell)

results = []
for _ in range(1000):
    state = bell.copy()
    result1, state = measure(state, 1)
    result2, state = measure(state, 2)
    results.append(str(result1) + str(result2))

print("Bell measurements:", set(results))

# ===============================================================
# NOTATKA
# ===============================================================
'''Na wykładzie nie rozumiałam jak dagger działa. Z kursu, który pojawił się na kanale baza-wiedzy dowiedziałam się, że dagger to operacja która najpierw sprzęża dany stan kubitu, następnie dokonuje
transpozycji. Pozwala ona na utworzenie inwersu macierzy U.

(A*)^T = (A^T)* = A†

* - sprzężenie
T - transpozycja

U U† = I

U† = U^-1

U - unitary matrix, macierz unitarna
I - macierz jednostkowa

Bramki kwantowe są bramkami unitarnymi, zatem dagger pozwala odwracać ich działanie.'''
