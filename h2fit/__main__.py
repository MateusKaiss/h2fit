import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import json

# ============================
# Inputs
# ============================
corrente = np.array([11.2, 22.6, 35.4, 49.9, 68.1])
temperatura = np.array([306.55, 315.85, 322.45, 326.25, 330.45])
vref = np.array([52.2, 51.1, 49.2, 47, 44.1])
pH2 = 0.65
pO2 = 0.21
N = 60

# ============================
# Model
# ============================
def calcular_vsaida(I, T, params):
    x1, x2, x3, x4, Rint, m, n = params

    v_nernst = 1.229 - 0.85e-3 * (T - 298.15) + 4.3085e-5 * T * (np.log(pH2) + 0.5 * np.log(pO2))
    perda_ativacao = -(x1 + x2 * T + x3 * T * np.log(pO2 / (5.08e6 * np.exp(-498 / T))) + x4 * T * np.log(I + 1e-6))
    perda_ohmica = Rint * I
    perda_concentracao = m * np.exp(n * I)

    v_saida = N * (v_nernst + perda_ativacao - perda_ohmica - perda_concentracao)
    return v_saida

# ============================
# Objective
# ============================
def func_objetivo(params):
    vs = calcular_vsaida(corrente, temperatura, params)
    erro = np.sum((vref - vs) ** 2)
    return erro

limites = [
    (-1.2, -0.8),         # x1
    (1e-3, 5e-3),         # x2
    (3.6e-5, 9.8e-5),     # x3
    (-2.6e-4, -0.954e-4), # x4
    (1e-4, 8e-4),         # Rint
    (1.972e-4, 0.2),      # m
    (0.03, 0.5002),       # n
]

print("Iniciando otimização com Differential Evolution...\n")
resultado = differential_evolution(
    func_objetivo,
    bounds=limites,
    maxiter=3000,
    popsize=25,
    tol=1e-8,
    polish=True
)
params_otimizados = resultado.x

# ============================
# refinig
# ============================
print("\nRefinando solução com L-BFGS-B...\n")
refinamento = minimize(
    func_objetivo,
    params_otimizados,
    bounds=limites,
    method='L-BFGS-B',
    options={'maxiter': 5000, 'ftol': 1e-12}
)
params_otimizados = refinamento.x

v_saida_otimizada = calcular_vsaida(corrente, temperatura, params_otimizados)
potencia_saida = v_saida_otimizada * corrente
erro_perc = np.abs((vref - v_saida_otimizada) / vref) * 100

nomes_parametros = ["x1", "x2", "x3", "x4", "Resistência Interna", "m", "n"]
print("\nParâmetros otimizados:")
for nome, valor in zip(nomes_parametros, params_otimizados):
    print(f"{nome}: {valor:.6f}")

print(f"\nErro percentual médio: {np.mean(erro_perc):.2f}%")
print("\nErros ponto a ponto:")
for i in range(len(corrente)):
    print(f"I = {corrente[i]:.2f} A | T = {temperatura[i]:.2f} K | Vref = {vref[i]:.2f} V | Vsim = {v_saida_otimizada[i]:.2f} V | Erro = {erro_perc[i]:.2f}%")

plt.figure()
plt.plot(corrente, vref, 'o-', label='Vref (real)')
plt.plot(corrente, v_saida_otimizada, 's--', label='Vsim (modelo ajustado)')
plt.xlabel('Corrente (A)')
plt.ylabel('Tensão (V)')
plt.title('Tensão em função da corrente')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(corrente, potencia_saida, 'o-', color='green', label='Potência de saída')
plt.xlabel('Corrente (A)')
plt.ylabel('Potência (W)')
plt.title('Potência em função da corrente')
plt.legend()
plt.grid(True)
plt.show()