# IA basica para negocios que predice 
# ingresos mensuales en función de la 
# inversión en publicidad
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generar datos simulados
np.random.seed(0)
inversion = np.random.rand(100, 1) * 10000  # Inversión en publicidad (en USD)
ingresos = 3 * inversion + np.random.randn(100, 1) * 10000  # Ingresos mensuales (en USD)

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(inversion, ingresos)

# Predecir ingresos con el modelo entrenado
ingresos_predichos = modelo.predict(inversion)

# Visualizar los resultados
plt.scatter(inversion, ingresos, color='blue', label='Datos simulados')
plt.plot(inversion, ingresos_predichos, color='red', label='Línea de regresión')
plt.xlabel('Inversión en publicidad (USD)')
plt.ylabel('Ingresos mensuales (USD)')
plt.title('Regresión lineal: Inversión vs Ingresos')
plt.legend()
plt.show()

# Función para predecir ingresos según una inversión dada
def predecir_ingresos(inversion_nueva):
    inversion_nueva = np.array([[inversion_nueva]])
    ingresos_nuevos = modelo.predict(inversion_nueva)
    return ingresos_nuevos[0][0]

# Ejemplo de predicción
inversion_ejemplo = 15000  # Inversión en USD
ingresos_predichos_ejemplo = predecir_ingresos(inversion_ejemplo)
print(f'Para una inversión de ${inversion_ejemplo} USD, se predicen ingresos de aproximadamente ${ingresos_predichos_ejemplo:.2f} USD.')

