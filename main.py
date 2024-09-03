import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
datos = pd.read_csv('altura_peso.csv', sep=',')

# Extracción de las columnas 'Altura' y 'Peso'
x = datos['Altura'].values
y = datos['Peso'].values

# Reemplazo de valores NaN, infinito positivo y negativo con la media
x = np.nan_to_num(x, nan=np.nanmean(x), posinf=np.nanmean(x), neginf=np.nanmean(x))
y = np.nan_to_num(y, nan=np.nanmean(y), posinf=np.nanmean(y), neginf=np.nanmean(y))

# Normalización de los datos
x_mean, x_std = np.mean(x), np.std(x)
y_mean, y_std = np.mean(y), np.std(y)

x = (x - x_mean) / x_std
y = (y - y_mean) / y_std