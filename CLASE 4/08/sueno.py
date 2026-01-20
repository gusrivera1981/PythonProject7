import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('sueno.csv')

# Calcular duración del sueño
df['hora_inicio'] = pd.to_datetime(df['hora_inicio'])
df['hora_fin'] = pd.to_datetime(df['hora_fin'])
df['horas_dormidas'] = (df['hora_fin'] - df['hora_inicio']).dt.total_seconds() / 3600

# Estadísticas básicas
media = df['horas_dormidas'].mean()
mediana = df['horas_dormidas'].median()
desv = df['horas_dormidas'].std()

print(f"Media: {media:.2f} horas")
print(f"Mediana: {mediana:.2f} horas")
print(f"Desviación estándar: {desv:.2f}")

# Histograma
plt.hist(df['horas_dormidas'], bins=8, color='skyblue', edgecolor='black')
plt.title('Distribución de horas de sueño')
plt.xlabel('Horas dormidas')
plt.ylabel('Frecuencia')
plt.show()
