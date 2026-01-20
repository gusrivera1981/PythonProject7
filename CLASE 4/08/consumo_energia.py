import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('consumo_energia.csv')

# Consumo total diario
consumo_diario = df.groupby('día')['consumo_kwh'].sum()

# Horario de mayor consumo
horario_max = df.groupby('hora')['consumo_kwh'].mean().idxmax()

print("Consumo total por día:\n", consumo_diario)
print(f"\nHora con mayor consumo promedio: {horario_max}:00 hrs")

# Gráfica de tendencia
plt.plot(consumo_diario.index, consumo_diario.values, marker='o')
plt.title('Consumo eléctrico diario (kWh)')
plt.xlabel('Día')
plt.ylabel('Consumo (kWh)')
plt.grid(True)
plt.show()
