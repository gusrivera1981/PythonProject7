import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('gastos.csv')

# Calcular gasto total por categoría
gasto_por_categoria = df.groupby('categoría')['monto'].sum()

# Calcular porcentaje
porcentaje = (gasto_por_categoria / gasto_por_categoria.sum()) * 100

# Mostrar resultados
print("Gasto total por categoría:\n", gasto_por_categoria)
print("\nPorcentaje por categoría:\n", porcentaje.round(2))

# Gráfica de pastel
plt.figure(figsize=(6,6))
plt.pie(gasto_por_categoria, labels=gasto_por_categoria.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribución de gastos mensuales')
plt.show()
