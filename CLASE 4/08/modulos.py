# Importamos el módulo 'math' que contiene funciones matemáticas.
import math

# Usamos la función 'sqrt' del módulo 'math' para calcular la raíz cuadrada.
raiz_cuadrada = math.sqrt(25)
print(f"La raíz cuadrada de 25 es: {raiz_cuadrada}")

# También podemos importar una función específica.
from random import randint

# Usamos la función 'randint' para generar un número entero aleatorio.
año_aleatorio = randint(1900, 2025)
print(f"Tu año aleatorio es: {año_aleatorio}")

#Modifica el programa para que use math.pow() y calcule la potencia de dos números de tu elección.
potencia = math.pow(2 , 3)
print(f"La potencia es : {potencia}")