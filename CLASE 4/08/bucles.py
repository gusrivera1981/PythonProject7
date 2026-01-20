nombre = ""
while nombre != "fin":
    nombre = input("Ingresa tu nombre (o escribe 'fin' para salir): ")
    if nombre != "fin":
        print(f"Hola, {nombre}")


contador = 1
while contador <= 5:
    texto = input(f"({contador}/5) Escribe algo: ")
    print(f"Escribiste: {texto}")
    contador += 1