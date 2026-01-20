# Pedimos un número al usuario
edad = int(input("Ingresa tu edad: "))

# Imprimimos el resultado (si todo salió bien)
print(f"Tu edad es: {edad}")

while True:
        try:
            edad = int(input("Ingresa tu edad: "))
            # Si el código llega aquí, significa que no hubo error
            print(f"Tu edad es: {edad}")
            break  # La palabra clave 'break' detiene el bucle
        except ValueError:
            print("¡Error! Por favor, ingresa un número válido.")










