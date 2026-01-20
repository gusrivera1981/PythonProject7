#funciones

def suma(a,b):
    return a + b

def resta(a,b):
    return  a - b

def multiplicar(a,b):
    return a * b

def dividir(a,b):
    if b == 0:
        return "error"
    return a / b

while True:
        try:
            num1 = float(input("Ingresa tu numero: "))
            num2 = float(input("Ingresa tu numero: "))


        except ValueError:
            print("¡Error!.")

        operacion= input("ingresa la operacion: (+,-,*,/)")

        if operacion == "+":
            resultado = suma(num1 , num2)

        elif operacion == "-":
            resultado = resta(num1 , num2)

        elif operacion == "*":
            resultado = multiplicar(num1 , num2)

        elif operacion == "/":
            resultado = dividir(num1 , num2)

        else:
            print("operacion no valida")
            continue


        print("resultado:", resultado)

        continuar = input("¿Quieres hacer otra operación? Escribe 'no' para salir: ")
        if continuar == "no":
            break



