#funciones

def sumar(a,b):
    return a + b

def restas(a,b):
    return a-b

def multiplicar(a,b):
    return a * b

def dividir(a,b):
    if b == 0 :
        return "error"
    return a / b

while True:
        try:
            num1 = float(input("Ingresa el primer numero: "))
            num2 = float(input("Ingresa el seg numero: "))

        except ValueError:
            print("¡Error!.")
            continue


        operacion = input("ingrese la operacion a realizar(+,-,*,/):")


        if operacion == "+":
            resultado = sumar(num1 , num2)

        elif operacion == "-":
            resultado = restas(num1, num2)

        elif operacion == "*":
            resultado = multiplicar(num1, num2)

        elif operacion == "/":
            resultado = dividir(num1, num2)

        else:
            print("operacion no valida")
            continue

        print (f"resultado: {resultado}")

        continuar = input("¿Quieres hacer otra operación? Escribe 'no' para salir: ")
        if continuar == "no":
          break


























