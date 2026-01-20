nombre = "Gustavo"
edad = 43
edad_futura = (edad) + 10

# forma : Concatenacion simple ( solo para strings)
saludo_concat = "hola, mi nombre es:" + nombre
print(saludo_concat)

# forma 2: usando F-strings ( la forma mas moderna y recomendable )
saludo_fstring = f"hola, mi nombre es {nombre} y tengo {edad} años."
print(saludo_fstring)

saludo_fstring = f"mi edad en 10 años sera: {edad_futura} "
print(saludo_fstring)

