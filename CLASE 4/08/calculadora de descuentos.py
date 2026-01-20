producto = input("ingrese el producto: ")
precio_producto = float(input("ingrese el valor del producto: "))
descuento_porcentaje = float(input("ingrese el % de descuento: " ))

descuento_aplicado = precio_producto * (descuento_porcentaje / 100 )
precio_total = precio_producto - descuento_aplicado

print(f"El precio original es: {precio_producto}")
print(f"El descuento aplicado es: {descuento_aplicado}")
print(f"El precio final con el descuento es: {precio_total}")

if descuento_porcentaje >=50:
    print("¡Wow! ¡Es un gran descuento!")






