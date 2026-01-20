from operator import truediv

# Un diccionario para almacenar información de productos de una tienda de mascotas
print("TIENDA DE MASCOTAS")
productos = {
    "articulo": "snacks",
    "precio": 0.25,
    "disponible": True,

}

print("\nRecorriendo claves y valores:")
for clave, valor in productos.items():
    print(f"{clave}: {valor}")

# Acceder a un valor usando su clave
print(f"\nEl nombre del articulo: {productos['articulo']}")

# Modificar un valor
productos["precio"] = 0.25,
print(f"el precio del producto es : {productos['precio']}")

if productos["disponible"]:
    print("¡Producto disponible para la venta! ")
else:
    print("!producto no disponible!.")




# Un diccionario de ejemplo
carro = {
    "marca": "Ford",
    "modelo": "Mustang",
    "año": 1964
}

# Opción 1: Recorrer las claves
print("\nRecorriendo las claves:")
for clave in carro:
    print(clave)

# Opción 2: Recorrer los valores
print("\nRecorriendo los valores:")
for valor in carro.values():
    print(valor)

# Opción 3: Recorrer ambos (clave y valor)
print("\nRecorriendo claves y valores:")
for clave, valor in carro.items():
    print(f"{clave}: {valor}")