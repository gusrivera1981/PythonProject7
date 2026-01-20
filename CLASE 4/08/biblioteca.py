# Clase Libro
class Libro:
    def __init__(self, titulo, autor):
        self.titulo = titulo
        self.autor = autor


class Biblioteca:  # No hereda de Libro
    def __init__(self):
        # Aquí inicializa una lista vacía para los libros
        self.libros = []

    def agregar_libro(self, libro):
        # Aquí añades el libro a la lista
        self.libros.append(libro)

    def mostrar_libros(self,):
        if not self.libros:
            print("La biblioteca está vacía.")
        else:
            print(" Lista de libros en la biblioteca:")
            for libro in self.libros:
                print(f"- {libro.titulo} (Autor: {libro.autor})")



libro1 = Libro("Cien años de soledad" , "Gabriel García Márquez")
libro2 = Libro("Don Quijote de la Mancha", "Miguel de Cervantes")
libro3 = Libro("1984", "George Orwell")
libro4 = Libro("El señor de los anillos", "J.R.R tolkien")


# Crear biblioteca
mi_biblioteca = Biblioteca()

# Agregar libros a la biblioteca
mi_biblioteca.agregar_libro(libro1)
mi_biblioteca.agregar_libro(libro2)
mi_biblioteca.agregar_libro(libro3)
mi_biblioteca.agregar_libro(libro4)

# Mostrar libros
mi_biblioteca.mostrar_libros()



#programa funcionamiento append
frutas = ["manzana", "banana"]
frutas.append("cereza")
for frutas in frutas:
    print(f"mi fruta favorita es : {frutas}")
print (f"\nfrutas",(frutas))