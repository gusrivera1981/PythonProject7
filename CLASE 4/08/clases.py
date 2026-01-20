# La palabra clave 'class' nos dice que estamos creando un nuevo plano.
class Carro:
    # Este es el constructor. La función __init__ se ejecuta cuando creamos un nuevo objeto.
    # 'self' se refiere al objeto que se está creando.
    def __init__(self, marca, color, año):
        # Aquí asignamos los valores que le damos al objeto.
        self.marca = marca
        self.color = color
        self.año = año

    def arrancar(self):
        print(f"El {self.marca} ha arrancado.")



# Creamos un objeto (una "instancia") de nuestra clase 'Carro'.
carro1 = Carro("Ford", "Rojo", "1950",)
carro2 = Carro("Chevrolet", "Azul", "1975")
carro3 = Carro("renault", "blanco", "1965")

# Accedemos a las "características" de nuestros objetos.
print(f"El carro 1 es un {carro1.marca} de color {carro1.color} del año {carro1.año}.")
print(f"El carro 2 es un {carro2.marca} de color {carro2.color} del año {carro2.año}.")
print(f"El carro 3 es un {carro3.marca} de color {carro3.color} del año {carro3.año}.")

carro1.arrancar()
carro2.arrancar()
carro3.arrancar()

class CarroElectrico(Carro):  # CarroElectrico hereda de Carro
    def __init__(self, marca, color, año, bateria):
        # La función super() llama al constructor de la clase padre
        super().__init__(marca, color, año)
        self.bateria = bateria

    def cargar(self):
        print(f"El {self.marca} se está cargando. Nivel de batería: {self.bateria}%")

    def arrancar(self):
        print(f"El {self.marca} arranca silenciosamente con su motor eléctrico.")

carro_electrico = CarroElectrico("Tesla", "Negro", 2024, 90)
carro_electrico.arrancar()  # ¿Crees que este método funcionará?
carro_electrico.cargar()