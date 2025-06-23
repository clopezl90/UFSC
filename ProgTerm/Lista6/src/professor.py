from .pessoa import Pessoa

class Professor(Pessoa):
    def __init__(self, nome, cpf, siape):
        super().__init__(nome, cpf)
        self.siape = siape
    def apresentar(self):
        print(f"Olá, sou o professor {self.nome} (SIAPE {self.siape}).")
