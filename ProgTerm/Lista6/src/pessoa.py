class Pessoa:
    def __init__(self, nome, cpf):
        self.nome = nome
        self.cpf = cpf
        self.endereco  = None

    def apresentar(self):
        print(f'Meu nome é {self.nome}')

    def set_endereco(self, endereco):
        self.endereco = endereco