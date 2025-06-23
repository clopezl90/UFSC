from .pessoa import Pessoa

class Aluno(Pessoa):
    def __init__(self, nome, cpf, matricula):
        super().__init__(nome, cpf)
        self.matricula = matricula
        self.nota = None

    def set_nota(self, nota):
        self.nota = nota

    def resumo(self):
        return f"Aluno {self.nome} (matrícula {self.matricula})"

    def __str__(self):
        return f'Aluno {self.nome}, matricula {self.matricula}'
