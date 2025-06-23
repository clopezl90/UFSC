
class Turma:
    def __init__(self, codigo, professor):
        self.codigo = codigo
        self.professor = professor
        self.alunos = []

    def adicionar_aluno(self, aluno):
        self.alunos.append(aluno)

    def listar(self):
        print(f"Turma {self.codigo} - Professor: {self.professor.nome}")
        for aluno in self.alunos:
            print(aluno.resumo())

    def media(self):
        notas_validas = [a.nota for a in self.alunos if a.nota is not None]
        if notas_validas:
            return sum(notas_validas) / len(notas_validas)
        return None