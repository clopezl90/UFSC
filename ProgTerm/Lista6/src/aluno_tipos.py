from .aluno import Aluno

class AlunoGrad(Aluno):
    def resumo(self):
        return f"[Graduação] {super().resumo()}"

class AlunoMSC(Aluno):
    def resumo(self):
        return f"[Mestrado] {super().resumo()}"

class AlunoDSC(Aluno):
    def resumo(self):
        return f"[Doutorado] {super().resumo()}"
