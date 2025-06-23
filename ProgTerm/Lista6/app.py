from src.professor import Professor
from src.aluno_tipos import AlunoDSC, AlunoMSC, AlunoGrad
from src.turma import Turma

professor = Professor("Rafael", "444.444.444-44", 2023786)
turma_prog_term = Turma('EMC123213', professor)
alunos=[]
with open("turma_EMC410235.csv", "r") as f:
    linhas = f.readlines()[1:]
    for linea in linhas:
        columnas = linea.strip().split(',')
        nome = columnas[0]
        matricula = columnas[1]
        cpf = columnas[2]
        tipo = columnas[3]

        if tipo == "Grad":
            aluno = AlunoGrad(nome, cpf, matricula)
            alunos.append(aluno)
        elif tipo == "MSc":
            aluno = AlunoMSC(nome, cpf, matricula)
            alunos.append(aluno)
        elif tipo == "DSc":
            aluno = AlunoDSC(nome, cpf, matricula)
            alunos.append(aluno)


for aluno in alunos:
    turma_prog_term.adicionar_aluno(aluno)

turma_prog_term.listar()

