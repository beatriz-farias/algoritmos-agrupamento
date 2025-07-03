import random
import math

# ---- Funções auxiliares ---- #

# Calcula a distância Euclidiana entre dois vetores
def distancia_euclidiana(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

# Soma dois vetores componente a componente
def soma_vetores(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

# Subtrai dois vetores componente a componente
def subtrai_vetores(v1, v2):
    return [a - b for a, b in zip(v1, v2)]

# Multiplica vetor por escalar
def multiplica_escalar(escalar, vetor):
    return [escalar * v for v in vetor]

# Função Gaussiana para calcular influência de vizinhos
def gaussiana(distancia, sigma):
    return math.exp(- (distancia ** 2) / (2 * sigma ** 2))


# ---- Classe Mapa Auto-Organizável (SOM) ---- #

class SOM:
    def __init__(self, linhas_grade, colunas_grade, tamanho_entrada, taxa_aprendizado=0.1, sigma=1.0, epocas=1000):
        self.linhas = linhas_grade                     # número de linhas da grade
        self.colunas = colunas_grade                   # número de colunas da grade
        self.tamanho_entrada = tamanho_entrada         # dimensão do vetor de entrada
        self.taxa_inicial = taxa_aprendizado           # taxa de aprendizado inicial
        self.sigma_inicial = sigma                     # raio inicial da vizinhança
        self.epocas = epocas                           # número de épocas de treinamento
        self.grade = self._inicializar_grade()         # inicializa pesos dos neurônios

    # Inicializa a grade com pesos aleatórios
    def _inicializar_grade(self):
        grade = []
        for i in range(self.linhas):
            linha = []
            for j in range(self.colunas):
                pesos = [random.uniform(0, 1) for _ in range(self.tamanho_entrada)]
                linha.append(pesos)
            grade.append(linha)
        return grade

    # Encontra o neurônio cuja saída é mais próxima da entrada (BMU)
    def _achar_bmu(self, entrada):
        menor_dist = float('inf')
        coordenadas_bmu = (0, 0)
        for i in range(self.linhas):
            for j in range(self.colunas):
                pesos = self.grade[i][j]
                dist = distancia_euclidiana(pesos, entrada)
                if dist < menor_dist:
                    menor_dist = dist
                    coordenadas_bmu = (i, j)
        return coordenadas_bmu

    # Calcula a distância entre dois neurônios na grade
    def _distancia_vizinhanca(self, c1, c2):
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    # Treina o mapa com os dados fornecidos
    def treinar(self, dados):
        for epoca in range(self.epocas):
            # Atualiza a taxa de aprendizado e sigma (decaimento exponencial)
            taxa = self.taxa_inicial * math.exp(-epoca / self.epocas)
            sigma = self.sigma_inicial * math.exp(-epoca / self.epocas)

            for entrada in dados:
                bmu_i, bmu_j = self._achar_bmu(entrada)

                for i in range(self.linhas):
                    for j in range(self.colunas):
                        coord_neuronio = (i, j)
                        dist_viz = self._distancia_vizinhanca((bmu_i, bmu_j), coord_neuronio)

                        if dist_viz <= sigma:
                            influencia = gaussiana(dist_viz, sigma)
                            pesos = self.grade[i][j]
                            diferenca = subtrai_vetores(entrada, pesos)
                            ajuste = multiplica_escalar(taxa * influencia, diferenca)
                            self.grade[i][j] = soma_vetores(pesos, ajuste)

    # Retorna a grade com os pesos atualizados
    def obter_pesos(self):
        return self.grade
