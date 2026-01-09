import cv2
import numpy as np
import os


class LinhaDetectorPartitura:
    def __init__(self, caminho_imagem, salvar_resultado=False, pasta_saida='teste/deteccao'):
        self.caminho_imagem = caminho_imagem
        self.salvar_resultado = salvar_resultado
        self.pasta_saida = pasta_saida
        self.img_original = None
        self.img_processada = None
        self.linhas_horizontais = []
        self.linhas_verticais = []

        if self.salvar_resultado:
            os.makedirs(self.pasta_saida, exist_ok=True)

    def carregar_e_processar(self):
        self.img_original = cv2.imread(self.caminho_imagem)
        if self.img_original is None:
            raise FileNotFoundError(
                f"Imagem não encontrada: {self.caminho_imagem}")

        gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.img_processada = binary

    def detectar_linhas(self):
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        linhas_h_img = cv2.morphologyEx(
            self.img_processada, cv2.MORPH_OPEN, kernel_h, iterations=1)
        contornos_h, _ = cv2.findContours(
            linhas_h_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        linhas_v_img = cv2.morphologyEx(
            self.img_processada, cv2.MORPH_OPEN, kernel_v, iterations=1)
        contornos_v, _ = cv2.findContours(
            linhas_v_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos_h:
            x, y, w, h = cv2.boundingRect(cnt)
            self.linhas_horizontais.append((x, y, w, h))
            if self.salvar_resultado:
                cv2.rectangle(self.img_original, (x, y),
                              (x+w, y+h), (0, 0, 255), 1)

        for cnt in contornos_v:
            x, y, w, h = cv2.boundingRect(cnt)
            self.linhas_verticais.append((x, y, w, h))
            if self.salvar_resultado:
                cv2.rectangle(self.img_original, (x, y),
                              (x+w, y+h), (0, 255, 0), 1)

    def salvar_resultados(self):
        if not self.salvar_resultado:
            return

        caminho_txt = os.path.join(self.pasta_saida, 'linhas_detectadas.txt')
        with open(caminho_txt, 'w') as f:
            f.write("tipo,x,y,w,h\n")
            for x, y, w, h in self.linhas_horizontais:
                f.write(f"horizontal,{x},{y},{w},{h}\n")
            for x, y, w, h in self.linhas_verticais:
                f.write(f"vertical,{x},{y},{w},{h}\n")

        nome_arquivo = os.path.basename(self.caminho_imagem)
        caminho_img_saida = os.path.join(self.pasta_saida, nome_arquivo)
        cv2.imwrite(caminho_img_saida, self.img_original)

    def processar(self):
        self.carregar_e_processar()
        self.detectar_linhas()
        self.salvar_resultados()
        ydistintos = self.y_distintos(self)

        return self.linhas_horizontais, self.linhas_verticais, ydistintos

    @staticmethod
    def y_distintos(self):
        ys_distintos = set()
        ys_res = set()

        y_extraido = self.linhas_horizontais
        print(y_extraido)
        for (x, y, w, h) in y_extraido:
            if h <= 2:
                ys_distintos.add(y)

        ys_ordenados = sorted(ys_distintos)
        i = len(ys_ordenados) - 1

        while i > 0:
            if (ys_ordenados[i] - ys_ordenados[i-1] > 2):
                ys_res.add(ys_ordenados[i])
            i -= 1
        ys_res.add(ys_ordenados[i])

        return sorted(ys_res)

    @staticmethod
    def pre_simbolo(pauta, simbolos):
        # Exemplo incompleto, preencha conforme seu uso
        ys_distintos = set()
        ys_res = set()

        with open(simbolos, 'r') as arquivo:
            for linha in arquivo:
                partes = linha.strip().split(',')
                # Parece que o arquivo simbolos tem 6 colunas
                if len(partes) != 6:
                    continue  # pula linhas mal formatadas

                arq, class_confidence, center_x, center_y, width, height = partes
                try:
                    center_x = int(center_x)
                    width = int(width)
                    center_y = int(center_y)
                    height = int(height)

                    x_min = center_x - width / 2
                    y_min = center_y - height / 2

                except ValueError:
                    continue  # pula valores inválidos

        # completar lógica conforme necessidade


# Exemplo de uso:

if __name__ == "__main__":
    arquivo = 'teste/deteccao/linhas_detectadas.txt'
    simbolos = 'teste/deteccao/todas_deteccoes.txt'

    # Usando a função estática para extrair valores Y distintos
    valores_y = LinhaDetectorPartitura.y_distintos(arquivo)
    print("Valores y distintos com h <= 2:", valores_y)
