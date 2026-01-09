import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

# Caminho da imagem de partitura
imagem_caminho = "teste/partitura_tratada_300dpi.jpg"

# 1. Carregar e binarizar imagem
img = cv2.imread(imagem_caminho, cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 2. Detectar bordas
edges = cv2.Canny(binary, 50, 150, apertureSize=3)

# 3. Aplicar Transformada de Hough
linhas = cv2.HoughLines(edges, 1, np.pi/180, threshold=200)

# 4. Converter imagem para colorido para desenhar as linhas
img_colorida = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Lista para armazenar coordenadas das linhas horizontais
linhas_detectadas = []

if linhas is not None:
    # Filtrar apenas linhas horizontais (theta ~ 90°)
    horizontais = [l[0] for l in linhas if np.abs(l[0][1] - np.pi/2) < 0.1]

    # Ordenar por rho (posição vertical)
    horizontais_ordenadas = sorted(horizontais, key=lambda x: x[0])

    # Selecionar as 5 primeiras (um sistema musical)
    top_5 = horizontais_ordenadas[:5]

    for rho, theta in top_5:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Desenhar linha na imagem
        cv2.line(img_colorida, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Armazenar as posições
        linhas_detectadas.append({
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2
        })

# 5. Salvar imagem com linhas desenhadas
cv2.imwrite("linhas_detectadas.png", img_colorida)

# 6. Salvar posições em CSV
with open("coordenadas_linhas.csv", "w", newline="") as csvfile:
    fieldnames = ['x1', 'y1', 'x2', 'y2']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for linha in linhas_detectadas:
        writer.writerow(linha)

print("Linhas salvas em 'linhas_detectadas.png' e coordenadas em 'coordenadas_linhas.csv'.")
