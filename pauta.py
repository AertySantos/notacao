import cv2
import numpy as np
import os

# Cria diretório de saída se não existir
os.makedirs('teste', exist_ok=True)
os.makedirs('teste/deteccao', exist_ok=True)

# Carrega a imagem
img = cv2.imread('partitura_fundo_branco5.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binariza e inverte
_, binary = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# --------- DETECÇÃO DE LINHAS HORIZONTAIS (pautas) ----------
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
horizontal_lines = cv2.morphologyEx(
    binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

# Encontra contornos
contours_h, _ = cv2.findContours(
    horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --------- DETECÇÃO DE LINHAS VERTICAIS (hastes e barras) ----------
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
vertical_lines = cv2.morphologyEx(
    binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

contours_v, _ = cv2.findContours(
    vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Salvar coordenadas
with open('teste/deteccao/linhas_detectadas.txt', 'w') as f:
    f.write("tipo,x,y,w,h\n")

    for cnt in contours_h:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)  # vermelho
        f.write(f"horizontal,{x},{y},{w},{h}\n")

    for cnt in contours_v:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)  # verde
        f.write(f"vertical,{x},{y},{w},{h}\n")

# Salvar imagem com linhas destacadas
cv2.imwrite('teste/deteccao/partitura_tratada_300dpi100.jpg', img)

print("Linhas detectadas e salvas com sucesso!")
