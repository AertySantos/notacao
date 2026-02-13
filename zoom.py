import cv2
import numpy as np

def colocar_em_pagina(input_path, output_path, page_size=1536, bg_color=255):
    """
    Centraliza a imagem em uma página quadrada (canvas).
    Não distorce a imagem.
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(input_path)

    h, w = img.shape

    # cria a "página"
    pagina = np.full((page_size, page_size), bg_color, dtype=np.uint8)

    # calcula posição para centralizar
    y_offset = (page_size - h) // 2
    x_offset = (page_size - w) // 2

    # cola a partitura na página
    pagina[y_offset:y_offset + h, x_offset:x_offset + w] = img

    cv2.imwrite(output_path, pagina)
    return pagina


# Exemplo de uso
if __name__ == "__main__":
    colocar_em_pagina(
         "tratamento/partitura_teste_1.png",
        "tratamento/partitura_teste_1.png",
        page_size=1536
    )

