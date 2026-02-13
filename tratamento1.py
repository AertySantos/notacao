import cv2
import numpy as np

class Trt1:

    def __init__(self):
        pass

    def colocar_em_pagina(self, input_img, output_path, page_size=1536, bg_color=255):
        """
        Centraliza a imagem em uma página quadrada (canvas).
        Não distorce a imagem.
        """
        img = input_img

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

    def preprocess_partitura(self, input_path, output_path=None):

        # ------------------ LOAD ------------------
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(input_path)


        # ------------------ DENOISE (preserva bordas) ------------------
        clean = cv2.bilateralFilter(img, 11, 50, 50)

        # ------------------ ILUMINAÇÃO / FUNDO ------------------
        background = cv2.medianBlur(clean, 51)
        norm = cv2.divide(
            clean.astype(np.float32),
            background.astype(np.float32) + 1,
            scale=255
        )
        norm = np.clip(norm, 0, 255).astype(np.uint8)

        # ------------------ SUAVIZAÇÃO LEVE ------------------
        smooth = cv2.GaussianBlur(norm, (1, 1), 0)

        # ------------------ BINARIZAÇÃO ------------------
        _, binarizada = cv2.threshold(
            smooth,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # ------------------ LIMPEZA DE RUÍDO FINO ------------------
        binarizada = cv2.morphologyEx(
            binarizada,
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8)
        )

        # ==========================================================
        # AJUSTE FINO DE ESPESSURA (YOLO-FRIENDLY)
        # ==========================================================

        # Começa da binarizada
        simbolos_reforcados = binarizada.copy()

        # ------------------ AFINAR LEVEMENTE ------------------
        kernel_afinar = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, 2)
        )
        simbolos_reforcados = cv2.erode(
            simbolos_reforcados,
            kernel_afinar,
            iterations=1
        )

        # ------------------ SUAVIZAR SERRILHADO ------------------
        simbolos_reforcados = cv2.GaussianBlur(
            simbolos_reforcados,
            (3, 3),
            0.5
        )

        # ------------------ NORMALIZAR PARA YOLO ------------------
        final = 255 - simbolos_reforcados
        self.colocar_em_pagina(final, output_path)
        #if output_path:
            #cv2.imwrite(output_path, final)

        return final


# Exemplo
if __name__ == "__main__":
    tr = Trt1()
    tr.preprocess_partitura(
        "paginas_pdf/pagina_1.png",
        "tratamento/partitura_teste_1.png"
    )







