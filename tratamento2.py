import cv2
import numpy as np

class Trt2:

    def __init__(self):
            pass

    def preprocess_partitura(self, input_path, output_path=None):

        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(input_path)

        # 1 — Remover ruído preservando bordas
        clean = cv2.bilateralFilter(img, 11, 20, 20)

        # 2 — Normalização de iluminação
        background = cv2.medianBlur(clean, 51)
        norm = cv2.divide(clean.astype(np.float32), background.astype(np.float32)+1, scale=255)
        norm = np.clip(norm, 0, 255).astype(np.uint8)

        # 3 — CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
        enhanced = clahe.apply(norm)

        # 4 — Threshold adaptativo
        binarizada = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            12
        )

        # 5 — Remover ruído pequeno
        binarizada = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

        # 6 — Reforçar linhas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
        linhas = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel)

        final = binarizada.copy()
        #final[linhas > 0] = 0  # reforça linhas

        # 7 — Inverte para fundo branco
        final = 255 - final
        #final = melhorar_linhas(final)
        if output_path:
            cv2.imwrite(output_path, final)

        return final

    


    def melhorar_linhas(self, img):
        # img = fundo branco, linhas pretas (0)

        # 1 — Detectar apenas linhas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
        linhas = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        # 2 — Afinar
        try:
            thin = cv2.ximgproc.thinning(linhas)
        except:
            thin = cv2.erode(linhas, np.ones((3,1), np.uint8), iterations=1)

        # 3 — Suavizar serrilhas
        suaves = cv2.GaussianBlur(thin, (5,3), 0)

        # 4 — Repor na imagem original (branco=255, preto=0)
        mask = linhas > 0
        result = img.copy()
        result[mask] = suaves[mask]
        #result[mask] = 0
        return result



# Exemplo
if __name__ == "__main__":
    preprocess_partitura(
        "paginas_pdf/pagina_1.png",
        "partitura_fundo_branco33.png"
    )







