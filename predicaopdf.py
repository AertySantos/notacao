from ultralytics import YOLO
import torch
import os
import glob
from pdf2image import convert_from_path


class Predicao:

    def pdf_para_imagens(self, arquivo_pdf, pasta_saida):
        """Converte um PDF em imagens PNG dentro da pasta especificada."""
        os.makedirs(pasta_saida, exist_ok=True)
        paginas = convert_from_path(arquivo_pdf, dpi=300)

        imagens_salvas = []
        for i, pagina in enumerate(paginas):
            caminho_img = os.path.join(pasta_saida, f"pagina_{i+1}.png")
            pagina.save(caminho_img, "PNG")
            imagens_salvas.append(caminho_img)

        return imagens_salvas

    def simbolos_detectados(self, pasta_ou_pdf):
        simbolos = []

        # Verifica se é PDF
        if pasta_ou_pdf.lower().endswith(".pdf"):
            print(" Convertendo PDF em imagens...")
            pasta_temp = "paginas_pdf"
            imagens = self.pdf_para_imagens(pasta_ou_pdf, pasta_temp)
            pasta = pasta_temp
        else:
            pasta = pasta_ou_pdf

        # Verifica GPU 1
        assert torch.cuda.device_count() > 1, "GPU 1 não disponível!"
        device = "cuda:1"

        model = YOLO("runs11/detect/yolov8x_15367/weights/best.pt")
        model.to(device)

        # Procura imagens
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_paths.extend(glob.glob(os.path.join(pasta, ext)))

        if len(image_paths) == 0:
            raise FileNotFoundError("Nenhuma imagem encontrada!")

        # Faz predições
        for img_path in image_paths:
            results = model.predict(img_path, device=device)
            nome_base = os.path.splitext(os.path.basename(img_path))[0]
            arquivo_saida = f"{nome_base}_deteccoespdf.txt"

            with open(arquivo_saida, "w") as f:
                for r in results:
                    filename = os.path.basename(r.path)
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        linha = (
                            f"{filename} {cls_id} {cls_name} {conf:.4f} "
                            f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n"
                        )

                        simbolos.append([
                            filename, cls_id, cls_name,
                            f"{conf:.4f}",
                            f"{x1:.1f}", f"{y1:.1f}",
                            f"{x2:.1f}", f"{y2:.1f}"
                        ])
                        f.write(linha)

        return simbolos


if __name__ == "__main__":
    pred = Predicao()
    
    # 👉 Aqui você coloca seu PDF
    resultado = pred.simbolos_detectados("partitura/bcdadabf-30ae-4f2a-8eec-5131c1ae21b3.pdf")

    print("Detecções realizadas:", len(resultado))
