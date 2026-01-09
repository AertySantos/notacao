from ultralytics import YOLO
import torch
import os
import glob


class Predicao:
    def simbolos_detectados(self, pasta):
        self.pasta = pasta
        simbolos = []

        # Verifica se a GPU 1 está disponível
        assert torch.cuda.device_count() > 1, "GPU 1 não disponível!"

        device = "cuda:1"

        # Carrega o modelo na GPU 1
        model = YOLO("runs11/detect/yolov8x_15367/weights/best.pt")
        model.to(device)

        # Coleta imagens
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_paths.extend(glob.glob(os.path.join(self.pasta, ext)))

        # LOOP pelas imagens — uma de cada vez
        for img_path in image_paths:

            results = model.predict(img_path, device=device)

            # Nome base sem extensão
            nome_base = os.path.splitext(os.path.basename(img_path))[0]

            # Arquivo TXT de saída
            arquivo_saida = f"{nome_base}_deteccoes.txt"

            # Arquivo de imagem com rótulos
            arquivo_imagem = f"{nome_base}_rotulado.jpg"

            with open(arquivo_saida, "w") as f:
                for r in results:
                    # Salva imagem anotada
                    r.save(filename=arquivo_imagem)  # <<<<<< AQUI SALVA A IMAGEM PREVISTA

                    filename = os.path.basename(r.path)

                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        descricao = (
                            f"{filename} {cls_id} {cls_name} "
                            f"{conf:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n"
                        )

                        simbolos.append([
                            filename, cls_id, cls_name,
                            f"{conf:.4f}", f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"
                        ])

                        f.write(descricao)

        return simbolos


if __name__ == "__main__":
    pred = Predicao()
    predicao = pred.simbolos_detectados("./paginas_pdf")
