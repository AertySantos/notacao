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

        # Usa explicitamente a GPU 1
        device = "cuda:1"

        # Carrega o modelo na GPU 1
        model = YOLO("runs11/detect/yolov8x_15367/weights/best.pt")
        model.to(device)

        # Pega todas as imagens com as extensões desejadas
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_paths.extend(glob.glob(os.path.join(self.pasta, ext)))

        # Faz a inferência
        results = model.predict(image_paths, device=1)

        for img_path in image_paths:
            results = model.predict(img_path, device=device)
            nome_base = os.path.splitext(os.path.basename(img_path))[0]
            arquivo_saida = f"{nome_base}_deteccoes84.txt"

            with open(arquivo_saida, "w") as f:
                for r in results:
                    filename = os.path.basename(r.path)
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        descricao = f"{filename} {cls_id} {cls_name} {conf:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n"
                        simbolos.append([
                            filename,
                            cls_id,
                            cls_name,
                            f"{conf:.4f}",
                            f"{x1:.1f}",
                            f"{y1:.1f}",
                            f"{x2:.1f}",
                            f"{y2:.1f}"
                        ])
                        f.write(descricao)

        return simbolos


if __name__ == "__main__":
    pred = Predicao()
    predicao = pred.simbolos_detectados("./paginas_pdf")
