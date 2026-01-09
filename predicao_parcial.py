from ultralytics import YOLO
import torch
import os
import cv2
import random

# Verifica se a GPU 1 está disponível
assert torch.cuda.device_count() > 1, "GPU 1 não disponível!"

# Usa explicitamente a GPU 1
device = "cuda:1"

# Carrega o modelo na GPU 1
model = YOLO("runs11/detect/yolov8x_15367/weights/best.pt")
model.to(device)

# Lista de imagens
image_paths = [
    "partitura_fundo_branco5.png",
]

# Classes que você quer detectar
classes_desejadas = [101, 73, 82, 78, 116, 24]

nomes_curto = {
    82: "nBk",     # noteheadBlack
    101: "rQ",    # restQuarter
    73: "gCf",
    78: "kSp",
    116: "TS4",      # timeSig4
    24: "beam"
}

# Gera cores diferentes para cada classe detectada
cores_por_classe = {}
for cls_id in classes_desejadas:
    # Cor RGB aleatória
    cores_por_classe[cls_id] = tuple(random.randint(0, 255) for _ in range(3))

# Faz a inferência
results = model.predict(image_paths, device=1, classes=classes_desejadas)

# Cria pasta para salvar imagens com rótulos desenhados
os.makedirs("imagens_com_rotulos", exist_ok=True)

# Salva todas as detecções em um arquivo
with open("todas_deteccoes_parcial.txt", "w") as f:
    for r in results:
        filename = os.path.basename(r.path)
        img = cv2.imread(r.path)

        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = nomes_curto.get(cls_id, model.names[cls_id])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Salva no txt
            f.write(
                f"{filename} {cls_id} {cls_name} {conf:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")

            # Escolhe cor da classe
            cor = cores_por_classe.get(cls_id, (0, 255, 0))

            # Desenha retângulo
            cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)

            # Escreve rótulo com fonte menor
            texto = f"{cls_name} {conf:.2f}"
            fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL
            tamanho_fonte = 0.4
            espessura = 1
            cv2.putText(img, texto, (x1, y1 - 6), fonte,
                        tamanho_fonte, cor, espessura)

        # Salva a imagem com rótulos
        output_path = os.path.join("imagens_com_rotulos", filename)
        cv2.imwrite(output_path, img)
