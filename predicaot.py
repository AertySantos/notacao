from ultralytics import YOLO
import torch
import cv2
import os
import random
import glob


class Predicao:
    def simbolos_detectados(self, pasta):
        self.pasta = pasta

        # Lista final para retornar (mesmo formato do código anterior)
        simbolos = []

        # -------------------------------------------------------------
        # 1) Checa GPU e carrega modelo
        # -------------------------------------------------------------
        assert torch.cuda.device_count() > 1, "GPU 1 não disponível!"
        device = "cuda:1"

        model = YOLO("runs11/detect/yolov8x_15367/weights/best.pt")
        model.to(device)

        # -------------------------------------------------------------
        # 2) Dicionário ID → código 3 letras (137 classes)
        # -------------------------------------------------------------
        nomes_3 = {
            0:'oBr',1:'oTm',2:'oTb',3:'oTa',4:'oTv',5:'aDF',6:'aDS',7:'aFl',8:'aNa',9:'aSh',
            10:'arp',11:'aAA',12:'aAB',13:'aMA',14:'aMB',15:'aSA',16:'aSB',17:'aSA2',18:'aSB2',
            19:'aTA',20:'aTB',21:'aDo',22:'bHe',23:'bSi',24:'bea',25:'bra',26:'cCA',27:'cCC',
            28:'cCT',29:'cTC',30:'cae',31:'cle',32:'cle8',33:'cod',34:'cSt',35:'cTS',36:'dFF',
            37:'dF3',38:'dF4',39:'dF5',40:'dFo',41:'dFP',42:'dMF',43:'dMP',44:'dMe',45:'dPP',
            46:'dP3',47:'dP4',48:'dP5',49:'dPi',50:'dR2',51:'dS1',52:'dSz',53:'fCl',54:'fCC',
            55:'fAA',56:'fAB',            57:'f00',58:'f01',59:'f02',60:'f03',61:'f04',62:'f05',
            63:'f1D',64:'f1U',65:'f16',66:'f1U2',67:'f3D',68:'f3U',69:'f6D',70:'f6U',71:'f8D',
            72:'f8U',73:'gCl',74:'gCC',75:'hai',76:'kFl',77:'kNa',78:'kSh',79:'kPP',80:'kPU',
            81:'lLi',82:'nBl',83:'nDW',84:'nHa',85:'nWh',86:'oMo',87:'oTr',88:'oTu',89:'oTI',
            90:'rDo',91:'r12',92:'r16',93:'r32',94:'r64',95:'r08',96:'rDW',97:'rHB',98:'rHN',
            99:'rHa',100:'rLo',101:'rQu',102:'rWh',103:'seg',104:'slu',105:'sLi',106:'ste',
            107:'sDB',108:'sUB',109:'tFi',110:'tSc',111:'tie',112:'tS0',113:'tS1',114:'tS2',
            115:'tS3',116:'tS4',117:'tS5',118:'tS6',119:'tS7',120:'tS8',121:'tS9',122:'tSC',
            123:'tCC',124:'t1r',125:'t2r',126:'t3r',127:'t4r',128:'uT1',129:'uT3',130:'uT4',
            131:'uT5',132:'uT6',133:'uT7',134:'uT8',135:'uT9',136:'uBr'
        }

        # -------------------------------------------------------------
        # 3) Lista de imagens para processar
        # -------------------------------------------------------------
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_paths.extend(glob.glob(os.path.join(self.pasta, ext)))

        # -------------------------------------------------------------
        # 4) Rodar inferência
        # -------------------------------------------------------------
        results = model.predict(image_paths, device=device)

        # Pasta de saída para imagens rotuladas
        os.makedirs("imagens_com_rotulos", exist_ok=True)

        # Arquivo TXT geral
        txt_saida = "deteccoes_completastt.txt"
        f = open(txt_saida, "w")

        # -------------------------------------------------------------
        # 5) Processa cada imagem
        # -------------------------------------------------------------
        for r in results:
            filename = os.path.basename(r.path)
            img = cv2.imread(r.path)

            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls3 = nomes_3[cls_id]

                # Escreve no TXT
                f.write(
                    f"{filename} {cls_id} {cls3} {conf:.4f} {x1} {y1} {x2} {y2}\n"
                )

                # ----------------------------
                # ADICIONA NA LISTA DE RETORNO
                # ----------------------------
                simbolos.append([
                    filename,
                    cls_id,
                    cls3,
                    f"{conf:.4f}",
                    f"{x1}",
                    f"{y1}",
                    f"{x2}",
                    f"{y2}"
                ])

                # Desenha box e texto
                random.seed(cls_id)
                cor = tuple(random.randint(0, 255) for _ in range(3))

                cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
                texto = f"{cls3} {conf:.2f}"
                cv2.putText(img, texto, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)

            # Salva imagem com rótulos
            out_path = os.path.join("imagens_com_rotulos", filename)
            cv2.imwrite(out_path, img)

        f.close()

        print("Processamento concluído!")

        
        return simbolos
