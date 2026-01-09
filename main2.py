from music21 import stream, note, metadata, key, meter
from music21.beam import Beams, Beam
from detecta_linha import LinhaDetectorPartitura
from tratamento_linha import Tratamento
from tratamento1 import Trt1
from tratamento2 import Trt2
from predicaot import Predicao

# Caminho da imagem a ser processada
caminho_imagem = 'img/partitura_tratada_300dpi.jpg'
saida_tratamento = 'img_linha/trt1.png'
saida1 = "img_trt/trt1.png"
saida2 = "img_trt/trt2.png"

trt = Tratamento()
trt.preprocess_partitura(caminho_imagem, saida_tratamento)

trt1 = Trt1()
trt1.preprocess_partitura(caminho_imagem, saida1)

trt2 = Trt2()
trt2.preprocess_partitura(caminho_imagem, saida2)


# Cria o detector
detector = LinhaDetectorPartitura(caminho_imagem , salvar_resultado=True, pasta_saida="linha/")

# Processa e obtém as linhas detectadas
linhas_horizontais, linhas_verticais, ydistintos = detector.processar()
print("Pauta posicao y")
print(ydistintos)

# Linhas da pauta (de cima para baixo), conforme fornecido
linhas_pauta = [ydistintos]

beam_status = None
# Função para determinar a nota musical com base na posição y


def nolimite(ymin, ymax, linhas_pauta):
    l0 = linhas_pauta[0]
    l4 = linhas_pauta[-1]

    if (ymin <= l0 and ymax <= l4):
        return True

    return False


def beam_limite_y(ymin, ymax, linhas_pauta):
    l0 = linhas_pauta[0]
    l4 = linhas_pauta[-1]
    espacamento_y = (l4 - l0)
    # print(f"l0 {l0}")
    # print(f"l4 {l4}")
   # Exemplo: beam pode estar até uma altura além da pauta
    margem = espacamento_y * 1.5

    if ymin >= l0 - margem and ymax <= l4 + margem:
        return True  # beam abaixo

    return False


def y_para_pitch(y, linhas, x, clave="G"):
    """
    y -> coordenada vertical do centro da nota
    linhas -> lista [y1,y2,y3,y4,y5] da pauta (cima → baixo)
    x -> não usado aqui mas deixado para compatibilidade
    clave -> "G", "F", "C_alto", "C_tenor"
    """

    # -------------------------------
    # 1) Definir referência por clave
    # -------------------------------
    if clave == "G":      # clave de sol
        ref_nome = "G"
        ref_oitava = 4
        ref_linha = linhas[3]     # linha 2 de baixo → index 3

    elif clave == "F":    # clave de fá
        ref_nome = "F"
        ref_oitava = 3
        ref_linha = linhas[1]     # linha 4 de baixo → index 1

    elif clave == "C_alto":   # clave de dó alto
        ref_nome = "C"
        ref_oitava = 4
        ref_linha = linhas[2]     # linha central

    elif clave == "C_tenor":  # clave de dó tenor
        ref_nome = "C"
        ref_oitava = 4
        ref_linha = linhas[1]     # linha 4 de baixo → index 1
    
    else:
        raise ValueError(f"Clave não reconhecida: {clave}")

    # ---------------------------------
    # 2) Distância entre linhas/espacos
    # ---------------------------------
    espacamento = (linhas[-1] - linhas[0]) / 4

    # Cada passo = meio espaçamento (linha ou espaço)
    steps = round((ref_linha - y) / (espacamento/2))

    # ---------------------------------
    # 3) Escala cíclica a partir da referência
    # ---------------------------------
    nomes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

    # índice do nome da REF na escala
    idx_ref = nomes.index(ref_nome)

    # nome avançando steps
    nome = nomes[(idx_ref + steps) % 7]

    # Cálculo da oitava
    oitava = ref_oitava + ((idx_ref + steps) // 7)

    # (debug)
    print(f"[pitch] y={y}, clave={clave}, ref={ref_nome}{ref_oitava}, steps={steps}, nome={nome}, oitava={oitava}")

    return nome, oitava


# Função para estimar se uma nota tem beam próximo (para colcheia)


def tem_beam(xmin, xmax, ymin, ymax, beams,lp):
    for bxmin, bymin, bxmax, bymax in beams:

        if not beam_limite_y(ymin, ymax, lp):
            continue

        # Beam horizontal cobre essa nota horizontalmente
        if bxmin <= xmin <= bxmax + 10:
            largura_nota = xmax - xmin
            centro_nota = xmin + largura_nota / 2
            largura_beam = bxmax - bxmin

            # Distância do centro da nota em relação ao beam
            pos_relativa = (centro_nota - bxmin) / largura_beam

            # Decidir a posição da nota dentro do beam
            if pos_relativa < 0.25:
                return True, 'start'
            elif pos_relativa > 0.75:
                return True, 'stop'
            else:
                return True, 'continue'
    return False, None


# Função principal
def processar_yolo(yolo_linhas, linhas_pauta, tolerancia_staff=25):
    """
    yolo_linhas -> lista de linhas detectadas pelo YOLO
    linhas_pauta -> lista de listas com as 5 linhas de cada pauta
                    exemplo: [[y1,y2,y3,y4,y5], [y1,y2,y3,y4,y5], ...]
    tolerancia_staff -> margem vertical para notas fora do combostaff
    """

    notas = []
    beams = []
    cst = []                # lista de staffs: (xmin, ymin, xmax, ymax)
    sharped_notes = {}      # dict: staff_id -> set de notas sustenidas
    formula_comp = ""
    fcomp = 0

    # ------------------------------------------------------------------
    # 1) Coletar COMBOSTAFFS detectados
    # ------------------------------------------------------------------
    for simb in yolo_linhas:
        if 'cSt' in simb:
            parts = simb
            bbox = tuple(map(float, parts[4:8]))  # xmin, ymin, xmax, ymax
            cst.append(bbox)

    # se não tiver nenhum combostaff, não segue
    if not cst:
        print("Nenhum combostaff detectado!")
        return [], {}

    # Inicializa o dicionário de sustenidos por staff
    for i in range(len(cst)):
        sharped_notes[i] = set()

    # ------------------------------------------------------------------
    # Função: determina qual staff a nota pertence (mesmo se estiver fora)
    # ------------------------------------------------------------------
    def staff_da_nota(y_meio):
        for i, (xmin, ymin, xmax, ymax) in enumerate(cst):
            if (ymin - tolerancia_staff) <= y_meio <= (ymax + tolerancia_staff):
                return i
        # Se realmente está muito longe de todos, escolhe o mais próximo
        distancias = [abs((ymin+ymax)/2 - y_meio) for (_,ymin,_,ymax) in cst]
        return distancias.index(min(distancias))

    # ------------------------------------------------------------------
    # 2) Coletar BEAMS
    # ------------------------------------------------------------------
    for linha in yolo_linhas:
        if 'bea' in linha:
            parts = linha
            beams.append(tuple(map(float, parts[4:8])))

    # ------------------------------------------------------------------
    # 3) Aplicar sustenidos por staff (armadura de clave)
    # ------------------------------------------------------------------
    for linha in yolo_linhas:
        parts = linha
        if len(parts) < 8:
            continue

        classe = parts[2]
        if classe == 'kSh':    # sustenido
            xmin, ymin, xmax, ymax = map(float, parts[4:8])
            y_meio = (ymin + ymax) / 2

            st = staff_da_nota(y_meio)     # staff correto

            if nolimite(ymin, ymax, linhas_pauta[st]):
                print(f"Staff da nota : {st}")
                nome, oitava = y_para_pitch(y_meio, linhas_pauta[st], yolo_linhas[st])
                sharped_notes[st].add(nome)

                print(f"Sustenido aplicado em staff {st}: {nome}")
        
    # ------------------------------------------------------------------
    # 4) Processamento de NOTAS e RITMOS
    # ------------------------------------------------------------------
    print("Notas: ")
    for linha in yolo_linhas:
        parts = linha
        classe = parts[2]
        
        print(parts)

        if len(parts) < 8:
            continue

        xmin, ymin, xmax, ymax = map(float, parts[4:8])
        y_meio = (ymin + ymax) / 2

        # Descobrir staff
        st = staff_da_nota(y_meio)
        linhas_staff = linhas_pauta[st]

        # --------------------------------------------------------------
        # NOTA COMUM
        # --------------------------------------------------------------
        if classe == 'nBl':  # nota cheia (exemplo)
            nome, oitava = y_para_pitch(y_meio, linhas_staff, xmin)

            # aplica sustenido se existir naquele staff
            if nome in sharped_notes[st]:
                pitch = f"{nome}#{oitava}"
            else:
                pitch = f"{nome}{oitava}"

            # beam?
            tem_b, pos = tem_beam(xmin, xmax, ymin, ymax, beams, linhas_staff)
            dur = 0.5 if tem_b else 1.0

            notas.append({
                "pitch": pitch,
                "duration": dur,
                "beam": pos,
                "staff": st
            })

        # --------------------------------------------------------------
        # PAUSA
        # --------------------------------------------------------------
        elif classe == 'rQu':   # pausa de 1/4
            notas.append({
                "pitch": "rest",
                "duration": 1.0,
                "staff": st
            })
    print(f"sssssssssssssssaaaaaaa{sharped_notes}")
    return notas, sharped_notes


def gerar_stream(notas, sharped_notes=None, time_signature='4/4'):
    """
    notas -> lista contendo dicts:
              {
                "pitch": "C4" ou "rest",
                "duration": 1.0,
                "beam": "start"/"continue"/"stop" (opcional),
                "staff": id_da_pauta
              }

    sharped_notes -> dict { staff_id -> set(["F", "C", ...]) }
    """

    # Quantidade de pautas existentes
    num_staves = 1
    if notas:
        num_staves = max(n["staff"] for n in notas) + 1

    # Cria um Score com múltiplas partes
    score = stream.Score()

    # Criar as partes (1 por pauta)
    parts = []
    for i in range(num_staves):
        p = stream.Part()
        p.id = f"Staff_{i}"
        p.insert(0, metadata.Metadata())
        p.metadata.title = f"Pauta {i+1}"

        # fórmula de compasso
        ts = meter.TimeSignature(time_signature)
        p.append(ts)

        # armadura de clave (se houver sustenidos na pauta)
        if sharped_notes and i in sharped_notes and len(sharped_notes[i]) > 0:
            ks = key.KeySignature(len(sharped_notes[i]))
            p.append(ks)

        parts.append(p)
        score.append(p)

    # Inserir notas em cada Part corretamente
    for n in notas:
        st = n.get("staff", 0)  # staff da nota

        dur = n['duration']
        if n['pitch'] == 'rest':
            el = note.Rest(quarterLength=dur)
        else:
            el = note.Note(n['pitch'], quarterLength=dur)

        # beams
        if 'beam' in n and n['beam'] in {'start', 'continue', 'stop'}:
            b = Beams()
            b.append(Beam(n['beam']))
            el.beams = b

        # adiciona na pauta correta
        parts[st].append(el)

    return score


partes = []
pred = Predicao()
partes = pred.simbolos_detectados("./img")
#cSt
entrada_yolo = sorted(
    partes, key=lambda linha: float(linha[4]))

# print(entrada_yolo)

notas_detectadas, sharped_notes = processar_yolo(entrada_yolo, linhas_pauta)


musica = gerar_stream(notas_detectadas, sharped_notes)

musica.show('text')
musica.write('musicxml', fp='saida_partitura.xml')