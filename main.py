from music21 import stream, note, metadata, key, meter
from music21.beam import Beams, Beam
from detecta_linha import LinhaDetectorPartitura
from predicao import Predicao

# Caminho da imagem a ser processada
caminho_imagem = 'teste/partitura_tratada_300dpi.jpg'

# Cria o detector
detector = LinhaDetectorPartitura(caminho_imagem, salvar_resultado=True)

# Processa e obtém as linhas detectadas
linhas_horizontais, linhas_verticais, ydistintos = detector.processar()

print(ydistintos)

# Linhas da pauta (de cima para baixo), conforme fornecido
linhas_pauta = ydistintos

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


def y_para_pitch(y, linhas, x):
    # segunda linha de cima para baixo (clave de sol = G4)
    referencia = linhas[3]
    espacamento = (linhas[-1] - linhas[0]) / 4
    steps = round((referencia - y) / (espacamento/2))
    nomes = ['G', 'A', 'B', 'C', 'D', 'E', 'F']
    oitava = 4 + (steps + 4) // 7
    nome = nomes[int(steps % len(nomes))]

    print(
        f"y:{y}, x:{x}, ref:{referencia},espacamento:{espacamento},steps:{steps},oitava:{oitava}, nome:{nome}")
    return nome, oitava

# Função para estimar se uma nota tem beam próximo (para colcheia)


def tem_beam(xmin, xmax, ymin, ymax, beams):
    for bxmin, bymin, bxmax, bymax in beams:

        if not beam_limite_y(ymin, ymax, linhas_pauta):
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


def processar_yolo(yolo_linhas):
    notas = []
    beams = []
    sharped_notes = set()  # Armazena quais notas devem ser sustenidas
    formula_comp = ""
    fcomp = 0

    # Parse beams antes
    for linha in yolo_linhas:
        if 'beam' in linha:
            parts = linha
            beams.append(tuple(map(float, parts[4:8])))
            # print(f"{linha}")

    # Coleta alterações de armadura de clave
    for linha in yolo_linhas:
        parts = linha
        if len(parts) < 8:
            continue
        classe = parts[2]
        if classe == 'keySharp':
            xmin, ymin, xmax, ymax = map(float, parts[4:8])
            if nolimite(ymin, ymax, linhas_pauta):
                nome, oitava = y_para_pitch(
                    (ymin + ymax) / 2, linhas_pauta, xmin)
                # nome = pitch[:-1]  # ex: "F" de "F4"
                sharped_notes.add(nome)
                print(f"Sustenido aplicado à nota: {nome}")

    for linha in yolo_linhas:
        parts = linha

        classe = parts[2]
        xmin, ymin, xmax, ymax = map(float, parts[4:8])

        if classe == 'noteheadBlack':

            # print(f"{(ymin+ymax)/2},{xmin}")
            nome, oitava = y_para_pitch((ymin+ymax)/2, linhas_pauta, xmin)

            if nome in sharped_notes:
                pitch = f"{nome}#{oitava}"
            else:
                pitch = f"{nome}{oitava}"

            # print(pitch)
            tem_b, pos = tem_beam(xmin, xmax, ymin, ymax, beams)

            if tem_b:
                # Suponha que você tenha acesso à lista de notas dentro de um beam
                beam_status = pos
                dur = 0.5
                # print(pos)
            else:
                beam_status = None
                dur = 1.0

            notas.append({
                'pitch': pitch,
                'duration': dur,
                'beam': beam_status
            })

        elif classe == 'restQuarter':
            notas.append({'pitch': 'rest', 'duration': 1.0})

        # elif classe == 'timeSig4': corrigir
           # if fcomp == 0:
            # fcomp = ymin
    print(f"sssssssssssssssaaaaaaa{sharped_notes}")
    return notas, sharped_notes

# Gera music21 Stream


def gerar_stream(notas, sharped_notes=None, time_signature='4/4'):
    s = stream.Part()  # usar Part para permitir compassos
    s.insert(0, metadata.Metadata())
    s.metadata.title = "Partitura Gerada"

    # Define fórmula de compasso
    ts = meter.TimeSignature(time_signature)
    s.append(ts)

    # Define a armadura de clave, se houver sustenidos
    if sharped_notes:
        # Cria armadura de clave a partir do número de sustenidos
        ks = key.KeySignature(len(sharped_notes))
        print(f"ks:{ks}")
        s.append(ks)

    # Adiciona as notas ao stream
    for n in notas:
        dur = n['duration']
        if n['pitch'] == 'rest':
            el = note.Rest(quarterLength=dur)
        else:
            el = note.Note(n['pitch'], quarterLength=dur)

        # Aplica informação de beam, se presente
        if 'beam' in n and n['beam'] in {'start', 'continue', 'stop'}:
            b = Beams()
            b.append(Beam(n['beam']))
            el.beams = b

        s.append(el)

    return s


partes = []
pred = Predicao()
partes = pred.simbolos_detectados("./img")

entrada_yolo = sorted(
    partes, key=lambda linha: float(linha[4]))

# print(entrada_yolo)

notas_detectadas, sharped_notes = processar_yolo(entrada_yolo)
musica = gerar_stream(notas_detectadas, sharped_notes)

musica.show('text')
musica.write('musicxml', fp='saida_partitura.xml')
