import os
from music21 import stream, note, metadata, key, meter, clef, pitch, layout, chord, duration
from music21.beam import Beams, Beam

# Importação dos seus módulos locais (Mantidos conforme solicitado)
try:
    from detecta_linha import LinhaDetectorPartitura
    from tratamento_linha import Tratamento
    from tratamento1 import Trt1
    from predicaot import Predicao
except ImportError:
    print("Aviso: Módulos locais (detecta_linha, tratamento, etc) não encontrados. O código não executará sem eles.")

class LeitorPartitura:
    def __init__(self, output_dir="./saida_omr"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Mapeamento de Time Signatures
        self.TS_MAP = {
            'timeSig0': '4/4', 'timeSig1': '1/4', 'timeSig2': '2/4',
            'timeSig3': '3/4', 'timeSig4': '4/4', 'timeSig5': '5/4',
            'timeSig6': '6/8', 'timeSig7': '7/8', 'timeSig8': '8/8',
            'timeSig9': '9/8', 'timeSigCommon': '4/4', 'timeSigCutCommon': '2/2',
            'tS2': '2/4', 'tS3': '3/4', 'tS4': '4/4', 'tS6': '6/8', 
            'tS9': '9/8', 'tSC': '4/4'
        }

    # ----------------------------------------------------------------------
    # MÉTODOS ESTÁTICOS (Helpers de Geometria e Lógica Musical)
    # ----------------------------------------------------------------------

    @staticmethod
    def nolimite(ymin, ymax, linhas_pauta_unica, margem=10):
        if not linhas_pauta_unica: return False
        l0 = linhas_pauta_unica[0]
        l4 = linhas_pauta_unica[-1]
        return ymin >= l0 - margem and ymax <= l4 + margem

    @staticmethod
    def y_para_pitch(y, linhas, clave="G"):
        """Converte a coordenada Y em uma nota musical baseada na Clave e nas linhas da pauta."""
        if clave == "G":
            ref_nome, ref_oitava, ref_linha = "G", 4, linhas[3] 
        elif clave == "F":
            ref_nome, ref_oitava, ref_linha = "F", 3, linhas[1] 
        else:
            ref_nome, ref_oitava, ref_linha = "G", 4, linhas[3]

        if not linhas: return "C", 4

        espacamento = (linhas[-1] - linhas[0]) / 4
        passo = espacamento / 2 
        steps = round((ref_linha - y) / passo)

        nomes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        idx_ref = nomes.index(ref_nome)
        indice_global = idx_ref + steps
        nome = nomes[indice_global % 7]
        oitava = ref_oitava + (indice_global // 7)
        oitava = max(0, min(8, oitava))
        return nome, oitava

    @staticmethod
    def detectar_staff(y, cst):
        """Retorna o índice da pauta baseado no Y do objeto"""
        if not cst: return 0
        for i, (_, ymin, _, ymax) in enumerate(cst):
            margem = 15 
            if (ymin - margem) <= y <= (ymax + margem):
                return i
        return min(range(len(cst)), key=lambda i: abs(((cst[i][1] + cst[i][3]) / 2) - y))

    @staticmethod
    def tem_beam(xmin, xmax, ymin, ymax, beams, staff_da_nota):
        for beam in beams:
            if beam['staff'] != staff_da_nota: continue
            bxmin, _, bxmax, _ = beam['bbox']
            
            # Lógica de Colisão Horizontal
            if (xmin <= bxmax + 10) and (xmax >= bxmin - 10):
                centro_nota = (xmin + xmax) / 2
                largura_beam = (bxmax - bxmin) or 1
                pos = (centro_nota - bxmin) / largura_beam

                if pos < 0.25: return True, 'start'
                elif pos > 0.75: return True, 'stop'
                else: return True, 'continue'
        return False, None

    @staticmethod
    def bbox_colidem(box_nota, box_flag, margem_x=5, margem_y=60):
        nx1, ny1, nx2, ny2 = box_nota
        fx1, fy1, fx2, fy2 = box_flag
        
        # Expansão da caixa da nota para buscar a flag
        nota_y1, nota_y2 = ny1 - margem_y, ny2 + margem_y
        nota_x1, nota_x2 = nx1 - margem_x, nx2 + margem_x

        if (nota_x2 < fx1) or (nota_x1 > fx2) or (nota_y2 < fy1) or (nota_y1 > fy2):
            return False
        return True

    # ----------------------------------------------------------------------
    # PIPELINE PRINCIPAL
    # ----------------------------------------------------------------------

    def processar_imagem(self, caminho_imagem, gerar_midi=True, gerar_xml=True):
            # Faz o fluxo direto sem pausa
            yolo, pauta, vert = self.obter_dados_brutos(caminho_imagem)
            return self.gerar_score_de_dados(yolo, pauta, vert, caminho_imagem)

    def _executar_deteccao_visual(self, caminho_imagem):
        """Roda os tratamentos de imagem e as redes neurais."""
        nome_arq = os.path.basename(caminho_imagem)
        saida_tratamento = os.path.join(self.output_dir, f"trt_{nome_arq}")
        pasta_linhas = os.path.join(self.output_dir, "linhas")

        # Tratamento de Imagem
        #Tratamento().preprocess_partitura(caminho_imagem, saida_tratamento)

        # Detecção YOLO
        pred = Predicao()
        # Assumindo que pred.simbolos_detectados aceita o diretório onde a img tratada está
        partes = pred.simbolos_detectados(os.path.dirname(caminho_imagem))
        yolo_linhas = sorted(partes, key=lambda l: float(l[4])) 

        # Detecção de Linhas (Pautas)
        detector = LinhaDetectorPartitura(
            yolo_linhas, caminho_imagem, salvar_resultado=True, pasta_saida=pasta_linhas
        )
        _, linhas_verticais, linhas_pauta = detector.processar()
        
        print(f"Pautas detectadas: {len(linhas_pauta)}")
        return yolo_linhas, linhas_pauta, linhas_verticais

    def _interpretar_yolo(self, yolo_linhas, linhas_pauta):
        """Converte as caixas do YOLO em dicionários de notas e metadados."""
        notas = []
        beams = []
        cst = []
        pontos = []
        lista_flags = []
        
        # Estruturas por Pauta
        # Inicialização dinâmica baseada nas pautas físicas detectadas
        if linhas_pauta:
            # Cria Bounding Box para cada pauta baseada nas linhas detectadas
            for p in linhas_pauta:
                cst.append((0, p[0]-20, 10000, p[-1]+20))
        else:
            # Fallback: tenta achar cSt no YOLO se a detecção de linha falhou
            for simb in yolo_linhas:
                if len(simb) > 2 and simb[2] == 'cSt':
                    cst.append(tuple(map(float, simb[4:8])))
            cst.sort(key=lambda b: (b[1] + b[3]) / 2)

        num_staves = len(cst)
        sharped_notes = {i: set() for i in range(num_staves)}
        flat_notes = {i: set() for i in range(num_staves)}
        time_signatures = {i: (None, float('inf')) for i in range(num_staves)}
        staff_claves = {i: 'G' for i in range(num_staves)} # Default G

        # 1. Extração de Beams e TimeSignatures (Passada Inicial)
        for linha in yolo_linhas:
            if len(linha) < 8: continue
            classe = linha[2]
            xmin, ymin, xmax, ymax = map(float, linha[4:8])
            y_meio = (ymin + ymax) / 2
            st = self.detectar_staff(y_meio, cst)

            if classe == 'bea' and st < len(linhas_pauta):
                beams.append({'bbox': (xmin, ymin, xmax, ymax), 'staff': st})
            
            elif classe in self.TS_MAP:
                # Lógica de Time Signature
                if st < num_staves:
                    ts_atual, x_atual = time_signatures[st]
                    if x_atual == float('inf') or xmin < x_atual:
                        time_signatures[st] = (self.TS_MAP[classe], xmin)

        # 2. Extração de Objetos Principais (Notas, Pausas, Claves, Acidentes)
        yolo_linhas.sort(key=lambda x: float(x[4])) # Ordena por X
        
        for linha in yolo_linhas:
            if len(linha) < 8: continue
            classe = linha[2]
            xmin, ymin, xmax, ymax = map(float, linha[4:8])
            y_meio = (ymin + ymax) / 2
            st = self.detectar_staff(y_meio, cst)

            # Validação de segurança
            if st >= len(linhas_pauta) or len(linhas_pauta[st]) != 5: continue
            linhas_staff = linhas_pauta[st]

            # --- CLAVES E ARMADURAS ---
            if classe == 'kSh':
                nome, _ = self.y_para_pitch(y_meio, linhas_staff, clave=staff_claves[st])
                sharped_notes[st].add(nome)
            elif classe == 'kFl':
                nome, _ = self.y_para_pitch(y_meio, linhas_staff, clave=staff_claves[st])
                flat_notes[st].add(nome)
            elif classe == 'fCl':
                staff_claves[st] = 'F' # Atualiza estado da pauta para Clave de Fá

            # --- NOTAS E PAUSAS ---
            elif classe == 'nHa': # Mínima
                self._add_nota(notas, 'nHa', xmin, y_meio, linhas_staff, staff_claves[st], sharped_notes[st], flat_notes[st], st, 2.0, None)
            elif classe == 'nWh': # Semibreve
                self._add_nota(notas, 'nWh', xmin, y_meio, linhas_staff, staff_claves[st], sharped_notes[st], flat_notes[st], st, 4.0, None)
            elif classe == 'nBl': # Nota Preta (Seminima/Colcheia)
                # Verifica Beam
                tem_b, pos_beam = self.tem_beam(xmin, xmax, ymin, ymax, beams, st)
                dur = 0.5 if tem_b else 1.0
                bbox = [xmin, ymin, xmax, ymax]
                self._add_nota(notas, 'nBl', xmin, y_meio, linhas_staff, staff_claves[st], sharped_notes[st], flat_notes[st], st, dur, pos_beam, bbox)
            
            elif classe == 'r08': # Pausa Colcheia
                notas.append({"pitch": "rest", "duration": 0.5, "staff": st, "x": xmin})
            elif classe == 'rQu': # Pausa Seminima
                notas.append({"pitch": "rest", "duration": 1.0, "staff": st, "x": xmin})
            
            # --- MODIFICADORES ---
            elif classe == 'f8U': # Bandeirola (Flag)
                lista_flags.append({"staff": st, "bbox": [xmin, ymin, xmax, ymax]})
            elif classe == 'aDo': # Ponto de aumento
                pontos.append({'staff': st, 'bbox': [xmin, ymin, xmax, ymax], 'y_center': y_meio})

        # 3. Pós-Processamento: Colisões
        self._aplicar_bandeiras(notas, lista_flags)
        self._aplicar_pontos(notas, pontos)

        return {
            'notas': notas,
            'sharped_notes': sharped_notes,
            'flat_notes': flat_notes,
            'time_signatures': time_signatures,
            'staff_claves': staff_claves,
            'num_staves': num_staves
        }

    def obter_dados_brutos(self, caminho_imagem):
        """Roda o YOLO e OpenCV e retorna as listas cruas para edição."""
        print(f"--- Detectando elementos em: {caminho_imagem} ---")
        # Chama seu método interno existente
        yolo_linhas, linhas_pauta, linhas_verticais = self._executar_deteccao_visual(caminho_imagem)
        return yolo_linhas, linhas_pauta, linhas_verticais

   
    def gerar_score_de_dados(self, yolo_linhas, linhas_pauta, linhas_verticais, caminho_original):
        """Recebe os dados (possivelmente editados) e gera o objeto Music21."""
        print("--- Interpretando dados e gerando partitura ---")
        print(f"--- Gerando partitura com {len(yolo_linhas)} elementos editados ---")
        # Interpretação Lógica
        dados_musicais = self._interpretar_yolo(yolo_linhas, linhas_pauta)
        
        # Geração do Stream
        score = self._gerar_music21_stream(dados_musicais)
        #print(f"resultados caminho {caminho_original}")
        # Salvamento automático (conforme lógica original)
        self._salvar_resultados(score, caminho_original, midi=True, xml=True)
        
        return score

    def _add_nota(self, lista, tipo, x, y, linhas, clave, sharps, flats, st, dur, beam, bbox=None):
        nome, oitava = self.y_para_pitch(y, linhas, clave=clave)
        acidente = None
        if nome in sharps: acidente = 'sharp'
        elif nome in flats: acidente = 'flat'
        
        lista.append({
            "step": nome, "octave": oitava, "accidental": acidente,
            "duration": dur, "beam": beam, "staff": st, "x": x, "bbox": bbox
        })

    def _aplicar_bandeiras(self, notas, flags):
        for flag in flags:
            candidatas = [n for n in notas if n.get('staff') == flag['staff'] and n.get('pitch') != 'rest']
            for nota in candidatas:
                if nota['duration'] == 1.0 and nota['bbox']:
                    if self.bbox_colidem(nota['bbox'], flag['bbox'], margem_x=1, margem_y=1):
                        nota['duration'] = 0.5
                        break

    def _aplicar_pontos(self, notas, pontos):
        for ponto in pontos:
            candidatas = [n for n in notas if n['staff'] == ponto['staff'] and n.get('pitch') != 'rest']
            melhor_nota = None
            menor_distancia = float('inf')
            
            for nota in candidatas:
                if not nota.get('bbox'): continue
                nota_direita = nota['bbox'][2]
                nota_y = (nota['bbox'][1] + nota['bbox'][3]) / 2
                
                # Mesma altura visual e nota à esquerda do ponto
                if abs(nota_y - ponto['y_center']) < 15:
                    dist = ponto['bbox'][0] - nota_direita
                    if 0 < dist < 40 and dist < menor_distancia:
                        menor_distancia = dist
                        melhor_nota = nota
            
            if melhor_nota:
                melhor_nota['duration'] *= 1.5

    # ----------------------------------------------------------------------
    # GERAÇÃO MUSIC21
    # ----------------------------------------------------------------------

    def _gerar_music21_stream(self, dados):
        """Constrói o objeto Score completo."""
        notas = dados['notas']
        staff_claves = dados['staff_claves']
        num_staves = dados['num_staves']
        
        score = stream.Score()
        
        # Detecta estrutura (Ex: Piano 2 pautas vs Voz 1 pauta)
        # Se houver 2 pautas e as claves forem G e F, assumimos Piano Grand Staff
        eh_piano = (num_staves == 2 and staff_claves.get(0) == 'G' and staff_claves.get(1) == 'F')
        
        partes_obj = []
        for i in range(num_staves):
            p = stream.Part(id=f"Staff_{i}")
            
            # Adiciona Clave inicial
            c_str = staff_claves.get(i, 'G')
            if c_str == 'F': p.append(clef.BassClef())
            else: p.append(clef.TrebleClef())
            
            # Adiciona Armadura
            sharps = len(dados['sharped_notes'].get(i, []))
            flats = len(dados['flat_notes'].get(i, []))
            if sharps > 0: p.append(key.KeySignature(sharps))
            elif flats > 0: p.append(key.KeySignature(-flats))
            
            # Adiciona Fórmula de Compasso
            ts_val = dados['time_signatures'].get(i, (None, None))[0]
            if ts_val: p.append(meter.TimeSignature(ts_val))
            
            partes_obj.append(p)

        # Popula as partes com as notas
        notas.sort(key=lambda n: n['x']) # Ordena temporalmente
        
        for n in notas:
            st_idx = n['staff']
            if st_idx >= len(partes_obj): continue
            
            current_part = partes_obj[st_idx]
            
            if n.get("pitch") == "rest":
                el = note.Rest()
            else:
                el = note.Note()
                el.pitch.step = n["step"]
                el.pitch.octave = n["octave"]
                if n.get("accidental"):
                    el.pitch.accidental = pitch.Accidental(n["accidental"])
            
            el.quarterLength = n["duration"]
            current_part.append(el)

        # Montagem do Layout
        if eh_piano:
            # Cria Grand Staff
            staff_group = layout.StaffGroup(partes_obj, symbol='brace', barTogether='yes')
            score.append(staff_group)
            score.append(partes_obj[0])
            score.append(partes_obj[1])
        else:
            # Adiciona partes individualmente
            for p in partes_obj:
                score.append(p)
                
        # Finalização (Barras de compasso)
        for p in partes_obj:
            try:
                p.makeMeasures(inPlace=True)
            except:
                pass

        return score

   
    def alterar_metadados_manual(self, score, compasso_num, novo_ts=None, nova_armadura_alteracoes=None, staff_index=0):
        """
        Altera fórmula de compasso ou armadura em um compasso específico.
        novo_ts: string (ex: '3/4')
        nova_armadura_alteracoes: int (ex: -2 para 2 bemóis, 1 para 1 sustenido, 0 para Do Maior)
        """
        try:
            
            parte = score.parts[staff_index]
            compassos = parte.getElementsByClass('Measure')
            
            if compasso_num > len(compassos):
                print("Erro: Compasso não existe.")
                return

            # O Music21 usa índice 0 para compasso 1 na lista, mas é mais seguro buscar pelo número
            alvo = compassos[compasso_num - 1]

            # 1. Alterar Fórmula de Compasso (Time Signature)
            if novo_ts:
                ts = meter.TimeSignature(novo_ts)
                # Remove TS existente se houver e insere o novo no início
                for el in alvo.getElementsByClass(meter.TimeSignature):
                    alvo.remove(el)
                alvo.insert(0.0, ts)
                print(f"✅ Compasso {compasso_num}: Mudado para {novo_ts}")

            # 2. Alterar Armadura (Key Signature)
            if nova_armadura_alteracoes is not None:
                ks = key.KeySignature(nova_armadura_alteracoes)
                for el in alvo.getElementsByClass(key.KeySignature):
                    alvo.remove(el)
                alvo.insert(0.0, ks)
                print(f"✅ Compasso {compasso_num}: Armadura alterada ({nova_armadura_alteracoes})")

        except Exception as e:
            print(f"❌ Erro ao alterar metadados: {e}")

    def inserir_simbolo_manual(self, score, compasso_num, tipo, valor, batida, duracao=1.0, staff_index=0):
    
        try:
            parte = score.parts[staff_index]
            compassos = parte.getElementsByClass(stream.Measure)
            
            if compasso_num > len(compassos):
                print(f"Erro: A partitura só tem {len(compassos)} compassos.")
                return

            alvo = compassos[compasso_num - 1]

            if tipo.lower() == 'nota':
                novo_item = note.Note(valor)
            else:
                novo_item = note.Rest()
            
            novo_item.duration = duration.Duration(duracao)
            alvo.insert(batida, novo_item)
            alvo.makeBeams(inPlace=True)
            print(f"✅ Inserido manualmente: {tipo} {valor if valor else ''} no Compasso {compasso_num}")
        except Exception as e:
            print(f"❌ Erro na inserção manual: {e}")

    def _salvar_resultados(self, score, caminho_original, midi=True, xml=True):
        base_name = os.path.splitext(os.path.basename(caminho_original))[0]
        
        if xml:
            path_xml = os.path.join(self.output_dir, f"{base_name}.musicxml")
            score.write('musicxml', fp=path_xml)
            print(f"Salvo MusicXML: {path_xml}")
            
        if midi:
            path_midi = os.path.join(self.output_dir, f"{base_name}.mid")
            score.write('midi', fp=path_midi)
            print(f"Salvo MIDI: {path_midi}")

# ----------------------------------------------------------------------
# EXEMPLO DE USO
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Instancia a classe
    ocr = LeitorPartitura(output_dir="./resultados_teste")
    
    # Processa uma imagem
    caminho = './tratamento/partitura_teste_1.png'
    
    if os.path.exists(caminho):
        partitura_stream = ocr.processar_imagem(caminho)
        
        # Opcional: Mostrar na tela (Requer MuseScore ou similar instalado)
        # partitura_stream.show()
    else:
        print("Imagem de teste não encontrada.")