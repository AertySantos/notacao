import os
import csv
from pathlib import Path
from difflib import SequenceMatcher
from tratamento1 import Trt1
from main7 import LeitorPartitura
import shutil
# --- CONFIGURAÇÕES ---
ARQUIVO_RELATORIO = "relatorio_comparacao.csv"
ARQUIVO_DETECCOES_TXT = "deteccoes_completa.txt" # O script vai LER deste arquivo
PASTA_IMAGENS = "package_aa"

# --- MAPEAMENTO (YOLO -> AGNOSTIC) ---
MAPA_YOLO_AGNOSTIC = {
    'gCl': 'clef', 'fCl': 'clef', 'cCl': 'clef',
    'kSh': 'accidental', 'kFl': 'accidental', 'kNa': 'accidental',
    'tS3': 'digit', 'tS4': 'digit', 'tS2': 'digit', 'tS1': 'digit',
    'nBl': 'note', 'nHa': 'note', 'nWh': 'note', 
    'bea': 'beam', 
    'aDo': 'dot', 
    'bSi': 'barline',
    'rQr': 'rest', 'rHa': 'rest', 'rWh': 'rest'
}

def inicializar_csv():
    """Cria o arquivo CSV e escreve o cabeçalho."""
    with open(ARQUIVO_RELATORIO, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Arquivo", 
            "Similaridade (%)", 
            "Qtd Gabarito", 
            "Qtd Detectado", 
            "Status",
            "Gabarito (Início)",
            "Detectado (Início)"
        ])

def adicionar_linha_csv(dados):
    """Salva uma linha no CSV (Modo Append)."""
    with open(ARQUIVO_RELATORIO, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(dados)

def carregar_agnostic(caminho_agnostic):
    """Lê o gabarito .agnostic e simplifica os tokens."""
    if not os.path.exists(caminho_agnostic):
        return None
    try:
        with open(caminho_agnostic, 'r', encoding='utf-8') as f:
            tokens = f.read().strip().replace('\t', ' ').split(' ')
            lista_limpa = []
            for t in tokens:
                if t.strip():
                    # Ex: 'clef.G-L2' vira 'clef'
                    base = t.split('.')[0].split('-')[0]
                    lista_limpa.append(base)
            return lista_limpa
    except Exception as e:
        print(f"Erro ao ler agnostic: {e}")
        return None

def carregar_deteccoes_do_txt(caminho_txt, nome_imagem_busca):
    """
    Lê o arquivo de texto JÁ EXISTENTE, filtra pela imagem e ordena por X.
    """
    deteccoes = []
    if not os.path.exists(caminho_txt):
        print(f"Aviso: Arquivo {caminho_txt} não encontrado.")
        return []

    with open(caminho_txt, 'r') as f:
        for linha in f:
            partes = linha.split()
            # Verifica se o nome da imagem está na linha
            # Ex: se buscamos 'music_01' ele acha 'music_01.png' ou 'temp_music_01.png'
            if len(partes) > 4 and nome_imagem_busca in partes[0]:
                classe_yolo = partes[2]
                try:
                    pos_x = float(partes[4]) # Coluna do X
                except: continue
                
                token = MAPA_YOLO_AGNOSTIC.get(classe_yolo, 'unknown')
                
                if token != 'unknown':
                    deteccoes.append((pos_x, token))

    # Ordena da esquerda para a direita (Fundamental para música)
    deteccoes.sort(key=lambda item: item[0])
    
    return [d[1] for d in deteccoes]

def processar_lote_musical():
    path_raiz = Path(PASTA_IMAGENS)
    
    # 1. Inicializa o CSV
    inicializar_csv()
    
    arquivos = list(path_raiz.rglob("*.png"))
    print(f"Iniciando processamento de {len(arquivos)} arquivos...")

    for i, arquivo_path in enumerate(arquivos):
        if arquivo_path.name.startswith("._"): continue

        print(f"\n[{i+1}/{len(arquivos)}] Processando: {arquivo_path.name}")
       # print("ddddddddddddddddddd")
        try:
            # --- A. TRATAMENTO DE IMAGEM ---
            os.makedirs("tratamento", exist_ok=True)
            caminho_entrada = str(arquivo_path)
            caminho_temp = f"tratamento/temp_{arquivo_path.name}"
            #print("ddddddddddddddddddd2")
            trt = Trt1()
            trt.preprocess_partitura(caminho_entrada, caminho_temp)
            #print("ddddddddddddddddddd3")
            # --- B. EXECUÇÃO DO OMR (Geração do XML) ---
            omr = LeitorPartitura()
            score_obj = omr.processar_imagem(caminho_temp)
            print("apaga arquivo")
            
            #apagar arquivo tratado
            if os.path.exists("tratamento"):
                shutil.rmtree("tratamento")

            os.makedirs("tratamento", exist_ok=True)

            os.makedirs("resultados", exist_ok=True)
            nome_xml = f"resultados/{arquivo_path.stem}.musicxml"
            score_obj.write('musicxml', fp=nome_xml)
            
            # print(f"   -> XML gerado: {nome_xml}")

            # --- C. COMPARAÇÃO (Lendo do TXT existente) ---
            # Carrega o gabarito
            gabarito = carregar_agnostic(arquivo_path.with_suffix(".agnostic"))
            
            # Carrega as detecções do arquivo txt geral
            # Usa o .stem (nome sem extensão) para garantir match no arquivo de texto
            deteccoes = carregar_deteccoes_do_txt(ARQUIVO_DETECCOES_TXT, arquivo_path.stem)
            
            score_sim = 0.0
            status = "Erro"
            #print("ddddddddddddddddddd4")
            if gabarito and deteccoes:
                matcher = SequenceMatcher(None, gabarito, deteccoes)
                score_sim = matcher.ratio() * 100
                status = "Sucesso"
                print(f"   -> Similaridade: {score_sim:.2f}%")
            elif not gabarito:
                status = "Sem Agnostic"
                print("   -> Aviso: .agnostic não encontrado")
            elif not deteccoes:
                status = "Sem Deteccoes no TXT"
                print("   -> Aviso: Nenhuma detecção encontrada no arquivo txt")
            #print("ddddddddddddddddddd5")
            # --- D. SALVA NO CSV ---
            adicionar_linha_csv([
                arquivo_path.name,
                f"{score_sim:.2f}",
                len(gabarito) if gabarito else 0,
                len(deteccoes) if deteccoes else 0,
                status,
                " ".join(gabarito[:3]) if gabarito else "",
                " ".join(deteccoes[:3]) if deteccoes else ""
            ])
            
        except Exception as e:
            print(f"   -> Erro ao processar: {e}")
            adicionar_linha_csv([arquivo_path.name, 0, 0, 0, f"Erro: {str(e)}", "", ""])

if __name__ == "__main__":
    processar_lote_musical()