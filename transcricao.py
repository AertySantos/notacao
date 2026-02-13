import os
import shutil
from pathlib import Path
import copy  # Importante para não alterar a lista original sem querer antes da hora

# Suas importações
from tratamento1 import Trt1
from main7 import LeitorPartitura

# --- FUNÇÃO DE EDIÇÃO (Mantida igual, apenas para referência) ---
import os
import shutil
import copy
from pathlib import Path

# Suas importações
from tratamento1 import Trt1
from main7 import LeitorPartitura

def modificar_dados_yolo(yolo_linhas):
    """
    Menu interativo para edição total:
    - [d] Deletar
    - [c] Alterar Classe
    - [x] Alterar Coordenadas (x1, y1, x2, y2)
    - [a] Adicionar Novo
    """
    
    # Nome do arquivo para uso em novos itens
    nome_padrao = yolo_linhas[0][0] if yolo_linhas else "manual.png"

    while True:
        # Ordena por X (esquerda -> direita) para facilitar leitura
        yolo_linhas.sort(key=lambda k: float(k[4]))

        print("\n" + "="*90)
        print(f"🔧 EDITOR DE DADOS (Total: {len(yolo_linhas)})")
        print("="*90)
        print(f"{'IDX':<4} | {'CLASSE':<6} | {'CONF':<5} | {'COORDENADAS (x1, y1, x2, y2)':<40}")
        print("-" * 90)

        for i, item in enumerate(yolo_linhas):
            classe = item[2]
            conf = f"{float(item[3]):.2f}"
            
            # Pega coords
            x1, y1 = float(item[4]), float(item[5])
            x2, y2 = float(item[6]), float(item[7])
            
            coords_str = f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
            print(f"{i:<4} | {classe:<6} | {conf:<5} | {coords_str:<40}")

        print("-" * 90)
        print("COMANDOS:")
        print(" [d N]      -> Deletar item N (ex: 'd 5')")
        print(" [c N]      -> Mudar classe (ex: 'c 5')")
        print(" [x N]      -> Editar COORDENADAS (x1,y1,x2,y2) do item N")
        print(" [a]        -> Adicionar novo símbolo")
        print(" [ENTER]    -> Salvar e Sair")
        
        entrada = input("\nComando > ").strip().lower()

        if entrada == "":
            break
        
        try:
            partes = entrada.split()
            cmd = partes[0]

            # --- ADICIONAR (a) ---
            if cmd == 'a':
                print("\n--- Adicionar Novo ---")
                nova_classe = input("Classe (ex: nBl): ").strip()
                if not nova_classe: continue
                
                print("Digite as coordenadas (ou Enter para cancelar):")
                try:
                    nx1 = float(input("x1 (Esq): "))
                    ny1 = float(input("y1 (Topo): "))
                    nx2 = float(input("x2 (Dir): "))
                    ny2 = float(input("y2 (Base): "))
                    
                    novo_item = [
                        nome_padrao, '999', nova_classe, '1.0000',
                        str(nx1), str(ny1), str(nx2), str(ny2)
                    ]
                    yolo_linhas.append(novo_item)
                    print(f"✅ Adicionado!")
                except:
                    print("Cancelado.")
                continue

            # Validação de índice para comandos d, c, x
            if len(partes) < 2:
                print("⚠️ Falta o índice (ex: x 3)")
                continue
            
            idx = int(partes[1])
            if idx < 0 or idx >= len(yolo_linhas):
                print("❌ Índice inválido.")
                continue

            # --- DELETAR (d) ---
            if cmd == 'd':
                removido = yolo_linhas.pop(idx)
                print(f"🗑️ Removido: {removido[2]}")
            
            # --- MUDAR CLASSE (c) ---
            elif cmd == 'c':
                nova = input(f"Nova classe para '{yolo_linhas[idx][2]}': ").strip()
                if nova:
                    yolo_linhas[idx][2] = nova
                    print(f"✏️ Alterado para {nova}")
            
            # --- EDITAR COORDENADAS (x) ---
            elif cmd == 'x':
                curr = yolo_linhas[idx]
                c_x1, c_y1 = float(curr[4]), float(curr[5])
                c_x2, c_y2 = float(curr[6]), float(curr[7])
                
                print(f"\nEditando '{curr[2]}' (Index {idx})")
                print("Digite novo valor ou pressione ENTER para manter o atual:")
                
                def ler_coord(msg, valor_atual):
                    val = input(f"{msg} [{valor_atual:.1f}]: ").strip()
                    return str(float(val)) if val else str(valor_atual)

                # Atualiza um por um
                yolo_linhas[idx][4] = ler_coord("x1 (Esq)", c_x1)
                yolo_linhas[idx][5] = ler_coord("y1 (Topo)", c_y1)
                yolo_linhas[idx][6] = ler_coord("x2 (Dir)", c_x2)
                yolo_linhas[idx][7] = ler_coord("y2 (Base)", c_y2)
                
                print("✅ Coordenadas atualizadas.")

            else:
                print("⚠️ Comando inválido.")

        except ValueError:
            print("❌ Erro: digite números válidos.")
        except Exception as e:
            print(f"❌ Erro: {e}")

    return yolo_linhas

# --- FLUXO PRINCIPAL ---
def processar_fluxo_com_edicao(caminho_imagem):
    img_path = Path(caminho_imagem)
    nome_base = img_path.stem
    
    # 1. TRATAMENTO
    os.makedirs("tratamento", exist_ok=True)
    os.makedirs("resultados", exist_ok=True)
    saida_temp = f"tratamento/temp_{img_path.name}"
    
    print(f"--> Pré-processando imagem...")
    trt = Trt1()
    trt.preprocess_partitura(str(img_path), saida_temp)
    
    # 2. OMR: OBTER DADOS BRUTOS
    ocr = LeitorPartitura()
    # Pega as listas cruas do YOLO
    yolo_linhas, linhas_pauta, linhas_verticais = ocr.obter_dados_brutos(saida_temp)
    
    # ---------------------------------------------------------
    # PASSO A: GERAR XML "SUJO" (BRUTO) PARA VISUALIZAÇÃO
    # ---------------------------------------------------------
    print("\nAttempting to generate RAW XML for inspection...")
    
    # Fazemos uma cópia profunda para garantir que a geração não estrague a lista original
    # caso o Music21 modifique algo internamente durante o processamento.
    lista_para_bruto = copy.deepcopy(yolo_linhas)
    
    try:
        # Gera o score baseado nos dados atuais (com erros)
        ocr.gerar_score_de_dados(lista_para_bruto, linhas_pauta, linhas_verticais, saida_temp)
        
        # O LeitorPartitura salva como "{nome}.musicxml" na pasta de saída padrão.
        # Vamos renomear para _BRUTO para não confundir.
        arquivo_gerado = f"resultados/temp_{img_path.name.replace('.png','')}.musicxml" # Ajuste conforme sua saída do main7
        # Se sua main7 salva na pasta definida no init, ajuste aqui. 
        # Supondo que main7 salva em ./resultados ou ./saida_omr com o nome do arquivo temp:
        
        # Vamos garantir o caminho correto baseado no seu código main7:
        # Se main7 usa o nome do arquivo de entrada (temp_pagina_1.png) -> temp_pagina_1.musicxml
        caminho_padrao_xml = f"resultados/{Path(saida_temp).stem}.musicxml"
        caminho_bruto_xml = f"resultados/{nome_base}_BRUTO.musicxml"
        
        if os.path.exists(caminho_padrao_xml):
            shutil.move(caminho_padrao_xml, caminho_bruto_xml)
            print(f"\n📄 XML BRUTO GERADO: {caminho_bruto_xml}")
            print("⚠️  Abra este arquivo agora no MuseScore/Finale para identificar os erros.")
        else:
            print("⚠️ Não foi possível localizar o XML gerado automaticamente para renomear.")

    except Exception as e:
        print(f"❌ Erro ao gerar XML Bruto (mas você ainda pode editar os dados): {e}")

    # ---------------------------------------------------------
    # PASSO B: EDITAR OS DADOS (Interação Humana)
    # ---------------------------------------------------------
    input("\nPresione [ENTER] quando estiver pronto para corrigir os dados aqui no terminal...")
    
    # Chama o editor passando a lista original (yolo_linhas)
    yolo_linhas = modificar_dados_yolo(yolo_linhas)
    
    # ---------------------------------------------------------
    # PASSO C: GERAR XML FINAL (CORRIGIDO)
    # ---------------------------------------------------------
    print("\n--> Gerando XML Final Corrigido...")
    
    # Gera novamente, agora com a lista editada
    score = ocr.gerar_score_de_dados(yolo_linhas, linhas_pauta, linhas_verticais, saida_temp)
    
    # Renomeia para o nome final limpo
    #caminho_final_xml = f"resultados/{nome_base}.musicxml"
    #if os.path.exists(caminho_padrao_xml):
    #    shutil.move(caminho_padrao_xml, caminho_final_xml)
    
    # Se quiser gerar MIDI também, faça a renomeação similar aqui
    
   # print(f"\n✅ SUCESSO! Arquivo Final: {caminho_final_xml}\n")
    print("\n--- Ajuste de Armadura e Compasso ---")
    mudar = input("Deseja alterar a armadura ou fórmula de compasso? (s/n): ").lower()
    if mudar == 's':
        try:
                # Pergunta a fórmula (ex: 3/4)
                ts_input = input("Nova Fórmula de Compasso (ex: 3/4) ou Enter para manter: ").strip()
                novo_ts = ts_input if ts_input else None
                
                # Pergunta a armadura (ex: -1 para 1 bemol)
                arm_input = input("Alteração de Armadura (ex: 1 para #, -2 para bb) ou Enter: ").strip()
                nova_arm = int(arm_input) if arm_input else None
                
                # CHAMA A FUNÇÃO
                ocr.alterar_metadados_manual(score, compasso_num=1, novo_ts=novo_ts, nova_armadura_alteracoes=nova_arm)
                print("✅ Metadados atualizados no objeto Score.")
                
        except ValueError:
            print("❌ Erro: Armadura deve ser um número inteiro.")

        # -------------------------------------
        
        # Salva o arquivo final atualizado
    nome_saida = f"resultados/{img_path.stem}.musicxml"
    score.write('musicxml', fp=nome_saida)

# --- EXECUÇÃO ---
if __name__ == "__main__":
    if os.path.exists("tratamento"):
        shutil.rmtree("tratamento")
        
    print("--- Fluxo: Visualizar Erro -> Corrigir -> Finalizar ---")
    path = input("Caminho da imagem: ").strip().replace('"', '')
    
    if os.path.exists(path):
        processar_fluxo_com_edicao(path)
    else:
        print("Arquivo não encontrado.")