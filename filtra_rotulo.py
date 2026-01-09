import os

# Mapeamento: ID original → novo ID
id_map = {
    22: 0,  # barlineHeavy
    23: 1,  # barlineSingle
    81: 2,  # legerLine
    105: 3  # staffLine
}

# Caminho da pasta de entrada e de saída
pasta_entrada = "dataset/test/labels"
pasta_saida = "labels"

# Cria a pasta de saída se não existir
os.makedirs(pasta_saida, exist_ok=True)

# Percorre todos os arquivos .txt na pasta de entrada
for root, _, files in os.walk(pasta_entrada):
    for nome_arquivo in files:
        if nome_arquivo.endswith(".txt"):
            caminho_arquivo = os.path.join(root, nome_arquivo)

            # Caminho correspondente na pasta de saída
            subpasta = os.path.relpath(root, pasta_entrada)
            destino_pasta = os.path.join(pasta_saida, subpasta)
            os.makedirs(destino_pasta, exist_ok=True)
            caminho_saida = os.path.join(destino_pasta, nome_arquivo)

            # Filtra e renumera os IDs
            with open(caminho_arquivo, "r") as infile, open(caminho_saida, "w") as outfile:
                for linha in infile:
                    partes = linha.strip().split()
                    if len(partes) >= 2:
                        try:
                            class_id = int(partes[0])
                            if class_id in id_map:
                                partes[0] = str(id_map[class_id])  # renumera
                                nova_linha = " ".join(partes)
                                outfile.write(nova_linha + "\n")
                        except ValueError:
                            continue

print("✅ Todos os arquivos foram processados, filtrados e renumerados com sucesso.")
