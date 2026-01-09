def extrair_valores_y_distintos(arquivo_txt):
    ys_distintos = set()
    ys_res = set()

    with open(arquivo_txt, 'r') as arquivo:
        for linha in arquivo:
            partes = linha.strip().split(',')
            if len(partes) != 5:
                continue  # pula linhas mal formatadas

            tipo, x, y, w, h = partes
            try:
                y = int(y)
                h = int(h)
            except ValueError:
                continue  # pula valores inválidos

            if h <= 2:
                ys_distintos.add(y)

    ys_ordenados = sorted(ys_distintos)

    i = len(ys_ordenados) - 1

    while i > 0:

        if (ys_ordenados[i] - ys_ordenados[i-1] > 2):
            ys_res.add(ys_ordenados[i])

        i -= 1
    ys_res.add(ys_ordenados[i])

    return sorted(ys_res)


def pre_simbolo(pauta, simbolos):
    ys_distintos = set()
    ys_res = set()

    with open(simbolos, 'r') as arquivo:
        for linha in arquivo:
            partes = linha.strip().split(',')
            if len(partes) != 5:
                continue  # pula linhas mal formatadas

            arq, class_confidence, center_x, center_y, width, height = partes
            try:
                center_x = int(center_x)
                width = int(width)
                center_y = int(center_y)
                height = int(height)

                x_min = center_x - width / 2
                y_min = center_y - height / 2

            except ValueError:
                continue  # pula valores inválidos

           # if h <= 2:
           #     ys_distintos.add(y)


# Exemplo de uso:
arquivo = 'teste/deteccao/linhas_detectadas.txt'
simbolos = 'teste/deteccao/todas_deteccoes.txt'
valores_y = extrair_valores_y_distintos(arquivo)
print("Valores y distintos com h <= 2:", valores_y)
