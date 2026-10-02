populacao = criar_populacao()
historico = []

for geracao in range(NUM_GERACOES):

    # Avaliar TODOS os indivíduos da geração.
    for agente in populacao:
        agente.fitness = avaliar(agente)

    valores = [agente.fitness for agente in populacao]

    melhor = max(valores)
    media = float(np.mean(valores))
    pior = min(valores)

    historico.append({
        "geracao": geracao,
        "melhor": melhor,
        "media": media,
        "pior": pior,
    })

    print(
        f"Geração {geracao:03d} | "
        f"Melhor: {melhor:.2f} | "
        f"Média: {media:.2f} | "
        f"Pior: {pior:.2f}"
    )

    # Produzir a próxima geração.
    populacao = nova_geracao(populacao)

# TODO:
# salvar o histórico em CSV
# gerar o gráfico de evolução
# executar/renderizar o melhor agente encontrado
