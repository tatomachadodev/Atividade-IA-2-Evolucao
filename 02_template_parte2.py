def avaliar(agente):
    """
    TODO - AGENTE + AMBIENTE + AÇÃO + RECOMPENSA + FITNESS

    1. Criar/reiniciar o ambiente.
    2. Receber o estado inicial.
    3. Enquanto o episódio não terminar:
        a) pedir uma ação ao agente;
        b) executar a ação no ambiente;
        c) receber novo estado e reward;
        d) acumular a recompensa.
    4. Fechar o ambiente.
    5. Retornar o fitness do indivíduo.
    """
    pass


def criar_populacao():
    """
    TODO - INDIVÍDUO + POPULAÇÃO

    Criar TAMANHO_POPULACAO agentes independentes.
    Todos devem começar com parâmetros aleatórios próprios.
    """
    pass


def selecionar(populacao):
    """
    TODO - SELEÇÃO

    Escolher os indivíduos que poderão gerar descendentes.
    O grupo deverá definir e explicar seu critério.
    """
    pass


def mutar(agente):
    """
    TODO - MUTAÇÃO + EXPLORAÇÃO

    Alterar aleatoriamente parte de W1, b1, W2 e b2.
    Utilizar obrigatoriamente:
         TAXA_MUTACAO
         INTENSIDADE_MUTACAO
    """
    pass


def nova_geracao(populacao):
    """
    TODO - ELITISMO + REPRODUÇÃO + MUTAÇÃO

    1. Selecionar os pais.
    2. Copiar NUM_ELITE melhores sem alteração.
    3. Criar descendentes a partir dos selecionados.
    4. Aplicar mutação nos descendentes.
    5. Retornar exatamente TAMANHO_POPULACAO agentes.
    """
    pass
