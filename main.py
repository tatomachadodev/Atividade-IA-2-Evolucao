import copy
import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)

# ---------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------

ENV_ID = "ALE/Freeway-v5"

TAMANHO_POPULACAO = 20
NUM_GERACOES = 30
NUM_ENTRADAS = 128
NUM_OCULTOS = 16
NUM_ACOES = 3

TAXA_MUTACAO = 0.10
INTENSIDADE_MUTACAO = 0.20
NUM_ELITE = 2

def criar_ambiente(render_mode=None):
    return gym.make(
    ENV_ID,
    obs_type="ram",
    render_mode=render_mode
    )
    
class Agente:
    def __init__(self):
    # Cada indivíduo começa com parâmetros diferentes.
        self.W1 = np.random.randn(NUM_ENTRADAS, NUM_OCULTOS)
        self.b1 = np.random.randn(NUM_OCULTOS)
        self.W2 = np.random.randn(NUM_OCULTOS, NUM_ACOES)
        self.b2 = np.random.randn(NUM_ACOES)
        self.fitness = 0.0
    
def agir(self, estado):
    # RAM: valores inteiros entre 0 e 255.
    estado = estado.astype(np.float32) / 255.0
    oculto = np.tanh(estado @ self.W1 + self.b1)
    saida = oculto @ self.W2 + self.b2
    # A maior saída define a ação escolhida.
    return int(np.argmax(saida))

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

    env = criar_ambiente()
    estado = env.reset()
    while True:
        acao = agente.agir(estado)
        estado, recompensa, terminado, truncado, info = env.step(acao)
        agente.fitness += recompensa
        if terminado or truncado:
            break

    env.close()
    return agente.fitness

def criar_populacao():
    return [Agente() for _ in range(TAMANHO_POPULACAO)]

