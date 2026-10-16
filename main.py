import argparse
import copy
import csv
import sys
from pathlib import Path

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

gym.register_envs(ale_py)

#-----------------
# CONFIG
#-----------------

ENV_ID = "ALE/Breakout-v5"

TAMANHO_POPULACAO = 20
NUM_GERACOES = 30
NUM_ENTRADAS = 128
NUM_OCULTOS = 32
NUM_ACOES = 4
LIMITE_MOVIMENTOS = 8000
EPISODIOS_POR_AGENTE = 3

TAXA_MUTACAO = 0.10
INTENSIDADE_MUTACAO = 0.20
NUM_ELITE = 3

RESULTADOS = Path("resultados")
EXPERIMENTOS = {"A": (0.01, 0.05), "B": (0.10, 0.20), "C": (0.40, 0.50)}
rng = np.random.default_rng(42)

class Agente:
  def __init__(self):
    self.W1 = np.random.randn(NUM_ENTRADAS, NUM_OCULTOS) * np.sqrt(2.0 / NUM_ENTRADAS)
    self.b1 = np.zeros(NUM_OCULTOS)
    self.W2 = np.random.randn(NUM_OCULTOS, NUM_ACOES) * np.sqrt(2.0 / NUM_OCULTOS)
    self.b2 = np.zeros(NUM_ACOES)
    self.fitness = 0.0

  def agir(self, estado):
    estado = estado.astype(np.float32) / 255.0 - 0.5

    oculto = np.tanh(estado @ self.W1 + self.b1)
    saida = oculto @ self.W2 + self.b2

    return int(np.argmax(saida))

def criar_ambiente(render_mode=None):
  return gym.make(ENV_ID, obs_type="ram", render_mode=render_mode)

def executar_episodio(
    ambiente, agente=None, seed=None, diagnosticar=False, acompanhar=False
):
  estado, informacoes = ambiente.reset(seed=seed)
  aleatorio = np.random.default_rng(seed) if agente is None else None
  recompensa_total = 0.0
  contagem_acoes = np.zeros(NUM_ACOES, dtype=int)
  trocas_acao = 0
  acao_anterior = None
  terminado = False
  passos = 0
  vidas = informacoes.get("lives")
  precisa_disparar = True

  while not terminado and passos < LIMITE_MOVIMENTOS:
    if precisa_disparar:
      acao = 1  
      precisa_disparar = False
    else:
      acao = agente.agir(estado) if agente else int(aleatorio.integers(NUM_ACOES))
    if diagnosticar:
      contagem_acoes[acao] += 1
      if acao_anterior is not None and acao != acao_anterior:
        trocas_acao += 1
      acao_anterior = acao
    estado, recompensa, terminated, truncated, informacoes = ambiente.step(acao)
    recompensa_total += recompensa
    terminado = terminated or truncated
    passos += 1
    novas_vidas = informacoes.get("lives", vidas)
    if vidas is not None and novas_vidas < vidas:
      precisa_disparar = True
    vidas = novas_vidas

  if not diagnosticar:
    return recompensa_total

  passos = int(contagem_acoes.sum())
  percentuais = 100 * contagem_acoes / max(1, passos)
  return recompensa_total, {
    "passos": passos,
    "noop_pct": float(percentuais[0]),
    "fire_pct": float(percentuais[1]),
    "right_pct": float(percentuais[2]),
    "left_pct": float(percentuais[3]),
    "troca_acao": trocas_acao,
    "acao_dominante_pct": float(max(percentuais))
  }

def avaliar(agente, ambiente, seed=42, diagnosticar=False, acompanhar=False):
  return executar_episodio(
    ambiente,
    agente=agente,
    seed=seed,
    diagnosticar=diagnosticar,
    acompanhar=acompanhar,
  )

def criar_populacao(tamanho=TAMANHO_POPULACAO):
  populacao = []
  for _ in range(tamanho):
    populacao.append(Agente())
  return populacao

def selecionar(populacao):
  ordenada = sorted(populacao, key=lambda a: a.fitness, reverse=True)
  num_pais = max(1, len(populacao) // 2)
  return ordenada[:num_pais]

def mutar(agente, taxa, intensidade):
  for nome in ("W1", "b1", "W2", "b2"):
    pesos = getattr(agente, nome)
    mascara = rng.random(pesos.shape) < taxa
    pesos += mascara * rng.normal(0, intensidade, pesos.shape)
  agente.fitness = 0.0
  return agente

def nova_geracao(populacao, num_elite, taxa, intensidade):
  pais = selecionar(populacao)
  elite = [copy.deepcopy(pai) for pai in pais[:num_elite]]
  filhos = []

  while len(elite) + len(filhos) < len(populacao):
    pai = pais[rng.integers(len(pais))]
    filho = copy.deepcopy(pai)
    filhos.append(mutar(filho, taxa, intensidade))

  return elite + filhos

def salvar_historico(historico, pasta):
  with (pasta / "fitness.csv").open("w", newline="", encoding="utf-8") as arquivo:
    escritor = csv.DictWriter(arquivo, fieldnames=["geracao", "melhor", "media", "pior"])
    escritor.writeheader()
    escritor.writerows(historico)

  geracoes = [linha["geracao"] for linha in historico]
  plt.figure(figsize=(8, 4))
  plt.plot(geracoes, [linha["melhor"] for linha in historico], label="Melhor")
  plt.plot(geracoes, [linha["media"] for linha in historico], label="Média")
  plt.xlabel("Geração")
  plt.ylabel("Fitness (recompensa acumulada)")
  plt.legend()
  plt.tight_layout()
  plt.savefig(pasta / "fitness.png", dpi=150)
  plt.close()


def treinar(tamanho, geracoes, elite, taxa, intensidade, pasta, seed, acompanhar=False):
    global rng
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    populacao = criar_populacao(tamanho)
    historico = []
    diagnosticos = []
    melhor_agente = None
    melhor_fitness = float("-inf")

    ambiente = criar_ambiente(render_mode="human" if acompanhar else None)
    try:
        for geracao in range(geracoes):
            print(f"Avaliando geração {geracao + 1}/{geracoes}...", flush=True)
            for indice, agente in enumerate(populacao, start=1):
                if acompanhar:
                    print(f"  Geração {geracao + 1}/{geracoes} | "
                          f"Indivíduo {indice}/{tamanho}", flush=True)
                seed_avaliacao = seed + geracao * 10_000 + indice
                notas = []
                metricas = None
                for episodio in range(EPISODIOS_POR_AGENTE):
                    resultado = avaliar(
                        agente,
                        ambiente,
                        seed=seed_avaliacao + episodio * 101,
                        diagnosticar=(episodio == 0),
                        acompanhar=acompanhar and episodio == 0,
                    )
                    if episodio == 0:
                        nota, metricas = resultado
                    else:
                        nota = resultado
                    notas.append(nota)
                agente.fitness = float(np.mean(notas))
                diagnosticos.append({
                    "geracao": geracao,
                    "individuo": indice,
                    "fitness": agente.fitness,
                    **metricas,
                })

            diagnosticos_geracao = diagnosticos[-tamanho:]
            repetitivos = sum(d["acao_dominante_pct"] >= 95 for d in diagnosticos_geracao)
            medias_acoes = {
                acao: float(np.mean([d[acao] for d in diagnosticos_geracao]))
                for acao in ("noop_pct", "fire_pct", "right_pct", "left_pct")
            }
            valores = [agente.fitness for agente in populacao]
            campeao = max(populacao, key=lambda agente: agente.fitness)
            if campeao.fitness > melhor_fitness:
                melhor_fitness = campeao.fitness
                melhor_agente = copy.deepcopy(campeao)

            linha = {
                "geracao": geracao,
                "melhor": float(max(valores)),
                "media": float(np.mean(valores)),
                "pior": float(min(valores)),
            }
            historico.append(linha)
            print(f"Geração {geracao:03d} | Melhor: {linha['melhor']:.2f} | "
                  f"Média: {linha['media']:.2f} | Pior: {linha['pior']:.2f}")
            print(f"  Ações médias: NOOP {medias_acoes['noop_pct']:.1f}% | "
                  f"FIRE {medias_acoes['fire_pct']:.1f}% | "
                  f"RIGHT {medias_acoes['right_pct']:.1f}% | "
                  f"LEFT {medias_acoes['left_pct']:.1f}% | "
                  f"Políticas repetitivas (>=95% uma ação): {repetitivos}/{tamanho}")

            if geracao < geracoes - 1:
                intensidade_efetiva = intensidade * (1.0 - 0.8 * geracao / max(1, geracoes - 1))
                populacao = nova_geracao(populacao, elite, taxa, intensidade_efetiva)
    finally:
        ambiente.close()

    pasta.mkdir(parents=True, exist_ok=True)
    salvar_historico(historico, pasta)
    with (pasta / "diagnostico.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(diagnosticos[0]))
        escritor.writeheader()
        escritor.writerows(diagnosticos)
    np.savez(pasta / "melhor_agente.npz", W1=melhor_agente.W1, b1=melhor_agente.b1,
             W2=melhor_agente.W2, b2=melhor_agente.b2)
    return melhor_agente, melhor_fitness

def comparar(agente, seed):
    ambiente = criar_ambiente()
    try:
        aleatorio = executar_episodio(ambiente, seed=seed + 50_000)
        evoluido = avaliar(agente, ambiente, seed=seed + 50_000)
    finally:
        ambiente.close()
    print(f"Comparação | Aleatório: {aleatorio:.2f} | Evoluído: {evoluido:.2f}")
    return aleatorio, evoluido

def visualizar(agente, seed):
    ambiente = criar_ambiente(render_mode="human")
    try:
        executar_episodio(ambiente, agente, seed)
    finally:
        ambiente.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Neuroevolução no Breakout")
    parser.add_argument("--populacao", type=int, default=TAMANHO_POPULACAO)
    parser.add_argument("--geracoes", type=int, default=NUM_GERACOES)
    parser.add_argument("--elite", type=int, default=NUM_ELITE)
    parser.add_argument("--experimento", choices=["A", "B", "C", "todos"], default="B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualizar", action="store_true",
                        help="Exibe o melhor agente ao terminar o treinamento.")
    parser.add_argument("--acompanhar", action="store_true",
                        help="Exibe todas as partidas enquanto os agentes são avaliados.")
    if argv is None and "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args(argv)

    if args.populacao < 2 or args.geracoes < 1 or not (1 <= args.elite <= args.populacao // 2):
        parser.error("Use população >= 2, gerações >= 1 e elite entre 1 e metade da população.")

    nomes = EXPERIMENTOS if args.experimento == "todos" else [args.experimento]
    for nome in nomes:
        taxa, intensidade = EXPERIMENTOS[nome]
        pasta = RESULTADOS / nome
        print(f"\nExperimento {nome} | Mutação: {taxa:.0%} | Intensidade: {intensidade}")
        agente, _ = treinar(args.populacao, args.geracoes, args.elite,
                           taxa, intensidade, pasta, args.seed, args.acompanhar)
        aleatorio, evoluido = comparar(agente, args.seed)
        with (pasta / "comparacao.csv").open("w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.writer(arquivo)
            escritor.writerow(["agente", "recompensa"])
            escritor.writerows([["aleatorio", aleatorio], ["evoluido", evoluido]])
        if args.visualizar:
            visualizar(agente, args.seed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.", file=sys.stderr)