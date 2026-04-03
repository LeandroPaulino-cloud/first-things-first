# first-things-first

## Rede Neural Evolutiva para o Dino Offline

Este projeto contém um protótipo do jogo estilo Dino offline com **apenas 2 indivíduos**, cada um controlado por uma rede neural evolutiva.

### Requisitos

- Python 3.10+
- `pygame`

Instalação:

```bash
pip install pygame
```

### Execução

```bash
python dino_evolution_game.py
```

### O que foi implementado

- Interface gráfica em estilo de jogo (janela com cenário e painel lateral).
- Dois indivíduos (`D0` e `D1`) rodando ao mesmo tempo.
- Entrada da rede neural para cada indivíduo:
  - distância para o obstáculo mais próximo à direita;
  - altura do obstáculo mais próximo.
- Evolução automática por geração:
  - quando ambos morrem, o melhor é preservado;
  - o segundo indivíduo vira um clone mutado do melhor.
- Controles totalmente pela interface:
  - taxa e força de mutação;
  - limiar de pulo da rede;
  - velocidade do jogo;
  - mutação manual;
  - troca de dino selecionado;
  - edição direta dos pesos e vieses (`w1`, `b1`, `w2`, `b2`) pelos botões `+` e `-`.

### Interação

- `ESPACO`: pulo manual do dino selecionado.
- `TAB`: alterna o indivíduo selecionado para edição.
- Botões e sliders no painel direito para ajustes em tempo real.
