import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import pygame

# ---------------------------
# Configurações gerais
# ---------------------------
WIDTH, HEIGHT = 1200, 700
GROUND_Y = 520
FPS = 60

# Cores
WHITE = (245, 245, 245)
BLACK = (30, 30, 30)
GRAY = (120, 120, 120)
LIGHT_GRAY = (210, 210, 210)
GREEN = (40, 180, 99)
RED = (220, 70, 70)
BLUE = (70, 120, 220)
ORANGE = (240, 150, 60)
PURPLE = (160, 90, 210)


@dataclass
class Obstacle:
    x: float
    w: int
    h: int

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), GROUND_Y - self.h, self.w, self.h)


class NeuralNetwork:
    """Rede neural mínima: 2 entradas -> 4 neurônios ocultos -> 1 saída."""

    def __init__(self):
        self.w1 = [[random.uniform(-1.5, 1.5) for _ in range(2)] for _ in range(4)]
        self.b1 = [random.uniform(-1.0, 1.0) for _ in range(4)]
        self.w2 = [random.uniform(-1.5, 1.5) for _ in range(4)]
        self.b2 = random.uniform(-1.0, 1.0)

    def clone(self) -> "NeuralNetwork":
        nn = NeuralNetwork()
        nn.w1 = [row[:] for row in self.w1]
        nn.b1 = self.b1[:]
        nn.w2 = self.w2[:]
        nn.b2 = self.b2
        return nn

    @staticmethod
    def _act(x: float) -> float:
        return math.tanh(x)

    def forward(self, distance_norm: float, height_norm: float) -> float:
        x = [distance_norm, height_norm]
        hidden = []
        for i in range(4):
            z = self.b1[i] + self.w1[i][0] * x[0] + self.w1[i][1] * x[1]
            hidden.append(self._act(z))
        out = self.b2
        for i in range(4):
            out += self.w2[i] * hidden[i]
        # Sigmoid para probabilidade de pulo
        return 1.0 / (1.0 + math.exp(-out))

    def mutate(self, rate: float, strength: float):
        for i in range(4):
            for j in range(2):
                if random.random() < rate:
                    self.w1[i][j] += random.uniform(-strength, strength)
            if random.random() < rate:
                self.b1[i] += random.uniform(-strength, strength)

        for i in range(4):
            if random.random() < rate:
                self.w2[i] += random.uniform(-strength, strength)
        if random.random() < rate:
            self.b2 += random.uniform(-strength, strength)


class Dino:
    def __init__(self, color: Tuple[int, int, int], x: int = 200):
        self.x = x
        self.y = GROUND_Y - 50
        self.w = 40
        self.h = 50
        self.vy = 0.0
        self.alive = True
        self.fitness = 0.0
        self.color = color
        self.nn = NeuralNetwork()

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)

    def reset(self):
        self.y = GROUND_Y - self.h
        self.vy = 0.0
        self.alive = True
        self.fitness = 0.0

    def jump(self):
        if self.y >= GROUND_Y - self.h - 1:
            self.vy = -12.5

    def update(self, dt: float, gravity: float):
        if not self.alive:
            return
        self.vy += gravity * dt
        self.y += self.vy * dt
        if self.y > GROUND_Y - self.h:
            self.y = GROUND_Y - self.h
            self.vy = 0.0
        self.fitness += dt


class UIButton:
    def __init__(self, x, y, w, h, text, callback, color=LIGHT_GRAY):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.color = color

    def draw(self, screen, font):
        pygame.draw.rect(screen, self.color, self.rect, border_radius=8)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=8)
        label = font.render(self.text, True, BLACK)
        screen.blit(label, label.get_rect(center=self.rect.center))

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()


class UISlider:
    def __init__(self, x, y, w, label, minv, maxv, value, fmt="{:.2f}"):
        self.x, self.y, self.w = x, y, w
        self.label = label
        self.minv = minv
        self.maxv = maxv
        self.value = value
        self.fmt = fmt
        self.dragging = False

    @property
    def knob_x(self):
        t = (self.value - self.minv) / (self.maxv - self.minv)
        return self.x + int(t * self.w)

    def draw(self, screen, font):
        pygame.draw.line(screen, BLACK, (self.x, self.y), (self.x + self.w, self.y), 3)
        pygame.draw.circle(screen, BLUE if self.dragging else ORANGE, (self.knob_x, self.y), 10)
        txt = f"{self.label}: {self.fmt.format(self.value)}"
        screen.blit(font.render(txt, True, BLACK), (self.x, self.y - 30))

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if abs(event.pos[0] - self.knob_x) < 14 and abs(event.pos[1] - self.y) < 14:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            x = max(self.x, min(self.x + self.w, event.pos[0]))
            t = (x - self.x) / self.w
            self.value = self.minv + t * (self.maxv - self.minv)


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Rede Neural Evolutiva - Dino (2 indivíduos)")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.small = pygame.font.SysFont("consolas", 16)

        self.gravity = 38.0
        self.scroll_speed = 360.0
        self.spawn_timer = 0.0
        self.spawn_interval = 1.3
        self.generation = 1
        self.best_fitness_ever = 0.0

        self.dinos: List[Dino] = [Dino(GREEN), Dino(PURPLE, x=260)]
        self.obstacles: List[Obstacle] = []

        self.selected_dino = 0

        # Controles de evolução / simulação
        self.mutation_rate = UISlider(780, 80, 360, "Mutação", 0.0, 1.0, 0.2)
        self.mutation_strength = UISlider(780, 150, 360, "Força mutação", 0.01, 2.0, 0.25)
        self.jump_threshold = UISlider(780, 220, 360, "Limiar pulo", 0.0, 1.0, 0.58)
        self.speed_slider = UISlider(780, 290, 360, "Velocidade", 120.0, 720.0, self.scroll_speed, fmt="{:.0f}")

        self.sliders = [
            self.mutation_rate,
            self.mutation_strength,
            self.jump_threshold,
            self.speed_slider,
        ]

        self.buttons = [
            UIButton(780, 350, 170, 42, "Nova Geração", self.force_new_generation),
            UIButton(970, 350, 170, 42, "Resetar Partida", self.reset_world),
            UIButton(780, 405, 170, 42, "Selecionar Dino", self.switch_dino),
            UIButton(970, 405, 170, 42, "Mutar Selecionado", self.mutate_selected),
        ]

        # Controles de pesos da rede
        self.param_buttons = self._build_param_buttons()

    def _build_param_buttons(self):
        buttons = []
        x_base = 780
        y_base = 500
        step = 28

        def make_cb(attr_path: Tuple[str, int, int], delta: float):
            def cb():
                nn = self.dinos[self.selected_dino].nn
                kind = attr_path[0]
                if kind == "w1":
                    i, j = attr_path[1], attr_path[2]
                    nn.w1[i][j] += delta
                elif kind == "b1":
                    i = attr_path[1]
                    nn.b1[i] += delta
                elif kind == "w2":
                    i = attr_path[1]
                    nn.w2[i] += delta
                elif kind == "b2":
                    nn.b2 += delta
            return cb

        # w1 (4x2)
        row = 0
        for i in range(4):
            for j in range(2):
                y = y_base + row * step
                buttons.append(UIButton(x_base + 250, y, 28, 24, "+", make_cb(("w1", i, j), 0.1)))
                buttons.append(UIButton(x_base + 282, y, 28, 24, "-", make_cb(("w1", i, j), -0.1)))
                row += 1

        # b1 (4)
        for i in range(4):
            y = y_base + row * step
            buttons.append(UIButton(x_base + 250, y, 28, 24, "+", make_cb(("b1", i, 0), 0.1)))
            buttons.append(UIButton(x_base + 282, y, 28, 24, "-", make_cb(("b1", i, 0), -0.1)))
            row += 1

        # w2 (4)
        for i in range(4):
            y = y_base + row * step
            buttons.append(UIButton(x_base + 250, y, 28, 24, "+", make_cb(("w2", i, 0), 0.1)))
            buttons.append(UIButton(x_base + 282, y, 28, 24, "-", make_cb(("w2", i, 0), -0.1)))
            row += 1

        # b2
        y = y_base + row * step
        buttons.append(UIButton(x_base + 250, y, 28, 24, "+", make_cb(("b2", 0, 0), 0.1)))
        buttons.append(UIButton(x_base + 282, y, 28, 24, "-", make_cb(("b2", 0, 0), -0.1)))

        return buttons

    def switch_dino(self):
        self.selected_dino = (self.selected_dino + 1) % 2

    def mutate_selected(self):
        self.dinos[self.selected_dino].nn.mutate(self.mutation_rate.value, self.mutation_strength.value)

    def reset_world(self):
        self.obstacles.clear()
        self.spawn_timer = 0
        for d in self.dinos:
            d.reset()

    def force_new_generation(self):
        self.end_generation()

    def spawn_obstacle(self):
        h = random.choice([40, 52, 64, 78])
        w = random.choice([22, 28, 34, 40])
        self.obstacles.append(Obstacle(WIDTH + random.randint(0, 80), w, h))

    def closest_obstacle_info(self, dino: Dino) -> Tuple[float, float]:
        candidates = [o for o in self.obstacles if o.x + o.w >= dino.x]
        if not candidates:
            return 1.0, 0.0
        obs = min(candidates, key=lambda o: o.x)

        # Entradas solicitadas: distância à direita e altura do obstáculo
        dist = max(0.0, obs.x - (dino.x + dino.w))
        distance_norm = min(1.0, dist / 600.0)
        height_norm = min(1.0, obs.h / 120.0)
        return distance_norm, height_norm

    def update(self, dt):
        self.scroll_speed = self.speed_slider.value

        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            self.spawn_obstacle()
            self.spawn_interval = random.uniform(1.0, 1.7)

        for o in self.obstacles:
            o.x -= self.scroll_speed * dt
        self.obstacles = [o for o in self.obstacles if o.x + o.w > -10]

        for d in self.dinos:
            if not d.alive:
                continue
            dist_n, h_n = self.closest_obstacle_info(d)
            jump_prob = d.nn.forward(dist_n, h_n)
            if jump_prob >= self.jump_threshold.value:
                d.jump()
            d.update(dt * 60.0, self.gravity)

            for o in self.obstacles:
                if d.rect.colliderect(o.rect):
                    d.alive = False
                    break

        if not any(d.alive for d in self.dinos):
            self.end_generation()

    def end_generation(self):
        ranked = sorted(self.dinos, key=lambda d: d.fitness, reverse=True)
        champion = ranked[0]
        self.best_fitness_ever = max(self.best_fitness_ever, champion.fitness)

        # Apenas 2 indivíduos:
        # - indivíduo 0: campeão preservado
        # - indivíduo 1: clone mutado do campeão
        base_nn = champion.nn.clone()
        child_nn = champion.nn.clone()
        child_nn.mutate(self.mutation_rate.value, self.mutation_strength.value)

        self.dinos[0].nn = base_nn
        self.dinos[1].nn = child_nn

        self.generation += 1
        self.reset_world()

    def draw_world(self):
        self.screen.fill(WHITE)

        # Céu e chão
        pygame.draw.rect(self.screen, (240, 248, 255), (0, 0, 760, HEIGHT))
        pygame.draw.line(self.screen, BLACK, (0, GROUND_Y), (760, GROUND_Y), 3)

        # Obstáculos
        for o in self.obstacles:
            pygame.draw.rect(self.screen, RED, o.rect)

        # Dinos
        for idx, d in enumerate(self.dinos):
            color = d.color if d.alive else GRAY
            pygame.draw.rect(self.screen, color, d.rect, border_radius=6)
            tag = self.small.render(f"D{idx}", True, BLACK)
            self.screen.blit(tag, (d.x + 8, d.y - 20))

        # HUD esquerdo
        lines = [
            f"Geracao: {self.generation}",
            f"Melhor fitness historico: {self.best_fitness_ever:.1f}",
            f"D0 fitness: {self.dinos[0].fitness:.1f} | vivo: {self.dinos[0].alive}",
            f"D1 fitness: {self.dinos[1].fitness:.1f} | vivo: {self.dinos[1].alive}",
            "Entradas da rede: distancia e altura do obstaculo mais proximo a direita",
        ]
        y = 15
        for line in lines:
            self.screen.blit(self.font.render(line, True, BLACK), (18, y))
            y += 28

        self.draw_right_panel()

    def draw_right_panel(self):
        pygame.draw.rect(self.screen, (250, 250, 250), (760, 0, 440, HEIGHT))
        pygame.draw.line(self.screen, BLACK, (760, 0), (760, HEIGHT), 2)

        title = self.font.render("Painel de Evolucao / Rede", True, BLACK)
        self.screen.blit(title, (780, 20))

        for s in self.sliders:
            s.draw(self.screen, self.small)
        for b in self.buttons:
            b.draw(self.screen, self.small)

        # Desenhar parâmetros da NN selecionada
        d = self.dinos[self.selected_dino]
        nn = d.nn
        self.screen.blit(self.small.render(f"Editando Dino {self.selected_dino}", True, BLUE), (780, 470))

        labels = []
        for i in range(4):
            for j in range(2):
                labels.append(f"w1[{i}][{j}] = {nn.w1[i][j]: .2f}")
        for i in range(4):
            labels.append(f"b1[{i}] = {nn.b1[i]: .2f}")
        for i in range(4):
            labels.append(f"w2[{i}] = {nn.w2[i]: .2f}")
        labels.append(f"b2 = {nn.b2: .2f}")

        y = 500
        for line in labels:
            self.screen.blit(self.small.render(line, True, BLACK), (780, y + 4))
            y += 28

        for b in self.param_buttons:
            b.draw(self.screen, self.small)

        hints = [
            "Tecla ESPACO: pulo manual do Dino selecionado",
            "Tudo (mutacao, limiar, velocidade e pesos) pode ser alterado na UI",
        ]
        y = HEIGHT - 54
        for h in hints:
            self.screen.blit(self.small.render(h, True, BLACK), (18, y))
            y += 20

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.dinos[self.selected_dino].jump()
                    if event.key == pygame.K_TAB:
                        self.switch_dino()

                for s in self.sliders:
                    s.handle(event)
                for b in self.buttons:
                    b.handle(event)
                for b in self.param_buttons:
                    b.handle(event)

            self.update(dt)
            self.draw_world()
            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    Game().run()
