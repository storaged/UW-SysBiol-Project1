# main.py
"""
Główny skrypt symulacji Geometrycznego Modelu Fishera.

Uruchomienie:
    python main.py

Aby zmienić parametry symulacji, edytuj plik config.py.

Aby użyć innej strategii selekcji / reprodukcji / środowiska, zmień
obiekty przekazywane do run_simulation() w funkcji main() poniżej.
Dostępne klasy bazowe do rozszerzeń: strategies.py
"""

import os
import numpy as np

import config
from environment import LinearShiftEnvironment
from population import Population
from mutation import IsotropicMutation
from selection import TwoStageSelection
from reproduction import AsexualReproduction
from visualization import plot_population, plot_frame, plot_stats
from stats import SimulationStats


# ---------------------------------------------------------------------------
# Główna pętla symulacji
# ---------------------------------------------------------------------------

def run_simulation(
    population: Population,
    environment,
    selection_strategy,
    reproduction_strategy,
    mutation_strategy,
    max_generations: int = config.max_generations,
    frames_dir: str = None,
    verbose: bool = True,
    target_size: int = None,
    sigma: float = None,
) -> SimulationStats:
    """
    Uruchamia pętlę ewolucyjną i zwraca zebrane statystyki.

    Pętla ewolucyjna (4 kroki zgodnie z treścią zadania):
        1. Mutacja
        2. Selekcja
        3. Reprodukcja
        4. Zmiana środowiska

    :param population:            obiekt Population
    :param environment:           obiekt implementujący EnvironmentDynamics
    :param selection_strategy:    obiekt implementujący SelectionStrategy
    :param reproduction_strategy: obiekt implementujący ReproductionStrategy
    :param mutation_strategy:     obiekt implementujący MutationStrategy
    :param max_generations:       liczba pokoleń do zasymulowania
    :param frames_dir:            katalog do zapisu klatek PNG (None = brak)
    :param verbose:               czy drukować postęp co 10 pokoleń
    :param target_size:           docelowy rozmiar populacji (nadpisuje config.N)
    :param sigma:                 parametr selekcji (nadpisuje config.sigma)
    :return:                      obiekt SimulationStats z wynikami
    """
    if target_size is None:
        target_size = config.N
    if sigma is None:
        sigma = config.sigma

    stats = SimulationStats()

    if frames_dir is not None:
        os.makedirs(frames_dir, exist_ok=True)

    for generation in range(max_generations):
        alpha = environment.get_optimal_phenotype()

        # Krok 1: Mutacja
        mutation_strategy.mutate(population)

        # Krok 2: Selekcja
        survivors = selection_strategy.select(population.get_individuals(), alpha)
        if not survivors:
            if verbose:
                print(f"Populacja wymarła w pokoleniu {generation}.")
            stats.mark_extinct(generation)
            break

        # Krok 3: Reprodukcja
        new_individuals = reproduction_strategy.reproduce(survivors, target_size)
        population.set_individuals(new_individuals)

        # Zbieranie statystyk i zapis klatki (nowa populacja vs aktualne optimum)
        stats.record(generation, population, alpha, sigma,
                     reproduction_strategy=reproduction_strategy)

        if frames_dir is not None:
            frame_path = os.path.join(frames_dir, f"frame_{generation:03d}.png")
            plot_frame(population, alpha, generation, stats,
                       save_path=frame_path, show_plot=False,
                       max_generations=max_generations,
                       sigma=sigma)

        # Krok 4: Zmiana środowiska
        environment.update()

        if verbose and generation % 10 == 0:
            r = stats.records[-1]
            print(f"  Pokolenie {generation:4d} | "
                  f"śr. fitness: {r.mean_fitness:.3f} | "
                  f"dist. od optimum: {r.distance_from_optimum:.3f} | "
                  f"var. fenotyp.: {r.phenotype_variance:.3f}")

    return stats


# ---------------------------------------------------------------------------
# Narzędzie do tworzenia GIF
# ---------------------------------------------------------------------------

def create_gif_from_frames(frames_dir: str, gif_filename: str,
                            duration: float = 0.2) -> None:
    """
    Łączy wszystkie pliki PNG z katalogu frames_dir w animację GIF.
    Wymaga: pip install imageio
    """
    import imageio
    filenames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not filenames:
        print("Brak klatek do złożenia w GIF.")
        return
    with imageio.get_writer(gif_filename, mode='I', duration=duration) as writer:
        for fname in filenames:
            writer.append_data(imageio.imread(os.path.join(frames_dir, fname)))


# ---------------------------------------------------------------------------
# Punkt wejścia
# ---------------------------------------------------------------------------

def main():
    # --- Ziarno losowości (config.seed = None → inna symulacja za każdym razem) ---
    if config.seed is not None:
        np.random.seed(config.seed)

    # --- Inicjalizacja komponentów ---
    env = LinearShiftEnvironment(
        alpha_init=config.alpha0,
        c=config.c,
        delta=config.delta,
    )
    pop = Population(
        size=config.N,
        n_dim=config.n,
        init_scale=config.init_scale,
        alpha_init=config.alpha0,   # populacja startuje blisko alpha0, nie wokół zera
    )
    selection = TwoStageSelection(
        sigma=config.sigma,
        threshold=config.threshold,
        N=config.N,
    )
    reproduction = AsexualReproduction()
    mutation = IsotropicMutation(
        mu=config.mu,
        mu_c=config.mu_c,
        xi=config.xi,
    )

    # --- Uruchomienie symulacji ---
    print("Rozpoczynam symulację...\n")
    frames_dir = "frames"
    stats = run_simulation(
        population=pop,
        environment=env,
        selection_strategy=selection,
        reproduction_strategy=reproduction,
        mutation_strategy=mutation,
        frames_dir=frames_dir,
        verbose=True,
    )

    print(f"\n{stats.summary()}")

    # --- GIF ---
    print("\nTworzenie GIF-a...")
    create_gif_from_frames(frames_dir, "simulation.gif")
    print("GIF zapisany jako simulation.gif")

    # --- Wykres statystyk ---
    plot_stats(stats, save_path="simulation_stats.png", show_plot=False)
    print("Wykres statystyk zapisany jako simulation_stats.png")


if __name__ == "__main__":
    main()
