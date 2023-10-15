import numpy as np
import matplotlib.pyplot as plt

# Constants
LINK_LENGTHS = [1, 1]  # Assume each link of the robotic arm is of unit length
TARGET_POINT = np.array([1.5, 1.5])  # The target point in the 2D plane
POPULATION_SIZE = 150
ANGLE_RANGE = (0, 2 * np.pi)

NUM_GENERATIONS = 2000
MUTATION_RATE = 0.75
MUTATION_AMOUNT = 0.3
CROSSOVER_RATE = 0.9
BEST_SOLUTIONS = []
# tournament_size = 2


# Initialization: Create a population of random angles
def initialize_population(pop_size, angle_range):
    return np.random.uniform(angle_range[0], angle_range[1], (pop_size, 2))

def fitness(angles):
    """Evaluate the fitness of an individual"""
    # Calculate the position of the end effector for given angles
    x = LINK_LENGTHS[0] * np.cos(angles[0]) + LINK_LENGTHS[1] * np.cos(
        angles[0] + angles[1]
    )
    y = LINK_LENGTHS[0] * np.sin(angles[0]) + LINK_LENGTHS[1] * np.sin(
        angles[0] + angles[1]
    )
    end_effector_position = np.array([x, y])

    # Calculate the distance to the target
    distance = np.linalg.norm(end_effector_position - TARGET_POINT)

    # The fitness is the inverse of the distance (we want to minimize the distance)
    fitness = 1 / (1 + distance)  # Adding 1 to avoid division by zero
    return fitness


# Evaluate the entire population
def evaluate_population(population):
    return np.array([fitness(ind) for ind in population])

# Genetic Algorithm Components
def tournament_selection(population, fitness_values, tournament_size=2):
    """Select two parents based on their fitnesses using tournament selection."""

    def select_one():
        selected_competitors = np.random.choice(
            len(population), size=tournament_size, replace=False
        )
        competitor_fitnesses = np.array(
            [fitness_values[i] for i in selected_competitors]
        )
        winner_index = selected_competitors[np.argmax(competitor_fitnesses)]
        return population[winner_index]

    # Select two parents
    parent1 = select_one()
    parent2 = select_one()
    while np.array_equal(parent1, parent2):  # Ensure distinct parents
        parent2 = select_one()

    return np.array([parent1, parent2])

def roulette_wheel_selection(population, fitness_values):
    """Select two parents based on their fitnesses using Roulette Wheel selection."""
    # Normalize the fitness values
    normalized_fitness = fitness_values / np.sum(fitness_values)
    # Select two parents based on their normalized fitness values
    parents_indices = np.random.choice(len(population), size=2, p=normalized_fitness)
    return population[parents_indices]

def crossover(parents):
    """Crossover: One-point Crossover"""
    if np.random.random() < CROSSOVER_RATE:
        crossover_point = np.random.randint(
            1, 2
        )  # One-point crossover between the two genes (angles)
        offspring1 = np.concatenate(
            (parents[0, :crossover_point], parents[1, crossover_point:])
        )
        offspring2 = np.concatenate(
            (parents[1, :crossover_point], parents[0, crossover_point:])
        )
    else:
        # If no crossover, return the parents as the offspring
        offspring1, offspring2 = parents[0], parents[1]

    return offspring1, offspring2

def mutate(individual):
    """Mutation: add noise to increase diversity"""
    for i in range(len(individual)):
        if np.random.rand() < MUTATION_RATE:
            mutation_value = np.random.uniform(-MUTATION_AMOUNT, MUTATION_AMOUNT)
            individual[i] += mutation_value
            # Ensure the angles remain within the specified range
            individual[i] = np.clip(individual[i], ANGLE_RANGE[0], ANGLE_RANGE[1])
    return individual

def compute_positions(angles):
    """Compute the positions of the joints and end effector for given angles."""
    x1 = LINK_LENGTHS[0] * np.cos(angles[0])
    y1 = LINK_LENGTHS[0] * np.sin(angles[0])

    x2 = x1 + LINK_LENGTHS[1] * np.cos(angles[0] + angles[1])
    y2 = y1 + LINK_LENGTHS[1] * np.sin(angles[0] + angles[1])

    return [(0, 0), (x1, y1), (x2, y2)]

def plot_robot(positions, color="b"):
    """Plot the robotic arm based on joint and end effector positions."""
    xs, ys = zip(*positions)
    plt.plot(xs, ys, "-o", color=color)


def create_new_generation(population, fitness_values):
    new_population = []
    for _ in range(POPULATION_SIZE // 2): 
        parents = tournament_selection(population, fitness_values)
        offspring1, offspring2 = crossover(parents)
        new_population.append(mutate(offspring1))
        new_population.append(mutate(offspring2))
    return np.array(new_population)


# Main Genetic Algorithm Loop
def run_genetic_algorithm():
    # Initialize the population and evaluate it
    population = initialize_population(POPULATION_SIZE, ANGLE_RANGE)

    for generation in range(NUM_GENERATIONS):
        # Evaluate the current population
        fitness_values = evaluate_population(population)

        # Store the best solution of this generation
        best_idx = np.argmax(fitness_values)
        best_solution = population[best_idx]
        BEST_SOLUTIONS.append((best_solution, fitness_values[best_idx]))

        # Create a new generation
        population = create_new_generation(population, fitness_values)

    # Retrieve the best overall solution
    return max(BEST_SOLUTIONS, key=lambda x: x[1])


best_overall_solution = run_genetic_algorithm()
print(best_overall_solution)

# Plot the initial robotic arm position (assuming angles [0, 0])
initial_positions = compute_positions([0, 0])
plot_robot(initial_positions, color="gray")

# Plot the target point
plt.plot(TARGET_POINT[0], TARGET_POINT[1], "ro", label="Target Point")

# Plot the robotic arm in the best solution position
best_positions = compute_positions(best_overall_solution[0])
plot_robot(best_positions, color="b")

plt.legend()
plt.grid(True)
plt.title("Robotic Arm Trajectory Optimization")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.show()
