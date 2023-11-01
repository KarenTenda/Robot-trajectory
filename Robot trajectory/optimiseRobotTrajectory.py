import numpy as np
import matplotlib.pyplot as plt

LINK_LENGTHS = [
    1,
    1,
    1,
]
TARGET_POINT = np.array([1.5, 1.5, 0.5])
NUM_POSITIONS = 3
TARGET_POINTS = [
    np.array([1.5, 1.5, 0.5]),
    np.array([0.5, 1.8, 0.2]),
    np.array([1.8, 0.5, 0.7]),
]
ANGLE_RANGE = (
    -np.pi,
    np.pi,
)  # (0, 2 * np.pi) depends if you want it in radians or angles
Z_RANGE = (-np.pi, np.pi)

NUM_GENERATIONS = 1000
MUTATION_RATE = 0.3
MUTATION_AMOUNT = 0.2
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.9
BEST_SOLUTIONS = []
POPULATION_SIZE = 100
GENERATIONS = []
FITNESSES = []

ADAPTIVE_INCREMENT = 0.5
MIN_MUTATION_RATE = 0.1
MAX_MUTATION_RATE = 0.95
FITNESS_THRESHOLD = 0.0001
ELITISM_COUNT = 20


# --------------------------------------------------------------------------------------------
#                                  Initialise Population
# --------------------------------------------------------------------------------------------
def initialize_population_multi_positions(pop_size, angle_range, num_positions):
    return np.random.uniform(
        angle_range[0], angle_range[1], (pop_size, num_positions, 2)
    )


# --------------------------------------------------------------------------------------------
#                                  Fitness Function
# --------------------------------------------------------------------------------------------


def fitness_single_target(angles, target):
    """Evaluate the fitness of an individual for a single target point."""
    theta1, theta2 = angles[0]
    
    x = LINK_LENGTHS[0] * np.cos(theta1) + LINK_LENGTHS[1] * np.cos(theta1 + theta2)
    y = LINK_LENGTHS[0] * np.sin(theta1) + LINK_LENGTHS[1] * np.sin(theta1 + theta2)
    z = theta1 + theta2
    end_effector_position = np.array([x, y, z])

    distance = np.linalg.norm(end_effector_position - target)
    fitness = 1 / (1 + distance)
    return fitness


def evaluate_population_single_target(population, target):
    """Evaluate the fitness of the entire population for a specific target."""
    return np.array([fitness_single_target(ind, target) for ind in population])


def fitness_multi_targets(angles):
    """Evaluate the fitness of an individual for multiple target points."""
    total_distance = 0
    for pos in range(NUM_POSITIONS):
        x = LINK_LENGTHS[0] * np.cos(angles[pos][0]) + LINK_LENGTHS[1] * np.cos(
            angles[pos][0] + angles[pos][1]
        )
        y = LINK_LENGTHS[0] * np.sin(angles[pos][0]) + LINK_LENGTHS[1] * np.sin(
            angles[pos][0] + angles[pos][1]
        )
        z = angles[pos][0] + angles[pos][1]
        end_effector_position = np.array([x, y, z])

        distance = np.linalg.norm(end_effector_position - TARGET_POINTS[pos])
        total_distance += distance

    fitness = 1 / (1 + total_distance)
    return fitness


def evaluate_population_multi_targets(population):
    return np.array([fitness_multi_targets(ind) for ind in population])


# --------------------------------------------------------------------------------------------
#                                  Selection
# --------------------------------------------------------------------------------------------


def tournament_selection(population, fitness_values):
    """Select two parents based on their fitnesses using tournament selection."""

    def select_one():
        selected_competitors = np.random.choice(
            len(population), size=TOURNAMENT_SIZE, replace=False
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
    normalized_fitness = fitness_values / np.sum(fitness_values)
    parents_indices = np.random.choice(len(population), size=2, p=normalized_fitness)
    return population[parents_indices]


# --------------------------------------------------------------------------------------------
#                                  Crossover
# --------------------------------------------------------------------------------------------


def crossover(parents):
    """Crossover: One-point Crossover for three angles"""
    if np.random.random() < CROSSOVER_RATE:
        crossover_point = np.random.randint(
            1, 3
        )  # Crossover point can be between any of the three angles
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


# --------------------------------------------------------------------------------------------
#                                  Mutation
# --------------------------------------------------------------------------------------------


def mutation(individual):
    """Mutation: add noise to increase diversity for 3 angles"""
    for angle in range(len(individual)):
        if np.random.rand() < MUTATION_RATE:
            mutation_value = np.random.uniform(-MUTATION_AMOUNT, MUTATION_AMOUNT)
            individual[angle] += mutation_value
            # Ensure the angles remain within the specified range
            if angle < 2:
                individual[angle] = np.clip(
                    individual[angle], ANGLE_RANGE[0], ANGLE_RANGE[1]
                )
            else:
                individual[angle] = np.clip(
                    individual[angle], ANGLE_RANGE[0], ANGLE_RANGE[1]
                )
    return individual


# --------------------------------------------------------------------------------------------
#                                  Creating new generation
# --------------------------------------------------------------------------------------------

def create_new_generation(population, fitness_values, prev_avg_fitness):
    """
        Steps:
        1.Sort population based on fitness values
        2.Elitism: Directly pass the best individuals to the next generation
        3.Fill the rest of the generation by crossover and mutation
        4.Compute average fitness and adjust mutation rate
    """
    global MUTATION_RATE

    sorted_indices = np.argsort(fitness_values)[::-1]
    new_population = []

    for i in range(ELITISM_COUNT):
        new_population.append(population[sorted_indices[i]])
 
    for _ in range((POPULATION_SIZE - ELITISM_COUNT) // 2):
        parents = tournament_selection(population, fitness_values)
        offspring1, offspring2 = crossover(parents)
        new_population.append(mutation(offspring1))
        new_population.append(mutation(offspring2))
 
    avg_fitness = np.mean(fitness_values)
    if abs(avg_fitness - prev_avg_fitness) < FITNESS_THRESHOLD:
        MUTATION_RATE = min(MUTATION_RATE + ADAPTIVE_INCREMENT, MAX_MUTATION_RATE)
    else:
        MUTATION_RATE = max(MUTATION_RATE - ADAPTIVE_INCREMENT, MIN_MUTATION_RATE)

    return np.array(new_population), avg_fitness


# --------------------------------------------------------------------------------------------
#                                  Genetic Algorithm
# --------------------------------------------------------------------------------------------

def GeneticAlgorithm_single_target(target):
    """Run the genetic algorithm for a specific target point."""
    population = initialize_population_multi_positions(
        POPULATION_SIZE, ANGLE_RANGE, 1
    ) 
    prev_avg_fitness = 0
    best_fitness_per_generation = []

    for _ in range(NUM_GENERATIONS):
        fitness_values = evaluate_population_single_target(population, target)

        best_fitness_this_generation = np.max(fitness_values)
        best_fitness_per_generation.append(best_fitness_this_generation)

        best_idx = np.argmax(fitness_values)
        best_solution = population[best_idx]
        BEST_SOLUTIONS.append((best_solution, fitness_values[best_idx]))

        population, prev_avg_fitness = create_new_generation(
            population, fitness_values, prev_avg_fitness
        )

    return max(BEST_SOLUTIONS, key=lambda x: x[1]), best_fitness_per_generation


def GeneticAlgorithm_multi_targets():
    """Run the genetic algorithm for a multi target points."""
    population = initialize_population_multi_positions(
        POPULATION_SIZE, ANGLE_RANGE, NUM_POSITIONS
    )
    prev_avg_fitness = 0
    best_fitness_per_generation = []

    for generation in range(NUM_GENERATIONS):
        GENERATIONS.append(generation)
        fitness_values = evaluate_population_multi_targets(population)

        best_fitness_this_generation = np.max(fitness_values)
        best_fitness_per_generation.append(best_fitness_this_generation)

        best_idx = np.argmax(fitness_values)
        best_solution = population[best_idx]
        FITNESSES.append(fitness_values[best_idx] * 100)
        BEST_SOLUTIONS.append((best_solution, fitness_values[best_idx]))

        population, prev_avg_fitness = create_new_generation(
            population, fitness_values, prev_avg_fitness
        )

    return max(BEST_SOLUTIONS, key=lambda x: x[1]), best_fitness_per_generation


def fitness_multi_positions_individual(angles):
    """Evaluate the fitness of an individual for multiple positions and return individual fitness values."""
    return np.array(
        [
            fitness_single_target(angles[i], TARGET_POINTS[i])
            for i in range(NUM_POSITIONS)
        ]
    )

# --------------------------------------------------------------------------------------------
#                                  Results for specific target point
# --------------------------------------------------------------------------------------------

results = []
SELECTED_ANGLES = []

for target in TARGET_POINTS:
    print("Target :", target)
    BEST_SOLUTIONS = [] 
    best_angles, best_fitness_values_for_target = GeneticAlgorithm_single_target(target)
    SELECTED_ANGLES.append(best_angles[0])
    results.append((best_angles, best_fitness_values_for_target))

    angles_with_theta3 = []
    for position_angles in best_angles[0]:
        theta1, theta2 = position_angles
        theta3 = theta1 + theta2
        angles_with_theta3.append([theta1, theta2, theta3])

    print("Angles in radians : ", angles_with_theta3)

    individual_fitness_values = fitness_single_target(best_angles[0], target)
    print("Fitness : ", individual_fitness_values)

    angles_with_theta3_degrees = np.degrees(angles_with_theta3)
    angles_in_degrees = np.degrees(best_angles[0])
    print("Angles in degrees : ", angles_with_theta3_degrees)
    plt.plot(best_fitness_values_for_target, label=str(target))
    print("..................................................................")



plt.title("Fitness vs Generations for Each Target")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.legend()
plt.grid(True)
plt.show()


# --------------------------------------------------------------------------------------------
#                                  Results for multi target points
# --------------------------------------------------------------------------------------------
# BEST_SOLUTIONS = []

# best_solution_with_adaptations, best_fitness_values = GeneticAlgorithm_multi_targets()

# angles_with_theta3 = []
# for position_angles in best_solution_with_adaptations[0]:
#     theta1, theta2 = position_angles
#     theta3 = theta1 + theta2
#     angles_with_theta3.append([theta1, theta2, theta3])

# print("Angles in radians : ", angles_with_theta3)

# individual_fitness_values = fitness_multi_positions_individual(angles_with_theta3)
# print("Fitness : ", individual_fitness_values)

# angles_with_theta3_degrees = np.degrees(angles_with_theta3)
# angles_in_degrees = np.degrees(best_solution_with_adaptations[0])
# print("Angles in degrees : ", angles_with_theta3_degrees)

# plt.figure(figsize=(10, 6))
# plt.plot(best_fitness_values)
# plt.title("Fitness vs Generations")
# plt.xlabel("Generation")
# plt.ylabel("Fitness")
# plt.grid(True)
# plt.show()


# --------------------------------------------------------------------------------------------
#                                  3D Plots
# --------------------------------------------------------------------------------------------


def plot_robot_3d(positions, ax, color="b"):
    """Plot the robotic arm in 3D based on joint and end effector positions."""
    xs, ys, zs = zip(*positions)
    ax.plot(xs, ys, zs, "-o", color=color)


def compute_positions_3d(angles):
    """Compute the positions of the joints and end effector for given angles in 3D."""
    theta1,theta2 = angles
    x1 = LINK_LENGTHS[0] * np.cos(theta1)
    y1 = LINK_LENGTHS[0] * np.sin(theta1)
    z1 = 0  # Assuming no vertical movement for the first link

    x2 = x1 + LINK_LENGTHS[1] * np.cos(theta1 + theta2)
    y2 = y1 + LINK_LENGTHS[1] * np.sin(theta1 + theta2)
    # For the z-coordinate, we assume it's linearly related to the angles.
    z2 = theta1 + theta2

    return [(0, 0, 0), (x1, y1, z1), (x2, y2, z2)]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.set_title("Optimization of Robotic Arm Trajectory")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Plot the initial robotic arm position (assuming angles [0, 0])
initial_positions_3d = compute_positions_3d([0, 0])
plot_robot_3d(initial_positions_3d, ax, color="gray")

# Plot the target points
for target in TARGET_POINTS:
    ax.scatter(*target, color="r", s=100, label="Target Point")

# Plot the robotic arm in the best solution positions
colors = ["b", "g", "m"]
# best_solution_with_adaptations[0]
for j, selected_angles in enumerate(SELECTED_ANGLES):
    for i, angles in enumerate(selected_angles):
        best_positions_3d = compute_positions_3d(angles)
        plot_robot_3d(best_positions_3d, ax, color=colors[j][i])

# Show the plot with interactions enabled
plt.show()
