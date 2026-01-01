import math
import random
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------
# GLOBAL CONSTANTS
#------------------------------------------
SHOP_X, SHOP_Y = 0, 0

# SA specific constants
INITIAL_TEMPERATURE = 1000
COOLING_RATE = 0.90  # Within the range 0.90-0.99
MIN_TEMPERATURE = 1
ITERATIONS_PER_TEMP = 100

# GA specific constants
POPULATION_SIZE = 75  # Within the range 50-100
MUTATION_RATE = 0.05  # Within the range 0.01-0.1
MAX_GENERATIONS = 500

#------------------------------------------
# COMMON UTILITY FUNCTIONS
#------------------------------------------
def euclidean_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_route_distance(route, shop_coords=(SHOP_X, SHOP_Y)):
    """Calculate the total distance of a route"""
    if not route:
        return 0

    total_distance = euclidean_distance(shop_coords[0], shop_coords[1],
                                        route[0]['x'], route[0]['y'])

    for i in range(len(route) - 1):
        total_distance += euclidean_distance(route[i]['x'], route[i]['y'],
                                             route[i + 1]['x'], route[i + 1]['y'])

    # Return to shop
    total_distance += euclidean_distance(route[-1]['x'], route[-1]['y'],
                                         shop_coords[0], shop_coords[1])

    return total_distance


def calculate_total_distance(solution):
    """Calculate the total distance for all vehicles"""
    total_distance = 0
    for vehicle in solution:
        route = vehicle["route"]
        total_distance += calculate_route_distance(route)
    return total_distance


def get_priority_penalty(solution):
    """Calculate penalty for not prioritizing high priority packages"""
    penalty = 0
    for vehicle in solution:
        position = 1
        for package in vehicle["route"]:
            # Lower priority number means higher priority
            # Multiply by position to penalize high priority packages delivered later
            penalty += package["priority"] * position
            position += 1
    return penalty / 100  # Scale down the penalty


def calculate_fitness(solution, packages):
    """Calculate fitness of a solution (lower is better)"""
    # Check for capacity violations - apply a severe penalty if any vehicle is overloaded
    for vehicle in solution:
        total_weight = sum(pkg["weight"] for pkg in vehicle["route"])
        if total_weight > vehicle["capacity"]:
            return 1e6  # Hard penalty for exceeding capacity
    # Create a set of package coordinates to check for uniqueness
    assigned_packages = set()

    for vehicle in solution:
        for i in range(len(vehicle["route"]) - 1):
            if vehicle["route"][i]["priority"] > vehicle["route"][i + 1]["priority"]:
                return 1e6  # Hard penalty if priorities are out of order

        for pkg in vehicle["route"]:
            # Use a tuple of package attributes as a unique identifier
            package_id = (pkg["x"], pkg["y"], pkg["weight"], pkg["priority"])
            assigned_packages.add(package_id)

    # Check if all packages are assigned
    total_packages = len(packages)
    if len(assigned_packages) != total_packages:
        return 1e6  # Penalize incomplete solution

    # Distance and priority penalty
    total_distance = calculate_total_distance(solution)
    priority_penalty = get_priority_penalty(solution)

    return total_distance + priority_penalty

def check_vehicle_capacity(vehicle, package=None):
    """Check if adding a package would exceed vehicle capacity"""
    current_weight = sum(pkg["weight"] for pkg in vehicle["route"])
    if package:
        return current_weight + package["weight"] <= vehicle["capacity"]
    return current_weight <= vehicle["capacity"]


def create_initial_solution(vehicles, packages):
    """Create a truly random initial solution"""
    solution = []

    # Create a copy of each vehicle with empty route
    for vehicle in vehicles:
        vehicle_copy = vehicle.copy()
        vehicle_copy["route"] = []
        solution.append(vehicle_copy)

    # Sort packages by priority before assignment (1 = highest priority)
    available_packages = sorted(packages, key=lambda p: p["priority"])

    # Try to assign each package to a random vehicle with available capacity
    unassigned = []
    for package in available_packages:
        # Try vehicles in random order
        vehicle_indices = list(range(len(solution)))
        random.shuffle(vehicle_indices)

        assigned = False
        for v_idx in vehicle_indices:
            if check_vehicle_capacity(solution[v_idx], package):
                solution[v_idx]["route"].append(package)
                assigned = True
                break

        if not assigned:
            unassigned.append(package)

    # Try to assign unassigned packages by making swaps or drop if no space
    if unassigned:
        # Sort unassigned packages by priority (lowest first)
        unassigned.sort(key=lambda p: p["priority"])

        for unassigned_pkg in unassigned:
            assigned = False
            vehicle_indices = list(range(len(solution)))
            random.shuffle(vehicle_indices)

            for v_idx in vehicle_indices:
                if not solution[v_idx]["route"]:
                    continue

                total_weight = sum(p["weight"] for p in solution[v_idx]["route"])
                smallest_idx = min(range(len(solution[v_idx]["route"])),
                                   key=lambda i: solution[v_idx]["route"][i]["weight"])
                smallest_pkg = solution[v_idx]["route"][smallest_idx]

                if total_weight + unassigned_pkg["weight"] - smallest_pkg["weight"] <= solution[v_idx]["capacity"]:
                    # Try to move smallest package elsewhere
                    for other_v_idx in vehicle_indices:
                        if v_idx != other_v_idx and check_vehicle_capacity(solution[other_v_idx], smallest_pkg):
                            solution[other_v_idx]["route"].append(smallest_pkg)
                            solution[v_idx]["route"].pop(smallest_idx)
                            if check_vehicle_capacity(solution[v_idx], unassigned_pkg):
                                solution[v_idx]["route"].append(unassigned_pkg)
                                assigned = True
                            break

                if assigned:
                    break



    # Shuffle or sort routes
    for vehicle in solution:
        if len(vehicle["route"]) > 1:
            if random.random() < 0.5:  # 50% chance to shuffle routes
                random.shuffle(vehicle["route"])
            else:
                vehicle["route"].sort(key=lambda p: p["priority"] + random.uniform(-0.5, 0.5))

    return solution


#------------------------------------------
# SIMULATED ANNEALING IMPLEMENTATION
#------------------------------------------
def get_neighbor_solution(solution):
    """Generate a neighbor solution by making a small change - for Simulated Annealing"""
    new_solution = []
    for vehicle in solution:
        new_solution.append({
            "capacity": vehicle["capacity"],
            "route": vehicle["route"].copy()
        })

    # Select random operation
    operation = random.choice(["swap", "move", "reverse"])

    if operation == "swap" and len(new_solution) >= 2:
        # Swap packages between two vehicles
        v1_idx = random.randint(0, len(new_solution) - 1)
        v2_idx = random.randint(0, len(new_solution) - 1)
        while v1_idx == v2_idx:
            v2_idx = random.randint(0, len(new_solution) - 1)

        if new_solution[v1_idx]["route"] and new_solution[v2_idx]["route"]:
            p1_idx = random.randint(0, len(new_solution[v1_idx]["route"]) - 1)
            p2_idx = random.randint(0, len(new_solution[v2_idx]["route"]) - 1)

            new_solution[v1_idx]["route"][p1_idx], new_solution[v2_idx]["route"][p2_idx] = \
                new_solution[v2_idx]["route"][p2_idx], new_solution[v1_idx]["route"][p1_idx]

    elif operation == "move":
        # Move a package from one vehicle to another
        v1_idx = random.randint(0, len(new_solution) - 1)
        v2_idx = random.randint(0, len(new_solution) - 1)

        if new_solution[v1_idx]["route"]:
            p1_idx = random.randint(0, len(new_solution[v1_idx]["route"]) - 1)
            package = new_solution[v1_idx]["route"].pop(p1_idx)
            new_solution[v2_idx]["route"].append(package)

    elif operation == "reverse":
        # Reverse a part of a route (useful for optimizing TSP-like parts)
        v_idx = random.randint(0, len(new_solution) - 1)
        if len(new_solution[v_idx]["route"]) >= 2:
            start = random.randint(0, len(new_solution[v_idx]["route"]) - 2)
            end = random.randint(start + 1, len(new_solution[v_idx]["route"]) - 1)
            new_solution[v_idx]["route"][start:end + 1] = reversed(new_solution[v_idx]["route"][start:end + 1])

    return new_solution


def simulated_annealing(vehicles, packages):
    """Implement the simulated annealing algorithm"""
    current_solution = create_initial_solution(vehicles, packages)
    best_solution = current_solution
    best_fitness = calculate_fitness(current_solution, packages)

    temperature = INITIAL_TEMPERATURE

    iteration = 0
    iteration_history = []  # To track progress

    while temperature > MIN_TEMPERATURE:
        for _ in range(ITERATIONS_PER_TEMP):
            next_solution = get_neighbor_solution(current_solution)
            current_fitness = calculate_fitness(current_solution, packages)
            next_fitness = calculate_fitness(next_solution, packages)

            # Calculate the change in fitness (delta E)
            delta_e = current_fitness - next_fitness  # Note: lower fitness is better

            # Accept the new solution if it's better or with a probability if it's worse
            if delta_e > 0 or random.random() < math.exp(delta_e / temperature):
                current_solution = next_solution
                current_fitness = next_fitness

                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()  # Make a deep copy
                    best_fitness = current_fitness
                    iteration_history.append((iteration, best_fitness))

        # Reduce temperature according to cooling schedule
        temperature *= COOLING_RATE
        iteration += 1

    return best_solution, best_fitness, iteration_history


#------------------------------------------
# GENETIC ALGORITHM IMPLEMENTATION
#------------------------------------------
def crossover(parent1, parent2):
    """Perform crossover between two parent solutions - for Genetic Algorithm"""
    # Create empty child with same vehicle structure
    child = []
    for vehicle in parent1:
        child.append({
            "capacity": vehicle["capacity"],
            "route": []
        })

    # Collect all packages from both parents
    all_packages = []
    for vehicle in parent1:
        all_packages.extend(vehicle["route"])

    # Use package assignment order from parent1 for first half
    # and parent2 for second half
    packages_from_parent1 = set()
    package_count = len(all_packages)

    # Get crossover point
    crossover_point = random.randint(1, package_count - 1) if package_count > 1 else 1

    # Collect packages from parent1 until crossover point using package attributes as identifier
    count = 0
    for vehicle in parent1:
        for package in vehicle["route"]:
            if count < crossover_point:
                # Use tuple of attributes as package identifier
                package_id = (package["x"], package["y"], package["weight"], package["priority"])
                packages_from_parent1.add(package_id)
                count += 1
            else:
                break
        if count >= crossover_point:
            break

    # Assign packages to child in order they appear in parents
    unassigned_packages = []

    # First, add packages from parent1 (until crossover point)
    for vehicle in parent1:
        for package in vehicle["route"]:
            package_id = (package["x"], package["y"], package["weight"], package["priority"])
            if package_id in packages_from_parent1:
                # Find a vehicle with enough capacity
                assigned = False
                for child_vehicle in child:
                    if sum(p["weight"] for p in child_vehicle["route"]) + package["weight"] <= child_vehicle[
                        "capacity"]:
                        child_vehicle["route"].append(package)
                        assigned = True
                        break

                if not assigned:
                    unassigned_packages.append(package)

    # Then, add remaining packages from parent2
    for vehicle in parent2:
        for package in vehicle["route"]:
            package_id = (package["x"], package["y"], package["weight"], package["priority"])
            if package_id not in packages_from_parent1:
                # Find a vehicle with enough capacity
                assigned = False
                for child_vehicle in child:
                    if sum(p["weight"] for p in child_vehicle["route"]) + package["weight"] <= child_vehicle[
                        "capacity"]:
                        child_vehicle["route"].append(package)
                        assigned = True
                        break

                if not assigned:
                    unassigned_packages.append(package)

    # Try to assign any unassigned packages to vehicles with available capacity
    for package in unassigned_packages:
        for child_vehicle in child:
            if sum(p["weight"] for p in child_vehicle["route"]) + package["weight"] <= child_vehicle["capacity"]:
                child_vehicle["route"].append(package)
                break

    return child


def mutate(solution):
    """Mutate a solution by making random changes - for Genetic Algorithm"""
    if random.random() < MUTATION_RATE:
        # Choose a mutation type
        mutation_type = random.choice(["swap", "move", "reverse", "shuffle"])

        if mutation_type == "swap" and len(solution) >= 2:
            # Swap packages between two vehicles
            v1_idx = random.randint(0, len(solution) - 1)
            v2_idx = random.randint(0, len(solution) - 1)
            while v1_idx == v2_idx:
                v2_idx = random.randint(0, len(solution) - 1)

            if solution[v1_idx]["route"] and solution[v2_idx]["route"]:
                p1_idx = random.randint(0, len(solution[v1_idx]["route"]) - 1)
                p2_idx = random.randint(0, len(solution[v2_idx]["route"]) - 1)

                solution[v1_idx]["route"][p1_idx], solution[v2_idx]["route"][p2_idx] = \
                    solution[v2_idx]["route"][p2_idx], solution[v1_idx]["route"][p1_idx]

        elif mutation_type == "move":
            # Move a package from one vehicle to another
            v1_idx = random.randint(0, len(solution) - 1)
            v2_idx = random.randint(0, len(solution) - 1)

            if solution[v1_idx]["route"]:
                p1_idx = random.randint(0, len(solution[v1_idx]["route"]) - 1)
                package = solution[v1_idx]["route"].pop(p1_idx)
                solution[v2_idx]["route"].append(package)

        elif mutation_type == "reverse":
            # Reverse a part of a route
            v_idx = random.randint(0, len(solution) - 1)
            if len(solution[v_idx]["route"]) >= 2:
                start = random.randint(0, len(solution[v_idx]["route"]) - 2)
                end = random.randint(start + 1, len(solution[v_idx]["route"]) - 1)
                solution[v_idx]["route"][start:end + 1] = reversed(solution[v_idx]["route"][start:end + 1])

        elif mutation_type == "shuffle":
            # Shuffle a vehicle's route
            v_idx = random.randint(0, len(solution) - 1)
            if len(solution[v_idx]["route"]) >= 2:
                random.shuffle(solution[v_idx]["route"])

    return solution


def genetic_algorithm(vehicles, packages):
    """Implement the genetic algorithm"""
    # Initialize population
    population = []
    for _ in range(POPULATION_SIZE):
        population.append(create_initial_solution(vehicles, packages))

    best_solution = None
    best_fitness = float('inf')
    generation_history = []  # To track progress

    # Evolution over generations
    for generation in range(MAX_GENERATIONS):
        # Calculate fitness for each individual
        fitness_values = []
        for solution in population:
            fitness = calculate_fitness(solution, packages)
            if not math.isfinite(fitness):
                fitness = 1e6  # Penalize badly instead of infinity
            fitness_values.append(fitness)

        # Update best solution
        min_fitness_idx = fitness_values.index(min(fitness_values))
        if fitness_values[min_fitness_idx] < best_fitness:
            best_solution = population[min_fitness_idx]
            best_fitness = fitness_values[min_fitness_idx]
            generation_history.append((generation, best_fitness))

        # Create weights for selection (inverse of fitness since lower is better)
        max_fitness = max(fitness_values)
        if not math.isfinite(max_fitness):
            raise ValueError("Non-finite fitness value detected.")

        # Compute raw weights
        weights = [max_fitness - f if math.isfinite(f) else 0 for f in fitness_values]

        # Make sure all weights are non-negative
        min_weight = min(weights)
        if min_weight < 0:
            weights = [w - min_weight for w in weights]  # shift weights to be >= 0

        total_weight = sum(weights)

        # If all weights are zero, fall back to uniform selection
        if total_weight == 0:
            weights = [1 / len(weights)] * len(weights)
        else:
            weights = [w / total_weight for w in weights]

        # Create new population
        new_population = []

        # Elitism: keep the best solution
        new_population.append(population[min_fitness_idx])

        while len(new_population) < POPULATION_SIZE:
            # Select parents
            parent1_idx = random.choices(range(len(population)), weights=weights)[0]
            parent2_idx = random.choices(range(len(population)), weights=weights)[0]

            # Crossover
            child = crossover(population[parent1_idx], population[parent2_idx])

            # Mutation
            child = mutate(child)

            # Add to new population
            new_population.append(child)

        # Replace old population
        population = new_population

    return best_solution, best_fitness, generation_history


#------------------------------------------
# FILE I/O AND DATA HANDLING
#------------------------------------------
def read_data(filename):
    """Read vehicle and package data from file"""
    vehicles = []
    packages = []

    with open(filename) as f:
        package_id = 1  # Initialize package ID counter
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith("Vehicle"):
                _, data = line.split(":")
                capacity = float(data.strip())
                vehicles.append({
                    "capacity": capacity,
                    "route": []
                })
            elif line.startswith("Package"):
                _, data = line.split(":")
                x, y, weight, priority = map(float, data.strip().split())
                packages.append({
                    "id": package_id,  # Add package ID
                    "x": x,
                    "y": y,
                    "weight": weight,
                    "priority": int(priority)
                })
                package_id += 1  # Increment ID for next package

    return vehicles, packages


#------------------------------------------
# VISUALIZATION AND OUTPUT FUNCTIONS
#------------------------------------------
def visualize_solution(solution, title="Package Delivery Route Optimization", filename="solution_plot.png"):
    """Visualize the solution with matplotlib"""
    plt.figure(figsize=(12, 10))

    # Plot shop location
    plt.scatter(SHOP_X, SHOP_Y, c='black', s=100, marker='s', label='Shop')

    # Colors for different vehicles
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    # Track total statistics
    total_vehicles_used = 0
    total_weight = 0
    total_distance = 0

    for i, vehicle in enumerate(solution):
        color = colors[i % len(colors)]
        route = vehicle["route"]

        if not route:
            continue

        total_vehicles_used += 1

        # Calculate vehicle statistics
        vehicle_weight = sum(package["weight"] for package in route)
        total_weight += vehicle_weight

        route_distance = calculate_route_distance(route)
        total_distance += route_distance

        # Plot packages
        for package in route:
            plt.scatter(package["x"], package["y"], c=color, s=50, alpha=0.7)
            # Use package ID instead of priority
            plt.text(package["x"] + 1, package["y"] + 1, f"#{package['id']}", fontsize=8)

        # Plot route
        x_coords = [SHOP_X] + [package["x"] for package in route] + [SHOP_X]
        y_coords = [SHOP_Y] + [package["y"] for package in route] + [SHOP_Y]

        # Include vehicle info in the label
        vehicle_label = f"Vehicle {i + 1} ({vehicle_weight:.1f}kg, {route_distance:.1f}km)"
        plt.plot(x_coords, y_coords, c=color, alpha=0.5, label=vehicle_label)

        # Add arrows to show direction
        for j in range(len(x_coords) - 1):
            plt.arrow(x_coords[j], y_coords[j],
                      (x_coords[j + 1] - x_coords[j]) * 0.9,
                      (y_coords[j + 1] - y_coords[j]) * 0.9,
                      head_width=1, head_length=1, fc=color, ec=color, alpha=0.5)

    # Add summary information text box
    summary_text = f"Summary:\n" \
                   f"Vehicles used: {total_vehicles_used}/{len(solution)}\n" \
                   f"Total weight: {total_weight:.1f}kg\n" \
                   f"Total distance: {total_distance:.1f}km"

    plt.figtext(0.02, 0.02, summary_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.xlabel('X-coordinate (km)')
    plt.ylabel('Y-coordinate (km)')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Plot saved as {filename}")


def visualize_progress(algorithm_history, algorithm_name):
    """Visualize the progress of the algorithm"""
    iterations = [item[0] for item in algorithm_history]
    fitness_values = [item[1] for item in algorithm_history]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, fitness_values, 'b-', linewidth=2)
    plt.xlabel('Iteration/Generation')
    plt.ylabel('Best Fitness (Lower is Better)')
    plt.title(f'{algorithm_name} Progress')
    plt.grid(True)
    #plt.savefig(f"{algorithm_name.lower().replace(' ', '_')}_progress.png")
    #print(f"{algorithm_name} progress plot saved as {algorithm_name.lower().replace(' ', '_')}_progress.png")


def print_solution(solution, total_distance):
    """Print details of the solution"""
    print("\nOptimized Solution:")
    print(f"Total distance: {total_distance:.2f} km")

    for i, vehicle in enumerate(solution):
        route = vehicle["route"]
        total_weight = sum(package["weight"] for package in route)
        route_distance = calculate_route_distance(route)

        print(f"\nVehicle {i + 1} (Capacity: {vehicle['capacity']} kg, Used: {total_weight:.2f} kg)")
        print(f"Route distance: {route_distance:.2f} km")

        if route:
            print("Route: Shop -> ", end="")
            for j, package in enumerate(route):
                # Include both package ID and priority
                print(f"(#{package['id']}, {package['x']:.1f}, {package['y']:.1f}, P{int(package['priority'])}) -> ",
                      end="")
                if (j + 1) % 2 == 0 and j < len(route) - 1:  # Changed to 2 packages per line for better readability
                    print("\n       ", end="")
            print("Shop")
        else:
            print("Route: Empty")


#------------------------------------------
# VALIDATION AND UTILITY FUNCTIONS
#------------------------------------------
def validate_solution(solution, packages):
    """Validate that all packages are assigned and weight constraints are met"""
    # Check that all packages are assigned
    assigned_packages = []
    for vehicle in solution:
        assigned_packages.extend(vehicle["route"])

    if len(assigned_packages) != len(packages):
        print(f"Error: Not all packages assigned. {len(assigned_packages)} assigned out of {len(packages)}")
        return False

    # Check that weight constraints are met
    for i, vehicle in enumerate(solution):
        total_weight = sum(package["weight"] for package in vehicle["route"])
        if total_weight > vehicle["capacity"]:
            print(f"Error: Vehicle {i + 1} is overloaded. Capacity: {vehicle['capacity']}, Load: {total_weight}")
            return False

    return True


def deep_copy_solution(solution):
    """Create a deep copy of a solution"""
    new_solution = []
    for vehicle in solution:
        new_vehicle = {
            "capacity": vehicle["capacity"],
            "route": []
        }
        for package in vehicle["route"]:
            new_vehicle["route"].append(package.copy())
        new_solution.append(new_vehicle)
    return new_solution


#------------------------------------------
# MAIN FUNCTION
#------------------------------------------
def main():
    # Set random seed based on current time for different results each run
    random.seed(int(time.time()))

    # Ask user for data file
    filename = input("Enter data file name (default: data.txt): ") or "data.txt"

    try:
        vehicles, packages = read_data(filename)
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    print(f"Loaded {len(vehicles)} vehicles and {len(packages)} packages.")

    # Ask user which algorithm to use
    while True:
        algorithm = input("Choose algorithm (1 for Simulated Annealing, 2 for Genetic Algorithm, 3 for Both): ")
        if algorithm in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")

    start_time = time.time()

    if algorithm == '1' or algorithm == '3':
        print("\nRunning Simulated Annealing...")
        sa_solution, sa_fitness, sa_history = simulated_annealing(vehicles, packages)
        sa_end_time = time.time()
        print(f"Simulated Annealing completed in {sa_end_time - start_time:.2f} seconds.")

        # Show assignment count even if some packages are dropped
        assigned = sum(len(v["route"]) for v in sa_solution)
        print(f"Simulated Annealing assigned {assigned}/{len(packages)} packages.")
        if assigned < len(packages):
            print(f"Warning: {len(packages) - assigned} packages were not assigned.")

        print_solution(sa_solution, sa_fitness)
        visualize_solution(sa_solution, "Simulated Annealing Solution", "sa_solution_plot.png")
        visualize_progress(sa_history, "Simulated Annealing")

    if algorithm == '2' or algorithm == '3':
        ga_start_time = time.time() if algorithm == '2' else sa_end_time
        print("\nRunning Genetic Algorithm...")
        ga_solution, ga_fitness, ga_history = genetic_algorithm(vehicles, packages)
        ga_end_time = time.time()
        print(f"Genetic Algorithm completed in {ga_end_time - ga_start_time:.2f} seconds.")

        assigned = sum(len(v["route"]) for v in ga_solution)
        print(f"Genetic Algorithm assigned {assigned}/{len(packages)} packages.")
        if assigned < len(packages):
            print(f"Warning: {len(packages) - assigned} packages were not assigned.")

        print_solution(ga_solution, ga_fitness)
        visualize_solution(ga_solution, "Genetic Algorithm Solution", "ga_solution_plot.png")
        visualize_progress(ga_history, "Genetic Algorithm")

    if algorithm == '3':
        # Compare results
        print("\nComparison:")
        print(f"Simulated Annealing: {sa_fitness:.2f} km in {sa_end_time - start_time:.2f} seconds")
        print(f"Genetic Algorithm: {ga_fitness:.2f} km in {ga_end_time - ga_start_time:.2f} seconds")

        # Plot comparison
        plt.figure(figsize=(12, 6))

        # Normalize iterations to percentage
        sa_iterations = [it / max(sa_history[-1][0], 1) * 100 for it, _ in sa_history]
        ga_iterations = [it / max(ga_history[-1][0], 1) * 100 for it, _ in ga_history]

        sa_fitness_values = [fit for _, fit in sa_history]
        ga_fitness_values = [fit for _, fit in ga_history]

        plt.plot(sa_iterations, sa_fitness_values, 'r-', label='Simulated Annealing')
        plt.plot(ga_iterations, ga_fitness_values, 'b-', label='Genetic Algorithm')
        plt.xlabel('Progress (%)')
        plt.ylabel('Best Fitness (Lower is Better)')
        #plt.title('Algorithm Comparison')
        plt.legend()
        plt.grid(True)
        #plt.savefig("algorithm_comparison.png")
        #print("Comparison plot saved as algorithm_comparison.png")


if __name__ == "__main__":
    main()