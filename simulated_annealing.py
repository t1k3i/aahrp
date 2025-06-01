from opfunu . cec_based import cec2022
import numpy as np
import math


# Choose the first five benchmark functions
functions = [cec2022.F12022,cec2022.F22022, cec2022.F32022, cec2022.F42022,
    cec2022.F52022, cec2022.F62022, cec2022.F72022, cec2022.F82022, cec2022.F92022,
    cec2022.F102022, cec2022.F112022, cec2022.F122022]

def save_results_to_txt(results, filename="coordinates.txt"):
    with open(filename, "w") as f:
        for coords in results:
            line = "\t".join(f"{x:.14f}" for x in coords)
            f.write(line + "\n")

def test_simulated_annealing(functions):
    print("-------------------------- SIMULATED ANNEALING --------------------------")
    
    results = []
    
    for f in functions:
        best_point, best_cost = simulated_annealing(f)
        print("-------------------------------------------------------------------------")
        print(f"Function: {f.__name__}")
        print(f"Best cost: {best_cost}")
        print(f"Best parameters: {best_point}")
        
        results.append(best_point)
    
    save_results_to_txt(results, "aimulated_annealing_coordinates.txt")
    
    print("Results saved to aimulated_annealing_coordinates.txt")
    print()

def simulated_annealing(function, t_init=1.0, t_final=1e-5, max_iter=500000, reanneal_interval=1000):
    func = function(ndim=20)
    lower = np.array(func.lb)
    upper = np.array(func.ub)
    ndim = 20

    current_point = np.random.uniform(lower, upper)
    current_cost = func.evaluate(current_point)
    best_point = current_point.copy()
    best_cost = current_cost

    t = t_init
    iteration = 0
    no_improve_counter = 0

    while t > t_final and iteration < max_iter:
        step_size = 0.1 * (t / t_init)
        step = np.random.uniform(-1, 1, size=ndim) * step_size * (upper - lower)
        candidate_point = current_point + step
        candidate_point = np.clip(candidate_point, lower, upper)

        candidate_cost = func.evaluate(candidate_point)
        delta = candidate_cost - current_cost

        if delta < 0 or np.random.rand() < math.exp(-delta / t):
            current_point = candidate_point
            current_cost = candidate_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_point = current_point.copy()
                no_improve_counter = 0
            else:
                no_improve_counter += 1
        else:
            no_improve_counter += 1

        t = t_init / (1 + math.log(1 + iteration))

        if no_improve_counter >= reanneal_interval:
            t = min(t_init, t * 1.2)
            current_point = np.random.uniform(lower, upper)
            current_cost = func.evaluate(current_point)
            no_improve_counter = 0

        iteration += 1

    return best_point, best_cost

test_simulated_annealing(functions)


