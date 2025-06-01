from opfunu.cec_based import cec2022
import numpy as np

functions = [cec2022.F12022, cec2022.F22022, cec2022.F32022, cec2022.F42022,
             cec2022.F52022, cec2022.F62022, cec2022.F72022, cec2022.F82022,
             cec2022.F92022, cec2022.F102022, cec2022.F112022, cec2022.F122022]

def save_results_to_txt(results, filename="coordinates.txt"):
    with open(filename, "w") as f:
        for coords in results:
            line = "\t".join(f"{x:.14f}" for x in coords)
            f.write(line + "\n")

def test_tabu_search(functions):
    print("-------------------------- TABU SEARCH --------------------------")
    
    results = []
    
    for f in functions:
        best_point, best_cost = tabu_search(f)
        print("-------------------------------------------------------------------------")
        print(f"Function: {f.__name__}")
        print(f"Best cost: {best_cost}")
        print(f"Best parameters: {best_point}")
        
        results.append(best_point)
    
    save_results_to_txt(results, "tabu_search_coordinates.txt")
    
    print("Results saved to tabu_search_coordinates.txt")
    print()

def gen_neighborhood(neighborhood_size, current_point, upper, lower, ndim):
    neighborhood = []

    for _ in range(neighborhood_size):
        step = np.random.uniform(-0.1, 0.1, size=ndim) * (upper - lower)
        neighbor = current_point + step
        neighbor = np.clip(neighbor, lower, upper)
        neighborhood.append(neighbor)

    return neighborhood

def tabu_search(function, max_iter=5000, tabu_max=100, neighborhood_size=200, print_interval=1000, tabu_threshold=1.0):
    func = function(ndim=20)
    lower = np.array(func.lb)
    upper = np.array(func.ub)
    ndim = 20
    
    current_point = np.random.uniform(lower, upper)
    current_cost = func.evaluate(current_point)
    best_point = current_point.copy()
    best_cost = current_cost
    
    tabu_list = []

    for iteration in range(1, max_iter + 1):
        neighborhood = gen_neighborhood(neighborhood_size, current_point, upper, lower, ndim)
        
        candidate_points = []
        candidate_costs = []
        
        for neighbor in neighborhood:
            cost = func.evaluate(neighbor)
            if any(np.linalg.norm(neighbor - t) < tabu_threshold for t in tabu_list) and cost >= best_cost:
                continue
            candidate_points.append(neighbor)
            candidate_costs.append(cost)
        
        if not candidate_points:
            tabu_list = []
            continue
        
        idx_best = np.argmin(candidate_costs)
        best_candidate = candidate_points[idx_best]
        best_candidate_cost = candidate_costs[idx_best]
        
        current_point = best_candidate
        current_cost = best_candidate_cost
        
        tabu_list.append(current_point)
        if len(tabu_list) > tabu_max:
            tabu_list.pop(0)

        if current_cost < best_cost:
            best_cost = current_cost
            best_point = current_point.copy()
        
        if iteration % print_interval == 0:
            print(f"Iteration {iteration}/{max_iter} — Best cost so far: {best_cost:.6f}")
    
    return best_point, best_cost

test_tabu_search(functions)
