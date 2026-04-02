import numpy as np
from scipy.optimize import differential_evolution, minimize

def optimize_composition(sr_equation_func, target_value, T, bounds=None, n_starts=50):
    if bounds is None:
        bounds = [(0, 40)] * 5

    def objective(x):
        if abs(sum(x) - 100) > 0.1:
            return 1e10
        pred = sr_equation_func(x, T)
        return (pred - target_value) ** 2

    def constraint_sum(x):
        return sum(x) - 100

    constraints = [{"type": "eq", "fun": constraint_sum}]

    result_de = differential_evolution(
        objective, bounds=bounds, seed=42, maxiter=1000, tol=1e-10,
        constraints=[{"type": "eq", "fun": constraint_sum}] if hasattr(differential_evolution, "constraints") else ()
    )

    best_result = result_de
    for _ in range(n_starts):
        x0 = np.random.dirichlet(np.ones(5)) * 100
        try:
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
            if res.fun < best_result.fun:
                best_result = res
        except Exception:
            continue

    return best_result


def pareto_front(sr_func1, sr_func2, T, n_samples=10000, bounds=None):
    if bounds is None:
        bounds = [(0, 40)] * 5

    compositions = []
    for _ in range(n_samples):
        x = np.random.dirichlet(np.ones(5)) * 100
        x = np.clip(x, bounds[0][0], bounds[0][1])
        x = x / x.sum() * 100
        compositions.append(x)

    obj1 = [sr_func1(x, T) for x in compositions]
    obj2 = [sr_func2(x, T) for x in compositions]

    pareto = []
    for i in range(len(compositions)):
        dominated = False
        for j in range(len(compositions)):
            if obj1[j] >= obj1[i] and obj2[j] <= obj2[i] and (obj1[j] > obj1[i] or obj2[j] < obj2[i]):
                dominated = True
                break
        if not dominated:
            pareto.append({"comp": compositions[i].tolist(), "obj1": obj1[i], "obj2": obj2[i]})

    return sorted(pareto, key=lambda x: x["obj1"])
