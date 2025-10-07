import numpy as np
from utils_metrics import evaluate 

def sample_indices(n, sample_fraction=0.9, random_state=None, rep=0):
    rng = np.random.default_rng(random_state+rep)
    perm =  rng.permutation(n)
    k = int(round(sample_fraction * n))
    return perm[:k], perm[k:]

def run_pipeline(V_clean, y_true, noise_specs, algos, k=40, repeats=5, random_state=None):
    n = V_clean.shape[1]
    results = []
    for noise in noise_specs:
        noise_name, noise_fn, noise_kwargs = noise["name"], noise.get("fn"), (noise.get("kwargs") or {})
        for algo in algos:
            algo_name, algo_fn, algo_kwargs = algo["name"], algo["fn"], (algo.get("kwargs") or {})
            rre_list, acc_list, nmi_list = [], [], []
            for rep in range(repeats):
                idx_sub, _ = sample_indices(n, random_state=random_state+rep)
                V_sub, y_sub = V_clean[:,idx_sub], y_true[idx_sub]
                if noise_fn is None:
                    V_noisy = V_sub
                else:
                    V_noisy, _ = noise_fn(V_sub, **{**noise_kwargs, "random_state": random_state+rep})
                print(f"Repetition: {rep+1}, Algorithm={algo_name}, Noise={noise_name}")
                W, H = algo_fn(V_noisy, k=k, random_state=random_state+rep, **algo_kwargs)
                rre, acc, nmi = evaluate(V_sub, W, H, y_sub)
                rre_list.append(rre)
                acc_list.append(acc)
                nmi_list.append(nmi)
            results.append({
                "noise": noise_name,
                "algo": algo_name,
                "RRE_mean": float(np.mean(rre_list)), "RRE_std": float(np.std(rre_list)),
                "ACC_mean": float(np.mean(acc_list)), "ACC_std": float(np.std(acc_list)),
                "NMI_mean": float(np.mean(nmi_list)), "NMI_std": float(np.std(nmi_list)),
            })
    return results


