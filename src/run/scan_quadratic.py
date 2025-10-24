import argparse
import numpy as np
import tqdm

from ..config.config import Config, Recorder
from ..algorithm import Algorithm, FirstOrderAlgorithm, StabOracleAlgorithm
from ..problem import Problem

def run_experiment(config, progress=True):
    problem = config.init_problem()
    alg = config.init_algorithm()

    x = problem.init_point()
    s = alg.init_state(x)
    rec = Recorder()
    p, g = None, None

    loss_val =  problem.mean_value(x).item()
    rec.record({ "iteration": 1, "loss": loss_val })
    iterator = range(1, int(config.headers.n_iter))
    if progress:
        iterator = tqdm.tqdm(iterator, leave=False)

    for i in iterator:
        if progress:
            iterator.set_postfix({ "L": f"{loss_val:.4e}" })

        if isinstance(alg, FirstOrderAlgorithm):
            # Distinguish probe (p) to measure gradients, from guess (g) used to evaluate loss obtained
            # classical use case: probe is the last iterate, guess is the average iterate
            p, g = (p if p is not None else x), (g if g is not None else x)
            f = problem.sample_value(p)
            grad = problem.sample_gradient(p)
            p, g, s = alg.update(p, f, grad, s)
            x = g
        elif isinstance(alg, StabOracleAlgorithm):
            grad = problem.sample_gradient(x)
            stab = problem.stab_oracle(x)
            x, s = alg.update(x, grad, stab, s)
        else:
            grad = problem.sample_gradient(x)
            x, s = alg.update(x, grad, s)

        ll = int(np.ceil(np.log10(i+1)))
        should_record = (ll < 2) or ((i + 1) % int(10 ** (ll-1)) == 0)

        if should_record:
            loss_garbage_val = 1e+120

            loss_val = problem.mean_value(x).item()
            rec.record({
                "iteration": i+1,
                "loss": loss_val if np.isfinite(loss_val) else loss_garbage_val,
                })

            if not np.isfinite(loss_val):
                # abort early, still mark loss as explosion
                for j in range(i+1, int(config.headers.n_iter)):
                    ll = int(np.ceil(np.log10(j+1)))
                    if (ll < 2) or ((j + 1) % int(10 ** (ll-1)) == 0):
                        rec.record({
                            "iteration": j+1,
                            "loss": loss_garbage_val,
                            })
                break

    config.register_recorded_data(rec.dump())
    config.write_to_file(config.savefile)

def parse_shortform_config(string):
    string = str(string)
    if ":" in string:
        split = string.split(':')
        if len(split) == 2:
            s0, s1 = [ int(float(s)) for s in split ]
            return np.arange(s0, s1+1)
        elif len(split) == 3:
            s0, s1, step = [ float(s) for s in split ]
            return np.arange(s0, s1 + 1e-8, step)
        else:
            raise ValueError(f"Unrecognized split '{split}'")
    elif "," in string:
        return string.split(',')
    else:
        return [ string ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--log10_eta", required=True)
    parser.add_argument("-a", "--algorithm", required=True)
    parser.add_argument("-v", "--noise_variance", default=0)
    parser.add_argument("-s", "--seed", default=1)
    parser.add_argument("--noprogress", action='store_true')
    parser.add_argument("-T", "--n_iter", default=1_000)
    parser.add_argument("-P", "--problem", required=True)
    parser.add_argument("-p", "--prefix", default=None)
    args = parser.parse_args()

    l_eta = list(map(lambda u: 10. ** float(u), parse_shortform_config(args.log10_eta)))
    n_iter = float(args.n_iter)
    l_noise_variance = list(map(float, parse_shortform_config(args.noise_variance)))
    l_seed = list(map(int, parse_shortform_config(args.seed)))

    l_algorithm = parse_shortform_config(args.algorithm)
    l_problem = parse_shortform_config(args.problem)

    for algorithm in l_algorithm:
        assert algorithm in Algorithm.class_dict, algorithm

    for problem  in l_problem:
        assert problem in Problem.class_dict, problem

    expected_count = len(l_algorithm) * len(l_problem) * len(l_seed) * len(l_noise_variance) * len(l_eta)
    count = 0

    for seed in l_seed:
      for problem in l_problem:
        for noise_variance in l_noise_variance:
          for algorithm in l_algorithm:
            for eta in l_eta:
              print(dict(
                      problem=problem,
                      problem_args={ "noise_variance": noise_variance, "seed": seed },
                      algorithm=algorithm,
                      algorithm_args={ "eta": eta },
                      n_iter=n_iter,
                      prefix=args.prefix,
                      ))
              c = Config(
                      problem=problem,
                      problem_args={ "noise_variance": noise_variance, "seed": seed },
                      algorithm=algorithm,
                      algorithm_args={ "eta": eta },
                      n_iter=n_iter,
                      prefix=args.prefix,
                      )
              count += 1
              run_experiment(c, progress=(not args.noprogress))
    print(f"Done, expected {expected_count}, got {count}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
