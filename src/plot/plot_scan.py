import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ..config.config import Config, Recorder
import pandas as pd
import argparse
import tqdm

from ..problem import Problem
from ..run.scan_quadratic import parse_shortform_config

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm")
parser.add_argument("-p", "--prefix", default="**")
parser.add_argument("-P", "--problem")
parser.add_argument("-v", "--noise_variance", default=0)
parser.add_argument("-S", "--savefile", default=None)
parser.add_argument("-D", "--dumpfile", default=None)
parser.add_argument('--noshow', action='store_true')
args = parser.parse_args()

target_times = [ 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000 ]
plotting_times = [ i * s for i in [1] + target_times for s in [ 1, 2, 3, 5 ] ]

all_special_times = sorted(np.unique(target_times + plotting_times))

l_algorithm = list(parse_shortform_config(args.algorithm))
l_problem = list(parse_shortform_config(args.problem))
l_noise_variance = list(map(float, parse_shortform_config(args.noise_variance)))
l_prefix = list(parse_shortform_config(args.prefix))

ymax = 1e6
ymax_auto = 1e-8

CLIP_LOSS = np.float64(+1e-40)

gathered_register = {} if args.dumpfile is not None else None


fig = plt.figure(figsize=( 5 * len(l_algorithm), 5 * len(l_problem) ))
gs = matplotlib.gridspec.GridSpec(len(l_problem), len(l_algorithm))
ax = []
t_ax = []
for i in range(len(l_problem)):
    l_ax = []
    for j in range(len(l_algorithm)):
        kw = {}
        if j > 0:
            kw["sharey"] = l_ax[0]
        if i > 0:
            kw["sharex"] = ax[0][j]
        p_ax = fig.add_subplot(gs[i, j], **kw)
        l_ax.append(p_ax)
    ax.append(l_ax)
    t_ax.append([ None for _ in l_ax ])


def fetch_table(algorithm, problem, l_noise_variance):
    target_configs = Config.all_recorded(
            problem=problem,
            algorithm=algorithm,
            prefix=l_prefix,
            )

    all_rec = [ Recorder() for _ in l_noise_variance ]

    for config in target_configs:
        step = config.headers.algorithm_args.get("eta", None)

        n_var = config.headers.problem_args.get("noise_variance", None)

        for n_idx, noise_variance in enumerate(l_noise_variance):
            if (n_var != noise_variance):
                continue

            for t in all_special_times:
                select = config.data[config.data["iteration"] == t]
                if select.size == 0:
                    continue
                loss_val = select["loss"].iloc[-1]

                if not np.isfinite(loss_val):
                    continue

                if "train_accuracy" in select:
                    train_acc, _ = select["train_accuracy"].iloc[-1]
                else:
                    train_acc = np.nan

                if "test_accuracy" in select:
                    test_acc, _ = select["test_accuracy"].iloc[-1]
                else:
                    test_acc = np.nan

                if (not np.isfinite(loss_val)) or (loss_val > 1e120):
                    # continue
                    loss_val = 1e120

                all_rec[n_idx].record({
                    "step_size": step,
                    "iteration": t,
                    "value": loss_val + 1e-40,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    })

    all_df = []

    for rec in all_rec:
        df = rec.to_dataframe()
        if df.size == 0:
            all_df.append( None )
            continue
        assert df.size > 0, f"Found no elements for ({algorithm}, {problem}), did you run the experiments?"
        df = df.sort_values(by='step_size')

        df = df.groupby([ 'step_size', 'iteration' ], as_index=False)\
                .agg(
                    mean = ( 'value', lambda x: x.mean() if x.size > 0 else np.nan ),
                    median = ( 'value', lambda x: x.median() if x.size > 0 else np.nan ),
                    train_acc = ( 'train_acc', lambda x: x[np.isfinite(x)].median() if np.isfinite(x).sum() > 0 else np.nan ),
                    test_acc = ( 'test_acc', lambda x: x[np.isfinite(x)].median() if np.isfinite(x).sum() > 0 else np.nan ),
                    )
        all_df.append(df)
    return all_df


def plot_all_perf(df, p_idx, problem, a_idx, algorithm):
    noise_var = l_noise_variance[0]
    assert len(l_noise_variance) == 1

    p_ax = ax[p_idx][a_idx]
    twin = None

    if gathered_register is not None:
        key = (p_idx, problem, a_idx, algorithm)
        gathered_register[key] = df

    for t in target_times:
        sub_df = df[df['iteration'] == t]
        if sub_df.size == 0 or t == 1:
            continue
        l, = p_ax.plot(sub_df['step_size'], sub_df['median'], '-o', label=f"T={t:.1e}")

    p_ax.legend()


def plot_all_perf_axis(p_idx, problem, a_idx, algorithm):
    noise_var = l_noise_variance[0]
    assert len(l_noise_variance) == 1

    p_ax = ax[p_idx][a_idx]

    p_ax.set_ylim(1e-16, ymax)
    if p_idx == 0:
        p_ax.set_title(f"'{algorithm}' (NV {noise_var:.0e})")
    p_ax.set_xlabel(r"Step-size $\eta$")

    if a_idx == 0:
        p_ax.set_ylabel(f"{problem}\nMedian Loss Value")

    p_ax.set_xscale('log')
    p_ax.set_yscale('log')
    p_ax.grid(which='major', alpha=0.8)
    p_ax.grid(which='minor', alpha=0.4)

lowest_y = 1.0

for p_idx, problem in enumerate(tqdm.tqdm(l_problem, leave=False, desc="Problem")):
    y_offset = 0
    lowest_y = 1.0

    for a_idx, algorithm in enumerate(tqdm.tqdm(l_algorithm, leave=False, desc="Algorithm")):

        plot_all_perf_axis(p_idx, problem, a_idx, algorithm)
        all_df = fetch_table(algorithm, problem, l_noise_variance)
        df = all_df[0]
        if df is None:
            continue
        df['median'] = (df['median'] - y_offset) + CLIP_LOSS
        plot_all_perf(df, p_idx, problem, a_idx, algorithm)
        lowest_y = min(lowest_y, df['median'].min())
        ymax_auto = max(ymax_auto, df['median'].max())

    ylim_min = 10. ** ( np.floor(np.log10(lowest_y * 0.9) ) )
    if np.isfinite(ymax_auto) and ymax_auto < 1e60:
        _ymax = min(ymax_auto * 10, ymax)
    else:
        _ymax = ymax
    for a_idx, _ in enumerate(l_algorithm):
        ax[p_idx][a_idx].set_ylim(ylim_min, _ymax)

fig.suptitle(r"Perf vs $\eta$, grouped by training time $T$")

for i in range(len(l_problem)):
    for j in range(len(l_algorithm)):
        if j > 0:
            ax[i][j].yaxis.label.set_visible(False)
        if i < len(l_problem) - 1:
            ax[i][j].xaxis.label.set_visible(False)

gs.tight_layout(fig, rect=[0, 0, 1, 0.97])

if args.savefile is not None:
    plt.savefig(args.savefile, dpi=800)

special_etas = {
        'cocob-backprop': [ 1e-2 ],
        'stab-sgd-oracle': [ 1e0 ],
        'stab-sgd-inline': [ 1e0 ],
        }

if gathered_register is not None:
    for key in gathered_register.keys():
        _, problem, _, algorithm = key
        df = gathered_register[key].copy()
        df = df[ (np.log10(df['step_size']) * 2) % 1 == 0 ]  # remove not aligned steps
        full_df = df.copy()

        df = df[ (np.log10(df['iteration'])) % 1 == 0 ]  # remove too many points

        df = df.astype({ 'iteration': np.float32 })
        variance = l_noise_variance[0]
        suffix = f"_T-ALL_P{problem}_v{variance:.1e}_A{algorithm}"
        filename = args.dumpfile + suffix + ".csv"
        df.to_csv(filename, index=False, float_format="%.3e")


        df = full_df  # restore all points for the horizons.

        results = []
        times = [ t for t in df['iteration'].unique() if t in target_times ]

        alt_plotting_times = [ i * t for t in target_times for i in [ 1, 3 ] ]
        plot_times = [ t for t in df['iteration'].unique() if t in alt_plotting_times ]

        for tune_h in times:
            for train_h in plot_times:
                if tune_h < 10:
                    continue
                sub = df[df['iteration'] == tune_h]
                eta = sub.iloc[sub['mean'].argmin()].loc['step_size'].item()

                sub = df[df['iteration'] == train_h]
                value = sub[sub['step_size'] == eta]
                if value.size == 0:
                    continue
                results.append([ float(tune_h), float(train_h), eta, value['mean'].item() ])

        if algorithm in special_etas:
            for eta in special_etas[algorithm]:
                for train_h in plot_times:
                    sub = df[df['iteration'] == train_h]
                    value = sub[sub['step_size'] == eta]
                    if value.size == 0:
                        continue
                    results.append([ np.nan, float(train_h), eta, value['mean'].item() ])

        columns = ['tune_horizon', 'train_horizon', 'step_size', 'value']
        df2 = pd.DataFrame(results, columns=columns)
        df2['algorithm'] = algorithm
        second_suffix = f"_Horizon_P{problem}_v{variance:.1e}_A{algorithm}"
        filename = args.dumpfile + second_suffix + ".csv"
        df2.to_csv(filename, index=False, float_format="%.3e")

if not args.noshow:
    plt.show()
