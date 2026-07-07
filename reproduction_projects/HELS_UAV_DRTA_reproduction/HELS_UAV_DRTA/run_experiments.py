"""
批量实验运行脚本
Usage: python run_experiments.py --scale small --env rural --n_runs 100
       python run_experiments.py --ablation --n_runs 10
       python run_experiments.py --comparison --n_runs 10
"""
import sys, os, argparse, subprocess, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.scenario_config import list_scenarios
from config.hyperparams import MAX_EPISODES

PYTHON = sys.executable


def run_single(scenario, run_id, device='cuda'):
    """Run one training run"""
    cmd = [PYTHON, 'train.py', '--scenario', scenario,
           '--run_id', str(run_id), '--device', device,
           '--episodes', str(MAX_EPISODES)]
    print(f"[Run] {scenario} #{run_id}")
    return subprocess.run(cmd)


def run_all(scenarios=None, n_runs=100, device='cuda'):
    """Run all specified scenarios"""
    if scenarios is None:
        scenarios = list_scenarios()
    total = len(scenarios) * n_runs
    print(f"Running {total} training jobs ({len(scenarios)} scenarios x {n_runs} runs each)")
    completed = 0
    for s in scenarios:
        for r in range(n_runs):
            ret = run_single(s, r, device)
            if ret.returncode != 0:
                print(f"  WARNING: {s} #{r} failed (exit {ret.returncode})")
            completed += 1
            if completed % 10 == 0:
                print(f"  Progress: {completed}/{total}")
    print(f"Complete: {completed}/{total}")


def run_ablation(scenario='large_rural', n_runs=10, device='cuda'):
    """Run ablation: MADDPG variants"""
    raise NotImplementedError(
        "Ablation requires MADDPG_Basic, MADDPG_Attn, MADDPG_RND variants.\n"
        "Set use_attention=False and/or use_rnd=False in MADDPG_IA constructor.")


def run_comparison(scenario='small_rural', n_runs=10, device='cuda'):
    """Run algorithm comparison: DQN, QMIX, MAPPO vs MADDPG-IA"""
    raise NotImplementedError(
        "Comparison requires DQN, QMIX, MAPPO implementations in baselines/.\n"
        "See baselines/dqn.py, baselines/qmix.py, baselines/mappo.py templates.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--comparison', action='store_true')
    parser.add_argument('--n_runs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.ablation:
        run_ablation(n_runs=args.n_runs, device=args.device)
    elif args.comparison:
        run_comparison(n_runs=args.n_runs, device=args.device)
    elif args.all:
        run_all(n_runs=args.n_runs, device=args.device)
    elif args.scenario:
        run_all(scenarios=[args.scenario], n_runs=args.n_runs, device=args.device)
    else:
        parser.print_help()
