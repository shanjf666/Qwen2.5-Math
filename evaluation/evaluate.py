import argparse
import math
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor


def pass_at_k(n, c, k):
    """
    Calculate pass@k metric using the combinatorial formula.
    
    Pass@k = 1 - C(n-c, k) / C(n, k)
    
    This represents the probability that at least one of the k selected
    samples (from n total) is correct, given c correct samples.
    
    Args:
        n: total number of samples per problem
        c: number of correct samples for this problem
        k: number of samples to select
    Returns:
        float: pass@k probability for this problem
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_at_k_metrics(score_mat, n_sampling):
    """
    Compute pass@k for all valid k values (powers of 2 up to n_sampling,
    plus n_sampling itself).
    
    Args:
        score_mat: list of lists, each inner list contains boolean scores
                   for one problem across n_sampling attempts.
        n_sampling: number of samples per problem.
    Returns:
        dict: mapping from k to pass@k percentage (0-100).
    """
    # Determine which k values to compute
    k_values = []
    k = 1
    while k <= n_sampling:
        k_values.append(k)
        k *= 2
    # Always include the actual n_sampling if not already a power of 2
    if n_sampling not in k_values:
        k_values.append(n_sampling)
    k_values = sorted(set(k_values))

    pass_k_results = {}
    for k in k_values:
        per_problem = []
        for scores in score_mat:
            n = len(scores)
            c = sum(scores)
            per_problem.append(pass_at_k(n, c, k))
        pass_k_results[k] = float(np.round(np.mean(per_problem) * 100, decimals=1))
    return pass_k_results


def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 

    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    n_sampling = max_len  # number of samples per problem

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "n_sampling": n_sampling,
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": mean_score[0]
    }

    # Compute pass@k metrics
    pass_k_metrics = compute_pass_at_k_metrics(score_mat, n_sampling)
    result_json["pass@k"] = pass_k_metrics
    # Print pass@k results clearly
    pass_k_str = ", ".join([f"pass@{k}: {v:.1f}" for k, v in pass_k_metrics.items()])
    print(f"[Pass@k] {pass_k_str}")

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        type_score_mat = {}  # for pass@k per type
        for i, sample in enumerate(samples):
            t = sample['type']
            if t not in type_scores:
                type_scores[t] = []
                type_score_mat[t] = []
            type_scores[t].append(sample['score'][-1])
            type_score_mat[t].append(score_mat[i])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

        # pass@k per type
        if n_sampling > 1:
            type_pass_k = {}
            for t, mat in type_score_mat.items():
                type_pass_k[t] = compute_pass_at_k_metrics(mat, n_sampling)
            type_pass_k = {k: v for k, v in sorted(type_pass_k.items(), key=lambda item: item[0])}
            result_json['type_pass@k'] = type_pass_k

    print(result_json)
    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
