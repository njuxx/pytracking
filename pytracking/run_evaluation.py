import os
import sys
import time
import argparse
import functools

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

from pytracking.benchmarks import AccuracyRobustnessBenchmark, EAOBenchmark#, F1Benchmark
from pytracking.evaluation.votdataset import VOTDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_params', nargs='+', help='Names of parameters.')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--tracker_result_root', type=str, help='tracker result root')
    parser.add_argument('--result_dirs', nargs='+')
    parser.add_argument('--vis', dest='vis', action='store_true')
    parser.add_argument('--show_video_level', dest='show_video_level', action='store_true')
    parser.add_argument('--num', type=int, help='number of processes to eval', default=1)
    args = parser.parse_args()

    tracker_name = args.tracker_name
    tracker_params = args.tracker_params
    # root = args.dataset_dir

    assert len(tracker_params) > 0
    args.num = min(args.num, len(tracker_params))

    if 'VOT2018' == args.dataset:
        dataset = VOTDataset()
        ar_benchmark = AccuracyRobustnessBenchmark(dataset, tracker_name, tracker_params)
        ar_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                tracker_params), desc='eval ar', total=len(tracker_params), ncols=100):
                ar_result.update(ret)
        benchmark = EAOBenchmark(dataset, tracker_name, tracker_params)
        eao_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                tracker_params), desc='eval eao', total=len(tracker_params), ncols=100):
                eao_result.update(ret)
        ar_benchmark.show_result(ar_result, eao_result,
                show_video_level=args.show_video_level)
    else:
        print('Not support dataset {}, please input again.'.format(args.dataset))
    # elif 'VOT2018-LT' == args.dataset:
    #     dataset = VOTLTDataset(args.dataset, root)
    #     dataset.set_tracker(tracker_dir, trackers)
    #     benchmark = F1Benchmark(dataset)
    #     f1_result = {}
    #     with Pool(processes=args.num) as pool:
    #         for ret in tqdm(pool.imap_unordered(benchmark.eval,
    #             trackers), desc='eval f1', total=len(trackers), ncols=100):
    #             f1_result.update(ret)
    #     benchmark.show_result(f1_result,
    #             show_video_level=args.show_video_level)
    #     if args.vis:
    #         draw_f1(f1_result)
