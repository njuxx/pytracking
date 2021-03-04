"""
    @author
"""

import warnings
import itertools
import numpy as np
import os
from glob import glob

from colorama import Style, Fore
from pytracking.utils.vot_utils import calculate_failures, calculate_accuracy

class AccuracyRobustnessBenchmark:
    """
    Args:
        dataset:
        burnin:
    """
    def __init__(self, dataset, tracker_name, tracker_params, burnin=10):
        self.dataset = dataset
        self.burnin = burnin
        self.tracker_name = tracker_name
        self.tracker_params = tracker_params

    def eval(self, eval_trackers=None):
        """
        Args:
            eval_tags: list of tag
            eval_trackers: list of tracker name
        Returns:
            ret: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.tracker_params
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        result = {}
        for tracker_name in eval_trackers:
            accuracy, failures = self._calculate_accuracy_robustness(tracker_name)
            result[tracker_name] = {'overlaps': accuracy,
                                    'failures': failures}
        return result

    def show_result(self, result, eao_result=None, show_video_level=False, helight_threshold=0.5):
        """pretty print result
        Args:
            result: returned dict from function eval
        """
        tracker_name_len = max((max([len(x) for x in result.keys()])+2), 12)
        if eao_result is not None:
            header = "|{:^"+str(tracker_name_len)+"}|{:^10}|{:^12}|{:^13}|{:^7}|"
            header = header.format('Tracker Name',
                    'Accuracy', 'Robustness', 'Lost Number', 'EAO')
            formatter = "|{:^"+str(tracker_name_len)+"}|{:^10.3f}|{:^12.3f}|{:^13.1f}|{:^7.3f}|"
        else:
            header = "|{:^"+str(tracker_name_len)+"}|{:^10}|{:^12}|{:^13}|"
            header = header.format('Tracker Name',
                    'Accuracy', 'Robustness', 'Lost Number')
            formatter = "|{:^"+str(tracker_name_len)+"}|{:^10.3f}|{:^12.3f}|{:^13.1f}|"
        bar = '-'*len(header)
        print(bar)
        print(header)
        print(bar)
        if eao_result is not None:
            tracker_eao = sorted(eao_result.items(),
                                 key=lambda x:x[1]['all'],
                                 reverse=True)[:20]
            tracker_names = [x[0] for x in tracker_eao]
        else:
            tracker_names = list(result.keys())
        for tracker_name in tracker_names:
        # for tracker_name, ret in result.items():
            ret = result[tracker_name]
            overlaps = list(itertools.chain(*ret['overlaps'].values()))
            accuracy = np.nanmean(overlaps)
            length = sum([len(x) for x in ret['overlaps'].values()])
            failures = list(ret['failures'].values())
            lost_number = np.mean(np.sum(failures, axis=0))
            robustness = np.mean(np.sum(np.array(failures), axis=0) / length) * 100
            if eao_result is None:
                print(formatter.format(tracker_name, accuracy, robustness, lost_number))
            else:
                print(formatter.format(tracker_name, accuracy, robustness, lost_number, eao_result[tracker_name]['all']))
        print(bar)

        if show_video_level and len(result) < 10:
            print('\n\n')
            header1 = "|{:^14}|".format("Tracker name")
            header2 = "|{:^14}|".format("Video name")
            for tracker_name in result.keys():
                header1 += ("{:^17}|").format(tracker_name)
                header2 += "{:^8}|{:^8}|".format("Acc", "LN")
            print('-'*len(header1))
            print(header1)
            print('-'*len(header1))
            print(header2)
            print('-'*len(header1))
            videos = list(result[tracker_name]['overlaps'].keys())
            for video in videos:
                row = "|{:^14}|".format(video)
                for tracker_name in result.keys():
                    overlaps = result[tracker_name]['overlaps'][video]
                    accuracy = np.nanmean(overlaps)
                    failures = result[tracker_name]['failures'][video]
                    lost_number = np.mean(failures)

                    accuracy_str = "{:^8.3f}".format(accuracy)
                    if accuracy < helight_threshold:
                        row += f'{Fore.RED}{accuracy_str}{Style.RESET_ALL}|'
                    else:
                        row += accuracy_str+'|'
                    lost_num_str = "{:^8.3f}".format(lost_number)
                    if lost_number > 0:
                        row += f'{Fore.RED}{lost_num_str}{Style.RESET_ALL}|'
                    else:
                        row += lost_num_str+'|'
                print(row)
            print('-'*len(header1))

    def _calculate_accuracy_robustness(self, tracker_name):
        overlaps = {}
        failures = {}
        all_length = {}
        dataset = self.dataset.get_sequence_list()
        for i in range(len(dataset)):
            video = dataset[i]
            gt_traj = video.ground_truth_rect
            tracker_trajs = load_tracker(self.tracker_name, tracker_name, video.name, False)
            overlaps_group = []
            num_failures_group = []
            for tracker_traj in tracker_trajs:
                num_failures = calculate_failures(tracker_traj)[0]
                overlaps_ = calculate_accuracy(tracker_traj, gt_traj,
                        burnin=10, bound=(video.width, video.height))[1]
                overlaps_group.append(overlaps_)
                num_failures_group.append(num_failures)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                overlaps[video.name] = np.nanmean(overlaps_group, axis=0).tolist()
                failures[video.name] = num_failures_group
        return overlaps, failures

def load_tracker(tracker_name, param_names, video_name, store=False):
    results_path = '/disk/xuxiang/pytracking/pytracking/tracking_results/'# env_settings.results_path
    results_path = os.path.join(results_path, tracker_name)

    if isinstance(param_names, str):
        param_names = [param_names]
    for name in param_names:
        pred_files = glob(os.path.join(results_path, name, video_name, video_name+'.txt'))
        pred_file = pred_files[0]
        # if len(pred_files) == 15:
        #     pred_files = pred_files
        # else:
        #     pred_files = pred_files[0:1]
        pred_traj = []
        # for traj_file in pred_files:
        with open(pred_file, 'r') as f:
            traj = [list(map(float, x.strip().split(',')))
                    for x in f.readlines()]
            pred_traj.append(traj)
        if store:
            self.pred_trajs[name] = pred_traj
        else:
            return pred_traj