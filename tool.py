import math
import os
import random
from collections import Counter

import librosa
import numpy
import soundfile
import torch

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.manifold import TSNE
from matplotlib import pyplot


def accuracy(predicts: torch.Tensor, labels: torch.Tensor, top_k: tuple or list = (1, 0)) -> list:
    max_k = max(top_k)
    batch_size = predicts.size(0)

    _, index = torch.topk(predicts, max_k, 1, True, True)
    index = index.t()
    correct = index.eq(labels.view(1, -1).expand_as(index))
    result = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        result.append(correct_k.mul_(100.0 / batch_size))
    return result


def solve_eer(scores: numpy.ndarray, labels: numpy.ndarray) -> tuple:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    ret = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(ret)
    return ret * 100, thresh


def solve_min_dcf(scores: numpy.ndarray, labels: numpy.ndarray, c_miss: float = 1, c_false_alarm: float = 1,
                  p_target: float = 0.01) -> tuple:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    c_det = c_miss * p_target * fnr + c_false_alarm * (1 - p_target) * fpr
    min_dcf = numpy.min(c_det)
    thresh = thresholds[numpy.argmin(c_det)]
    return min_dcf * 100, thresh


class To:
    # Brainless ans safe "to" functions

    @staticmethod
    def __check_iter__(x):
        return type(x) is list or type(x) is tuple

    @staticmethod
    def __gpu__(tensor: torch.Tensor or torch.Module,
                gpu: int = 0,
                catch_except: bool = False) -> torch.Tensor:
        if catch_except:
            try:
                tensor = tensor.to(gpu)
            except Exception as e:
                print(e)
            return tensor
        else:
            return tensor.to(gpu)

    @staticmethod
    def gpu(tensors: torch.Tensor or torch.Module or list or tuple,
            device: int = 0,
            catch_except: bool = False) -> list or torch.Tensor:
        if torch.cuda.is_available():
            if To.__check_iter__(tensors):
                return [To.__gpu__(i, device, catch_except) for i in tensors]
            else:
                return To.__gpu__(tensors, device, catch_except)
        else:
            return tensors

    @staticmethod
    def __detach__(tensor: torch.Tensor or torch.Module,
                   catch_except: bool = False) -> torch.Tensor or torch.Module:
        if torch.is_tensor(tensor):
            if catch_except:
                try:
                    tensor = tensor.detach()
                except Exception as e:
                    print(e)
                return tensor
            else:
                return tensor.detach()
        else:
            return tensor

    @staticmethod
    def detach(tensors: torch.Tensor or torch.Module or tuple or list,
               catch_except: bool = False) -> list or torch.Tensor or torch.Module:
        if To.__check_iter__(tensors):
            return [To.__detach__(i, catch_except) for i in tensors]
        else:
            return To.__detach__(tensors, catch_except)

    @staticmethod
    def __cpu__(tensor: torch.Tensor or torch.Module,
                catch_except: bool = False) -> torch.Tensor or torch.Module:
        if torch.is_tensor(tensor):
            if catch_except:
                try:
                    tensor = tensor.cpu()
                except Exception as e:
                    print(e)
                return tensor
            else:
                return tensor.cpu()
        else:
            return tensor

    @staticmethod
    def cpu(tensors: torch.Tensor or torch.Module or tuple or list,
            catch_except: bool = False) -> list or torch.Tensor or torch.Module:
        if type(tensors) is tuple or type(tensors) is list:
            return [To.__cpu__(i, catch_except) for i in tensors]
        else:
            return To.__cpu__(tensors, catch_except)

    @staticmethod
    def __array__(tensor: torch.Tensor or tuple or list,
                  new_dim: bool = False,
                  catch_except: bool = False) -> numpy.ndarray:
        if To.__check_iter__(tensor):
            tensor = [To.__array__(i, False, catch_except) for i in tensor]
            return numpy.stack(tensor) if new_dim else numpy.concatenate(tensor)
        elif torch.is_tensor(tensor):
            if catch_except:
                try:
                    tensor = To.__cpu__(To.__detach__(tensor)).numpy()
                except Exception as e:
                    print(e)
                return tensor
            else:
                return To.__cpu__(To.__detach__(tensor)).numpy()
        else:
            return tensor

    @staticmethod
    def array(tensors: torch.Tensor or numpy.ndarray or tuple or list,
              package: bool = False,
              new_dim: bool = False,
              catch_except: bool = False) -> list or numpy.ndarray:
        if type(tensors) is numpy.ndarray:
            return tensors
        if package is False:
            if To.__check_iter__(tensors):
                return [To.__array__(i, new_dim, catch_except) for i in tensors]
            else:
                return To.__array__(tensors, new_dim, catch_except)
        else:
            return To.__array__(tensors, new_dim, catch_except)

    @staticmethod
    def __tensor__(array: numpy.ndarray or torch.Tensor or tuple or list,
                   new_dim: bool = False,
                   catch_except: bool = False) -> torch.Tensor:
        if To.__check_iter__(array):
            array = [To.__tensor__(i, False, catch_except) for i in array]
            return torch.stack(array) if new_dim else torch.cat(array)
        elif type(array) is numpy.ndarray:
            if catch_except:
                try:
                    array = torch.FloatTensor(array)
                except Exception as e:
                    print(e)
                return array
            else:
                return torch.FloatTensor(array)
        else:
            return To.__detach__(array, catch_except)

    @staticmethod
    def tensor(arrays: numpy.ndarray or torch.Tensor or tuple or list,
               package: bool = False,
               new_dim: bool = False,
               catch_except: bool = False) -> list or torch.Tensor:
        if package is False:
            if To.__check_iter__(arrays):
                return [To.__tensor__(i, new_dim, catch_except) for i in arrays]
            else:
                return To.__tensor__(arrays, new_dim, catch_except)
        else:
            return To.__tensor__(arrays, new_dim, catch_except)


to = To()


def draw(data: numpy.array, labels: list or tuple, class_number, save_path=None):
    tsne = TSNE(n_components=2, n_iter=3000, perplexity=10.0, init='pca', square_distances=True)
    result = tsne.fit_transform(data)

    marker = ['^', '2', '3', 'x', 'p', 's', 'h', 'd', 'D', '*']
    len_per_color = math.ceil(len(labels) / len(marker))

    color = set()
    while len(color) < len_per_color:
        color.add((numpy.random.random(), numpy.random.random(), numpy.random.random()))
    color = list(color)
    markers = []
    colors = []
    for i in marker:
        for j in color:
            markers.append(i)
            colors.append(j)
    colors = colors[:len(labels)]
    markers = markers[:len(labels)]
    pyplot.scatter(result[:, 0], result[:, 1], c=colors, marker=markers)
    if save_path:
        pyplot.savefig(save_path)
    pyplot.show()


def load_wave(path: str, file_name: str = None, sample_rate: int = None) -> numpy.array:
    if file_name is None:
        file_path = path
    else:
        file_path = os.path.join(path, file_name)
    if sample_rate is None:
        wave, sample_rate = soundfile.read(file_path)
    else:
        wave, sample_rate = librosa.load(file_path, sr=sample_rate)
    return wave


def same_length(wave: numpy.array, length: int, sample_num: int = 1) -> numpy.array:
    if wave.shape[0] < length:
        wave = numpy.pad(wave, (0, length - wave.shape[0] + 1), 'wrap')
    if sample_num != 1:
        starts = numpy.linspace(0, wave.shape[0] - length, num=sample_num)
        return numpy.stack([wave[int(i):int(i) + length] for i in starts], axis=0).astype(numpy.float)
    else:
        start = random.randint(0, wave.shape[0] - length)
        return wave[start:start + length].astype(numpy.float)


def entropy(prob1: torch.Tensor, prob2: torch.Tensor) -> torch.Tensor:
    return -torch.sum(prob1 * torch.log(prob2 + 1e-6), dim=-1)


def setup_seed(seed=7):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def as_norm(enroll_test: numpy.array, enroll_cohort: numpy.array, test_cohort: numpy.array, top_k: int = 400,
            cross_select: bool = True, gpu: int = None, array: bool = True) -> numpy.array or torch.tensor:
    '''
    :param enroll_test: shape(E, 1)
    :param enroll_cohort: shape(E, T)
    :param test_cohort: shape(E, T)
    :param top_k: 400
    :param cross_select: True
    :param gpu: None
    :param array: True
    :return: shape(E, T)
    '''
    enroll_test, enroll_cohort, test_cohort = to.tensor([enroll_test, enroll_cohort, test_cohort])
    if gpu is not None:
        enroll_test, enroll_cohort, test_cohort = to.gpu([enroll_test, enroll_cohort, test_cohort], gpu)
    enroll_cohort_select, enroll_cohort_idx = torch.topk(enroll_cohort, top_k)[1]
    test_cohort_select, test_cohort_idx = torch.topk(test_cohort, top_k)[1]

    if cross_select:
        enroll_cohort_idx, test_cohort_idx = test_cohort_idx, enroll_cohort_idx
        enroll_cohort_mask = torch.scatter(torch.zeros_like(enroll_cohort), 1, enroll_cohort_idx, 1).bool()
        test_cohort_mask = torch.scatter(torch.zeros_like(test_cohort), 1, test_cohort_idx, 1).bool()
        # (E, K)
        enroll_cohort_select = torch.masked_select(enroll_cohort, enroll_cohort_mask).view(enroll_cohort.size(0), -1)
        # (T, K)
        test_cohort_select = torch.masked_select(test_cohort, test_cohort_mask).view(test_cohort.size(0), -1)
    # (E, 1)
    enroll_cohort_std, enroll_cohort_mean = torch.std_mean(enroll_cohort_select, dim=-1, keepdim=True)
    # (T, 1)
    test_cohort_std, test_cohort_mean = torch.std_mean(test_cohort_select, dim=-1, keepdim=True)
    enroll_test = 0.5 * ((enroll_test - enroll_cohort_mean) / enroll_cohort_std + (
            enroll_test - test_cohort_mean) / test_cohort_std)
    return to.array(enroll_test) if array else enroll_test
