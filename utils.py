import os
import torch
from torch import nn
import torch.distributed as dist
import timm.models.hub as timm_hub


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
    If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.
    """

    def get_cached_file_path():
        # a hack to sync the file path across processes
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

        return cached_file

    if is_main_process():
        timm_hub.download_cached_file(url, check_hash, progress)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return get_cached_file_path()


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(torch.float16)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(torch.float16)

    model.apply(_convert_weights_to_fp16)


def convert_weights_to_bf16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_bf16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(torch.bfloat16)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(torch.bfloat16)

    model.apply(_convert_weights_to_bf16)


def save_result(result, result_dir, filename, remove_duplicate=""):
    import json

    print("Dump result")

    # Make the temp dir for saving results
    if not os.path.exists(result_dir):
        if is_main_process():
            os.makedirs(result_dir)
        if is_dist_avail_and_initialized():
            torch.distributed.barrier()

    result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, get_rank()))
    final_result_file = os.path.join(result_dir, "%s.json" % filename)

    json.dump(result, open(result_file, "w"))

    if is_dist_avail_and_initialized():
        torch.distributed.barrier()

    if is_main_process():
        print("rank %d starts merging results." % get_rank())
        # combine results from all processes
        result = []

        for rank in range(get_world_size()):
            result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, rank))
            res = json.load(open(result_file, "r"))
            result += res

        print("Remove duplicate")
        if remove_duplicate:
            result_new = []
            id_set = set()
            for res in result:
                if res[remove_duplicate] not in id_set:
                    id_set.add(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        json.dump(result, open(final_result_file, "w"))
        print("result file saved to %s" % final_result_file)

    return final_result_file
