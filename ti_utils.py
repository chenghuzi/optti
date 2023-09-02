from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm


def find_msh(dir:Path)->Path:
    mshf = None
    for f in dir.glob("*.msh"):
        print(f'Found msh file: {f}')
        mshf = f
        break
    if mshf is None:
        raise ValueError("No msh file found")

    return mshf


def vector_cos_angles(a:np.ndarray, b:np.ndarray)->np.ndarray:
    dot_product = np.einsum('ij,ij->i', a, b)
    norms_a = np.linalg.norm(a, axis=1)
    norms_b = np.linalg.norm(b, axis=1)
    cos_angles = dot_product / (norms_a * norms_b)
    return cos_angles

def align_mesh_idx(
    elm_centers_base:torch.Tensor,
    elm_centers2:torch.Tensor,
    device:str,
    dist_threshold:float = 1.0,
    block_size:int = 400,
    residual_block_size:int = 20,
    )->Tuple[torch.Tensor, torch.Tensor]:
    """
    Now since the dimensionality of the meshes are different, we need to
    use one of them as a reference and find the closest element in the other
    mesh for each element in the reference mesh.
    Then, for the base mesh, we select two types of regions we are interested in:
    the target region and the non-target region.
    For each region, we find their corresponding elements in the other mesh.
    Then we calculate the mean amplitude in each region.

    Args:
        elm_centers_base (torch.Tensor): _description_
        elm_centers2 (torch.Tensor): _description_
        device (str): _description_

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: _description_
    """

    n_elms = elm_centers_base.shape[0]


    new_indices_2 = torch.zeros(n_elms, dtype=torch.int64).to(device) - 1
    distances = (torch.zeros(n_elms, dtype=torch.int64).to(device) - 1).float()

    for idx in tqdm(range(n_elms // block_size +1)):
        # TODO: we can add some repeated elements in the corner
        s2_left = idx* block_size
        s2_right = (idx+1)* block_size
        if idx > 0:
            s2_left = max(0, int(s2_left - block_size * 9))
        # s2_right = min(n_elms, int(s2_right + block_size * 9))
        s2_right = int(s2_right + block_size * 9)

        block_dist_matrix = torch.cdist(
            elm_centers_base[idx* block_size:(idx+1)* block_size],
            elm_centers2[s2_left:s2_right])

        bdmm = block_dist_matrix.min(dim=1)

        if  bdmm.values.max() > dist_threshold:
            good_indices = torch.where(bdmm.values < dist_threshold)[0]
            if len(good_indices) == 0:
                continue

            gidx = good_indices + int(idx* block_size)
            new_indices_2[gidx] = good_indices + s2_left
            distances[gidx] = bdmm.values[good_indices].float()
        else:
            new_indices_2[idx* block_size:(idx+1)* block_size] = bdmm.indices + s2_left
            distances[idx* block_size:(idx+1)* block_size] = bdmm.values

    # print((idx+1)* block_size, n_elms)
    # import ipdb; ipdb.set_trace() # fmt: off
    bad_ids = torch.where(new_indices_2<0)[0]

    for idx in tqdm(range(len(bad_ids)//residual_block_size+1)):
        s2_left = idx*residual_block_size #bad_ids[idx*sz]
        if (idx+1)*residual_block_size >= len(bad_ids):
            s2_right = len(bad_ids)
        else:
            s2_right = (idx+1)*residual_block_size
        bbids = bad_ids[s2_left:s2_right]
        bdmm = torch.cdist(
            elm_centers_base[bbids],
            elm_centers2
        ).min(dim=1)
        new_indices_2[bbids] = bdmm.indices
        distances[bbids] = bdmm.values.float()

    return new_indices_2, distances
