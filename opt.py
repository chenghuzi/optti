import time
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simnibs
import torch
from tqdm import tqdm
from itertools import combinations, permutations
from ti_lib import run_tDCS_sim
from ti_utils import align_mesh_idx, find_msh
import json
import pathos.multiprocessing as mp


def compute_TI_amplitude(
        vecE_base: torch.Tensor,
        vecE_2: torch.Tensor,
        magnE_base: torch.Tensor,
        magnE_2: torch.Tensor) -> torch.Tensor:
    cosine_alpha = (vecE_base * vecE_2).sum(dim=1) / (
        vecE_base.norm(dim=1) * vecE_2.norm(dim=1)
    )
    mask = magnE_base >= magnE_2
    # flip according to the condition |E1| > |E2|
    tmp_magnE = magnE_base[torch.logical_not(mask)]
    magnE_base[torch.logical_not(mask)] = magnE_2[torch.logical_not(mask)]
    magnE_2[torch.logical_not(mask)] = tmp_magnE

    tmp_E = vecE_base[torch.logical_not(mask)]
    vecE_base[torch.logical_not(mask)] = vecE_2[torch.logical_not(mask)]
    vecE_2[torch.logical_not(mask)] = tmp_E

    vec_diffE = vecE_base - vecE_2
    # TODO is here the cross product
    tmp_amp = torch.cross(vecE_2, vec_diffE, dim=1).norm(
        dim=1) / vecE_2.norm(dim=1)

    angle_mask = (magnE_base * cosine_alpha > magnE_2).float()
    amplitude = 2 * (angle_mask * magnE_2 + (1 - angle_mask) * tmp_amp)
    return amplitude


def analyze_BE_TI_from_tDCS_sims_processed(subject_dir: Path,
                                           output_dir_1: Path,
                                           output_dir_2: Path,
                                           ):
    """Try to extract the bi-electrode  TI effect using two tDCS simulations.
    NOTICE: this is not as good as we expected.

    Args:
        output_dir (Path): _description_
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size = 1
    E_f1_info = torch.load(output_dir_1 / f'vecE-size-{size}.pt')
    E_f2_info = torch.load(output_dir_2 / f'vecE-size-{size}.pt')

    elm_centers_base = E_f1_info['elm_centers_int'].float().to(device)
    vecE_base = E_f1_info['vecE'].to(device)
    magnE_base = E_f1_info['magnE'].to(device)

    elm_centers2 = E_f2_info['elm_centers_int'].float().to(device)
    vecE_2 = E_f2_info['vecE'].to(device)
    magnE_2 = E_f2_info['magnE'].to(device)

    if not elm_centers_base.shape[0] < elm_centers2.shape[0]:
        print('swapping meshes')
        elm_centers_base, elm_centers2 = elm_centers2, elm_centers_base
        vecE_base, vecE_2 = vecE_2, vecE_base
        magnE_base, magnE_2 = magnE_2, magnE_base

    new_indices_2, distances = align_mesh_idx(elm_centers_base, elm_centers2,
                                              device,
                                              block_size=20,
                                              dist_threshold=2
                                              )
    plt.hist(distances.data.cpu().numpy(), bins=100)
    plt.savefig('distances.png')

    elm_centers2 = elm_centers2[new_indices_2]

    amplitude = compute_TI_amplitude(vecE_base, vecE_2, magnE_base, magnE_2)
    return amplitude


def analyze_BE_TI_from_tDCS_sims(subject_dir: Path,
                                 output_dir_1: Path,
                                 output_dir_2: Path,
                                 just_gray_matter: bool = False,
                                 ):
    """Try to extract the bi-electrode  TI effect using two tDCS simulations.

    Args:
        output_dir (Path): _description_
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    head_mesh_base = simnibs.read_msh(find_msh(output_dir_1))
    if just_gray_matter:
        head_mesh_base = head_mesh_base.crop_mesh(2)
    elm_centers_base = torch.from_numpy(
        head_mesh_base.elements_baricenters()[:]).to(device)

    head_mesh_2 = simnibs.read_msh(find_msh(output_dir_2))
    if just_gray_matter:
        head_mesh_2 = head_mesh_2.crop_mesh(2)

    elm_centers2 = torch.from_numpy(
        head_mesh_2.elements_baricenters()[:]).to(device)

    if not elm_centers_base.shape[0] < elm_centers2.shape[0]:
        print('swapping meshes')
        head_mesh_base, head_mesh_2 = head_mesh_2, head_mesh_base
        elm_centers_base, elm_centers2 = elm_centers2, elm_centers_base

    new_indices_2, distances = align_mesh_idx(
        elm_centers_base, elm_centers2, device)

    plt.hist(distances.data.cpu().numpy(), bins=100)
    plt.savefig('distances.png')
    elm_centers2 = elm_centers2[new_indices_2]

    magnE_base = torch.from_numpy(head_mesh_base.field['magnE'][:]).to(device)
    vecE_base = torch.from_numpy(head_mesh_base.field['E'][:]).to(device)

    magnE_2 = torch.from_numpy(head_mesh_2.field['magnE'][:]).to(device)[
        new_indices_2]
    vecE_2 = torch.from_numpy(head_mesh_2.field['E'][:]).to(device)[
        new_indices_2]

    # Now let's calculate the amplitude of the whole mesh.
    amplitude = compute_TI_amplitude(vecE_base, vecE_2, magnE_base, magnE_2)

    return amplitude


def determine_base_align_mesh(output_folder: Path):
    """Determin the base mesh and aligne the other meshes to it.


    Args:
        output_folder (Path): _description_
    """
    pass


def main():
    cores = 6
    # Input and output folders
    subject_folder = Path('data/m2m_ernie')
    output_folder = Path('data/single_sgp_sims/')
    summary_f = output_folder / 'summary.json'

    output_folder.mkdir(exist_ok=True)

    eeg_loc_df = pd.read_csv(
        'data/m2m_ernie/eeg_positions/EEG10-10_UI_Jurak_2007.csv', header=None)
    locations = eeg_loc_df[4].values[:3]

    # current = 1mA
    current = 0.001
    tdcs_summary = {'results': []}
    electrode_pairs = list(permutations(locations, 2))
    all_args = [
        (subject_folder,
         output_folder,
         cathode_centre,
         anode_centre,
         current)
        for (cathode_centre, anode_centre) in electrode_pairs]

    def run_sim_wrapper(args):
        cathode_centre = args[2]
        anode_centre = args[3]
        res_info = run_tDCS_sim(*args)

        return {
            'cathode': cathode_centre,
            'anode': anode_centre,
            'res_info': res_info,
        }

    t0 = time.time()
    res_all = mp.Pool(cores).map(run_sim_wrapper, all_args)
    dt = time.time() - t0
    tdcs_summary['results'] = res_all
    print(f'Finished in {dt} seconds. Efficiency: {dt/ len(all_args)}')
    json.dump(tdcs_summary, open(summary_f, 'w'), indent=2)

    # for (cathode_centre, anode_centre) in electrode_pairs:
    #     res_dir = run_tDCS_sim(
    #         subject_folder,
    #         output_folder,
    #         cathode_centre,
    #         anode_centre,
    #         current,
    #     )
    #     tdcs_summary['results'].append([
    #         cathode_centre,
    #         anode_centre,
    #         str(res_dir)d,
    #     ])
    #     json.dump(tdcs_summary, open(summary_f, 'w'), indent=2)


    # import ipdb; ipdb.set_trace() # fmt: off
    # cathode, anode
    # run_tDCS_sim(
    #     subject_folder,
    #     output_folder,
    #     cathode_centre,
    #     anode_centre,
    #     current,
    #     )

    # cathode_centre = 'P7'
    # anode_centre = 'F7'
    # run_tDCS_sim(
    #     subject_folder,
    #     output_folder,
    #     cathode_centre,
    #     anode_centre,
    #     current,
    #     )
    # analyze_tDCS_sim(subject_folder, output_folder)

    # output_dir_1 = Path('data/single_sgp_sims/P8-F8-c-0.001')
    # output_dir_2 = Path('data/single_sgp_sims/P7-F7-c-0.001')

    # # post_process_tDCS_sim(output_dir_1)
    # # post_process_tDCS_sim(output_dir_2)
    # # analyze_BE_TI_from_tDCS_sims_processed(
    # #     subject_folder,
    # #     output_dir_1,
    # #     output_dir_2
    # # )
    # analyze_BE_TI_from_tDCS_sims(
    #     subject_folder,
    #     output_dir_1,
    #     output_dir_2
    # )


if __name__ == '__main__':
    main()
