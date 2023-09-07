import time
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simnibs
import torch
from tqdm import tqdm
from itertools import permutations
from ti_lib import run_tDCS_sim, analyze_TI_from_sims, compute_BE_TI_from_tDCS_sims
import json
import pathos.multiprocessing as mp


# def determine_base_align_mesh(output_folder: Path):
#     """Determin the base mesh and aligne the other meshes to it.


#     Args:
#         output_folder (Path): _description_
#     """
#     pass


def pre_compute_tDSC_sims(subject_folder: Path, output_folder: Path,
                          current: float,
                          cores: int = 6):
    """
    Pre-computes tDCS simulations for all possible electrode pairs.

    Args:
    - subject_folder (Path): Path to the folder containing subject data.
    - output_folder (Path): Path to the folder where the simulation results
        will be saved.
    - current (float): The current to be used in the simulation.
    - cores (int): The number of CPU cores to use for parallel processing.

    Returns:
    - tdcs_summary (dict): A dictionary containing the simulation results for
        all electrode pairs.
    """

    summary_f = output_folder / 'summary.json'

    output_folder.mkdir(exist_ok=True)

    eeg_loc_df = pd.read_csv(
        'data/m2m_ernie/eeg_positions/EEG10-10_UI_Jurak_2007.csv', header=None)
    locations = eeg_loc_df[4].values[:3]

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
    return tdcs_summary


def test():
    # Input and output folders
    subject_folder = Path('data/m2m_ernie')
    output_folder = Path('data/single_sgp_sims/')

    current = 0.001

    cathode_centre = 'P7'
    anode_centre = 'F7'
    output_dir_1 = Path(
        f'data/single_sgp_sims/{cathode_centre}-{anode_centre}-c-{current}')
    cathode_centre = 'Fpz'
    anode_centre = 'Fp1'
    output_dir_2 = Path(
        f'data/single_sgp_sims/{anode_centre}-{cathode_centre}-c-{current}')

    xyz = (0.7879, 21.1037, -22.0770)
    r = 2

    focality, focality_info, elm_centers, ti_magn = analyze_TI_from_sims(
        subject_folder,
        output_dir_1,
        output_dir_2,
        [(xyz[0], xyz[1], xyz[2]),
            (xyz[0] + 5, xyz[1] + 5, xyz[2] + 5),
         ],
        [r, r + 0.1],
    )
    print(focality, focality_info, elm_centers, ti_magn)


def main():
    test()
    pass


if __name__ == '__main__':
    main()
