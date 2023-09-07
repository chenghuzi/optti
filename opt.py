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
from ti_lib import run_tDCS_sim, compute_TI_max_magnitude, analyze_BE_TI_from_tDCS_sims
import json
import pathos.multiprocessing as mp


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

    # t0 = time.time()
    # res_all = mp.Pool(cores).map(run_sim_wrapper, all_args)
    # dt = time.time() - t0
    # tdcs_summary['results'] = res_all
    # print(f'Finished in {dt} seconds. Efficiency: {dt/ len(all_args)}')
    # json.dump(tdcs_summary, open(summary_f, 'w'), indent=2)

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

    output_dir_1 = Path('data/single_sgp_sims/Fp1-Fp2-c-0.001')
    output_dir_2 = Path('data/single_sgp_sims/Fpz-Fp1-c-0.001')

    # # post_process_tDCS_sim(output_dir_1)
    # # post_process_tDCS_sim(output_dir_2)
    # # analyze_BE_TI_from_tDCS_sims_processed(
    # #     subject_folder,
    # #     output_dir_1,
    # #     output_dir_2
    # # )

    analyze_BE_TI_from_tDCS_sims(
        subject_folder,
        output_dir_1,
        output_dir_2
    )


if __name__ == '__main__':
    main()
