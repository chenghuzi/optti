import json
import multiprocessing as omp
from itertools import combinations, permutations
from pathlib import Path
import torch
from typing import Tuple

# import pathos.multiprocessing as mp

from .ti_lib import run_tDCS_sim
from .ti_utils import read_eeg_locations, align_mesh_idx
import time


def run_sim_worker(a_args):
    pid = omp.current_process().pid
    args, mp_args = a_args

    cathode_centre = args[2]
    anode_centre = args[3]
    finished_count, fc_lock, total_count = mp_args

    res_info = run_tDCS_sim(*args)
    with fc_lock:
        finished_count.value += 1
        print(f'({pid}) Finished {finished_count.value} of {total_count} \
simulations. Using {cathode_centre}-{anode_centre}.')

    res_info = "good"
    return {
        'cathode': cathode_centre,
        'anode': anode_centre,
        'res_info': res_info,
    }


def run_all_sims(model_dir,
                 pre_calculation_dir,
                 current,
                 electrode_shape,
                 electrode_dimensions,
                 electrode_thickness,
                 just_gray_matter,
                 eeg_electrode_pairs,
                 cores):
    with omp.Manager() as manager:
        fc_lock = manager.Lock()
        # value = manager.Value(float, 0.0)
        print(f"{cores} cores will be used.")
        finished_count = manager.Value('i', 0)
        total_count = len(eeg_electrode_pairs)
        all_args = [
            (
                (model_dir,
                 pre_calculation_dir,
                 cathode_centre,
                 anode_centre,
                 current,
                 electrode_shape,
                 electrode_dimensions,
                 electrode_thickness,
                 1,
                 1,
                 False,
                 just_gray_matter,
                 ),
                (finished_count,
                 fc_lock,
                 total_count
                 )
            )
            for (cathode_centre, anode_centre) in eeg_electrode_pairs]
        sims_res = []
        if cores == 1:

            for all_arg in all_args[:1]:
                sims_res.append(run_sim_worker(all_arg))
        else:
            sims_res = omp.Pool(cores).map(run_sim_worker, all_args)

        return sims_res


class OptTI:

    def __init__(self, model_dir: str,
                 pre_calculation_dir: str,
                 eeg_coord_sys: str = '10-10',
                 electrode_shape: str = 'ellipse',
                 electrode_dimensions: Tuple = (10, 10),
                 electrode_thickness=5,
                 current: float = 0.01,
                 just_gray_matter: bool = False,
                 device: str = 'cpu'
                 ) -> None:
        """

        Args:
            model_dir (str): Path to the directory containing the
                subject data like head mesh.
            pre_calculation_dir (str): Path to the directory where
                pre-calculated data will be stored.
            eeg_coord_sys (str, optional): The coordinate system used for
                EEG electrodes. Defaults to '10-10'.
            electrode_shape (str, optional): The shape of the electrodes.
                Defaults to 'ellipse'.
            electrode_dimensions (Tuple, optional): The dimensions of the
                electrodes. Defaults to (10, 10). Unit is mm.
            electrode_thickness (int, optional): The thickness of
                the electrodes. Defaults to 5.
            just_gray_matter (bool, optional): Whether to only consider
                gray matter. Defaults to False.
        """
        if device == 'cuda':
            assert torch.cuda.is_available()
        self.device = device

        self.model_dir = Path(model_dir)

        self.pre_calculation_dir = Path(pre_calculation_dir)
        self.pre_calculation_dir.mkdir(exist_ok=True, parents=True)

        if eeg_coord_sys == '10-10':
            self.eeg_coord_info = read_eeg_locations()
        else:
            raise NotImplementedError(
                'Only 10-10 is supported at the moment.')

        self.electrodes = list(self.eeg_coord_info.keys())
        self.electrode_ref = self.electrodes[0]
        self.electrode_base_pairs = [
            (self.electrode_ref, e) for e in self.electrodes[1:]]
        self.electrode_pairs = list(
            permutations(self.electrodes, 2))

        self.electrode_shape = electrode_shape
        self.electrode_dimensions = electrode_dimensions
        self.electrode_thickness = electrode_thickness
        self.current = current
        self.just_gray_matter = just_gray_matter

    def pre_calculate(self, cores: int = -1):
        """Calculate all necessary data for the optimization.
        """
        self.summary_f = self.pre_calculation_dir / \
            f'summary-{self.electrode_shape}-{self.electrode_dimensions}\
-{self.electrode_thickness}.json'
        print(self.electrode_base_pairs)
        print(len(self.electrode_base_pairs))
        pass
        if cores == -1:
            # by default use all physical cores
            cores = omp.cpu_count()

        sims_res = run_all_sims(self.model_dir, self.pre_calculation_dir,
                                self.current,
                                self.electrode_shape,
                                self.electrode_dimensions,
                                self.electrode_thickness,
                                self.just_gray_matter,
                                self.electrode_base_pairs,
                                cores)
        print(self.electrode_pairs)
        # json.dump({
        #     'sims_res': sims_res
        # }, open(self.summary_f, 'w'), indent=2)

    def prune_storage(self) -> None:
        """"""
        pass

    def _read_sim_directly(self, e1: str, e2: str) -> torch.Tensor:
        res_dir = self.pre_calculation_dir / \
            (f'{e1}-{e2}-c-{self.current}-{self.electrode_shape}' +
             f'-{self.electrode_dimensions}-{self.electrode_thickness}')

        print(res_dir)
        assert res_dir.is_dir()
        res_tensors = res_dir / 'vecE-size-1.pt'
        assert res_tensors.is_file()
        res = torch.load(res_tensors)
        return res

    def read_sim(self, e1: str, e2: str):
        print(e1, e2)
        assert e1 in self.electrodes and e2 in self.electrodes
        if e1 == self.electrode_ref:
            # we can directly get the result
            res = self._read_sim_directly(e1, e2)
            res['elm_centers'] = res['elm_centers'].to(self.device)
            res['vecE'] = res['vecE'].to(self.device)
            res['magnE'] = res['magnE'].to(self.device)
            return res

        elif e2 == self.electrode_ref:
            # we can get (e2, e1) result and then invert it
            # with E((e1, e2)) = -E((e2, e1))
            res = self._read_sim_directly(e2, e1)
            res['vecE'] = -res['vecE']  # Other parts are the same except vecE.

            res['elm_centers'] = res['elm_centers'].to(self.device)
            res['vecE'] = res['vecE'].to(self.device)
            res['magnE'] = res['magnE'].to(self.device)
            return res
        else:
            # we can compute (e_ref, e1) and (e_ref, e2) and then
            # get E(e1, e2) by E((e_ref, e2)) - E((e_ref, e1))
            e_ref = self.electrode_ref
            ep_base_1 = (e_ref, e1)
            ep_base_2 = (e_ref, e2)
            res_ep1 = self.read_sim(ep_base_1[0], ep_base_1[1])
            res_ep2 = self.read_sim(ep_base_2[0], ep_base_2[1])
            # if len(res_ep1['elm_centers']) > len(res_ep2['elm_centers']):
            new_indices_ep2, distances_ep2 = align_mesh_idx(
                res_ep1['elm_centers'], res_ep2['elm_centers'],
                self.device,
                block_size=20,
                dist_threshold=2,
                verbose=True,
            )
            res_ep1['elm_centers'] = res_ep1['elm_centers'].to(self.device)
            res_ep1['vecE'] = res_ep1['vecE'].to(self.device)
            res_ep1['magnE'] = res_ep1['magnE'].to(self.device)

            res_ep2['elm_centers'] = res_ep2['elm_centers'].to(self.device)[
                new_indices_ep2]
            res_ep2['vecE'] = res_ep2['vecE'].to(self.device)[new_indices_ep2]
            res_ep2['magnE'] = res_ep2['magnE'].to(self.device)[
                new_indices_ep2]

            vecE_diff = res_ep2['vecE'] - res_ep1['vecE']
            vecE_diff_norm = torch.norm(vecE_diff, dim=1)
            return {
                'elm_centers': res_ep1['elm_centers'],
                'vecE': vecE_diff,
                'magnE': vecE_diff_norm,
            }

    def run(self):
        pass


def test_optti():
    opt = OptTI('data/m2m_ernie',
                'data/optti/sims')
    opt.pre_calculate(cores=8)
    pass


if __name__ == '__main__':
    test_optti()
    pass
