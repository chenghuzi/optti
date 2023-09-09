import copy
import multiprocessing as omp
import random
import time
from itertools import permutations
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import simnibs
import torch
from simnibs import transformations
from tqdm import tqdm

# import pathos.multiprocessing as mp
from .ti_lib import (
    compute_TI_focality,
    compute_TI_max_magnitude,
    res2mask_and_vol,
    run_tDCS_sim,
)
from .ti_utils import align_mesh_idx, find_msh, read_eeg_locations, tags_needed


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
                 electrode_shape: str = 'rect',
                 electrode_dimensions: Tuple = (10, 10),
                 electrode_thickness=5,
                 current: float = 0.001,
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
            current (float, optional): The current used in the simulation.
                Defaults to 0.001, i.e., 1 mA.
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

        assert electrode_shape in ['rect', 'ellipse']
        self.electrode_shape = electrode_shape
        self.electrode_dimensions = electrode_dimensions
        self.electrode_thickness = electrode_thickness
        self.current = current
        self.just_gray_matter = just_gray_matter

        self._focality_cache = dict()

    def pre_calculate(self, cores: int = -1):
        """Calculate all necessary data for the optimization.
        """
        self.summary_f = self.pre_calculation_dir / \
            f'summary-{self.electrode_shape}-{self.electrode_dimensions}\
-{self.electrode_thickness}.json'
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

    @staticmethod
    def sim_dir_fn_gen(pre_cal_dir: Path, e1, e2, c, e_shape,
                       e_dim, e_thc) -> Path:
        res_dir = pre_cal_dir / f'{e1}-{e2}-c-{c}-{e_shape}-{e_dim}-{e_thc}'
        return res_dir

    def align_mesh(self):
        """Specify a base electrode pair and align all other simulations
        to the base pair.
        """

        base_ep = self.electrode_base_pairs[0]
        base_res, base_res_fn = self._read_sim_directly(base_ep[0], base_ep[1])
        base_res['aligned'] = True
        torch.save(base_res, base_res_fn)

        base_res['elm_centers'] = base_res['elm_centers'].to(self.device)
        base_res['vecE'] = base_res['vecE'].to(self.device)
        base_res['magnE'] = base_res['magnE'].to(self.device)
        other_eps = self.electrode_base_pairs[1:]

        for ep in tqdm(other_eps):
            res, res_fn = self._read_sim_directly(ep[0], ep[1])
            if 'aligned' in res and res['aligned']:
                continue
            res['elm_centers'] = res['elm_centers'].to(self.device)
            res['vecE'] = res['vecE'].to(self.device)
            res['magnE'] = res['magnE'].to(self.device)
            # align mesh indices

            new_indices, _distances = align_mesh_idx(
                base_res['elm_centers'], res['elm_centers'],
                self.device,
                block_size=20,
                dist_threshold=2,
                verbose=True,
            )
            res['elm_centers'] = res['elm_centers'][new_indices].to('cpu')
            res['vecE'] = res['vecE'][new_indices].to('cpu')
            res['magnE'] = res['magnE'][new_indices].to('cpu')
            res['aligned'] = True
            # save the aligned results
            torch.save(res, res_fn)

    def prune_storage(self) -> None:
        """"""
        pass

    def _read_sim_directly(self, e1: str, e2: str) -> Tuple[torch.Tensor, Path]:
        res_dir = self.sim_dir_fn_gen(self.pre_calculation_dir,
                                      e1, e2, self.current,
                                      self.electrode_shape,
                                      self.electrode_dimensions,
                                      self.electrode_thickness)
        assert res_dir.is_dir()
        res_tensors = res_dir / 'vecE.pt'
        assert res_tensors.is_file()
        res = torch.load(res_tensors)
        return res, res_tensors

    def read_tDCS(self, e1: str, e2: str):
        assert e1 in self.electrodes and e2 in self.electrodes
        if e1 == self.electrode_ref:
            # we can directly get the result
            res, _ = self._read_sim_directly(e1, e2)
            res['elm_centers'] = res['elm_centers'].to(self.device)
            res['vecE'] = res['vecE'].to(self.device)
            res['magnE'] = res['magnE'].to(self.device)
            return res

        elif e2 == self.electrode_ref:
            # we can get (e2, e1) result and then invert it
            # with E((e1, e2)) = -E((e2, e1))
            res, _ = self._read_sim_directly(e2, e1)
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
            res_ep1 = self.read_tDCS(ep_base_1[0], ep_base_1[1])
            res_ep1['elm_centers'] = res_ep1['elm_centers'].to(self.device)
            res_ep1['vecE'] = res_ep1['vecE'].to(self.device)
            res_ep1['magnE'] = res_ep1['magnE'].to(self.device)

            res_ep2 = self.read_tDCS(ep_base_2[0], ep_base_2[1])
            if 'aligned' in res_ep2 and res_ep2['aligned']:
                aligned = True
            else:
                new_indices_ep2, _distances_ep2 = align_mesh_idx(
                    res_ep1['elm_centers'], res_ep2['elm_centers'],
                    self.device,
                    block_size=20,
                    dist_threshold=2,
                    verbose=True,
                )
                res_ep2['elm_centers'] = res_ep2['elm_centers'].to(self.device)[
                    new_indices_ep2]
                res_ep2['vecE'] = res_ep2['vecE'].to(self.device)[
                    new_indices_ep2]
                res_ep2['magnE'] = res_ep2['magnE'].to(self.device)[
                    new_indices_ep2]
                aligned = False

            vecE_diff = res_ep2['vecE'] - res_ep1['vecE']
            vecE_diff_norm = torch.norm(vecE_diff, dim=1)
            return {
                'elm_centers': res_ep1['elm_centers'],
                'vecE': vecE_diff,
                'magnE': vecE_diff_norm,
                'aligned': aligned,
            }

    def get_TI_density(
        self,
        ep1: Tuple[str, str],
        ep2: Tuple[str, str],
        save_TI_mesh: bool = False,
        mesh_path: str = './TI.msh',
        nii_path: str = './TI',
        return_mesh: bool = False
    ):
        # read reference mesh
        res_dir = self.sim_dir_fn_gen(self.pre_calculation_dir,
                                      self.electrode_base_pairs[0][0],
                                      self.electrode_base_pairs[0][1],
                                      self.current,
                                      self.electrode_shape,
                                      self.electrode_dimensions,
                                      self.electrode_thickness)
        mesh = simnibs.read_msh(find_msh(res_dir))
        mesh = mesh.crop_mesh(tags_needed)

        tDCS_res_1 = self.read_tDCS(ep1[0], ep1[1])
        tDCS_res_2 = self.read_tDCS(ep2[0], ep2[1])

        amp_TI = compute_TI_max_magnitude(
            tDCS_res_1['vecE'].float(), tDCS_res_2['vecE'].float()
        ).cpu().numpy()

        if save_TI_mesh:
            mesh.elm.nr
            mesh.add_element_field(amp_TI, 'TI')
            # mesh.elmdata = [ed for ed in mesh.elmdata if ed.field_name == 'TI']
            mesh.crop_mesh(2)
            mesh.write(mesh_path)
            print(f'TI mesh saved to {mesh_path}.')
            create_masks = False
            create_label = False
            transformations.interpolate_to_volume(
                mesh_path, self.model_dir, nii_path,
                create_masks=create_masks,
                create_label=create_label)

        if return_mesh:
            return amp_TI, mesh
        else:
            return amp_TI

    def get_TI_focality(
        self,
        ep1: Tuple[str, str],
        ep2: Tuple[str, str],
        coords: Union[Tuple[float, float, float], List[Tuple[float, float, float]]],
        rs: Union[List[float], float],
        just_gray_matter: bool = True,
    ):
        # read reference mesh
        res_dir = self.sim_dir_fn_gen(self.pre_calculation_dir,
                                      self.electrode_base_pairs[0][0],
                                      self.electrode_base_pairs[0][1],
                                      self.current,
                                      self.electrode_shape,
                                      self.electrode_dimensions,
                                      self.electrode_thickness)
        res_tensors = res_dir / 'vecE.pt'
        assert res_tensors.is_file()

        if len(self._focality_cache) == 0:
            res_ref, mesh_mask, mesh_vols = res2mask_and_vol(
                res_tensors, self.device, just_gray_matter)

            self._focality_cache['mesh_vols'] = mesh_vols
            self._focality_cache['mesh_mask'] = mesh_mask
            self._focality_cache['res_ref'] = res_ref

        tDCS_res_1 = self.read_tDCS(ep1[0], ep1[1])
        tDCS_res_2 = self.read_tDCS(ep2[0], ep2[1])

        amp_TI = compute_TI_max_magnitude(
            tDCS_res_1['vecE'].float(), tDCS_res_2['vecE'].float(),
            tof16=False,
        )
        focality = compute_TI_focality(
            self.model_dir,
            self._focality_cache['res_ref']['elm_centers'].double().clone(),
            amp_TI,
            coords,
            rs,
            mesh_vols=self._focality_cache['mesh_mask'].clone(),
            mesh_mask=self._focality_cache['mesh_mask'].clone(),
        )
        return focality

    def random_search(self,
                      coords: Union[Tuple[float, float, float],
                                    List[Tuple[float, float, float]]],
                      rs: Union[List[float], float],
                      total: int = 100
                      ):
        ep_groups = []
        focalities = []
        for i in range(total):
            t0 = time.time()
            found_ep_groups = False
            while not found_ep_groups:
                ep1 = random.choice(self.electrode_pairs)
                found_ep2 = False
                while not found_ep2:
                    other_electrodes = copy.deepcopy(self.electrode_pairs)
                    ep2 = random.choice(other_electrodes)

                    if len(set(ep2) & set(ep1)) == 0:
                        found_ep2 = True
                ep_group = {ep1, ep2}
                if ep_group not in ep_groups:
                    found_ep_groups = True

            # focalities
            focality = self.get_TI_focality(
                ep1, ep2, coords, rs)

            focalities.append(focality)
            ep_groups.append(ep_group)

            assert len(ep_groups) == len(focalities)

            _focalities_arr = np.array(focalities)
            best_fc = np.max(_focalities_arr)
            mean_fc = np.mean(_focalities_arr)
            dt = time.time() - t0
            print(
                f'Iter {i+1}/{total} ({dt:.1f}s), best: {best_fc:.3f}, '
                f'mean: {mean_fc:.3f}, '
                f'generated by {ep_groups[np.argmax(_focalities_arr)]}.'
            )
            torch.save({
                'focalities': focalities,
                'ep_groups': ep_groups,
            }, 'local.res-random_search.pt')

        return focalities, ep_groups


def test_optti():
    opt = OptTI('data/m2m_ernie',
                'data/optti/sims')
    opt.pre_calculate(cores=8)
    pass


if __name__ == '__main__':
    # test_optti()
    pass
