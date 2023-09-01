import shutil
import time
from pathlib import Path
import torch
import numpy as np
import simnibs
from simnibs import run_simnibs, sim_struct
from typing import Tuple, Union
from scipy.spatial import distance
from tqdm import tqdm
import matplotlib.pyplot as plt


def add_tdcs_electrodes(s: sim_struct.SESSION,
                        current: float,
                        centres,
                        electrode_shape: str,
                        electrode_dimensions,
                        electrode_thickness,
                        pos_ydirs=(None, None),
                        ) -> None:
    """Add tDCS electrodes to a sim_struct.SESSION

    Args:
        s (sim_struct.SESSION): _description_
        current (float): DC current, currents in A. Not in mA!
        centres (_type_): cathode, anode centre locations
        electrode_shape (_type_): electrode shape
        electrode_dimensions (_type_): the dimensions of the electrode
        electrode_thickness (_type_): the thickness of the electrode
        pos_ydirs (tuple, optional): the y-directions of electrodes.
            Defaults to (None, None).
    """
    if electrode_shape == 'ellipse':
        assert pos_ydirs == (
            None, None), 'pos_ydir must be None for ellipse electrodes'

    # Initialize a tDCS simulation
    tdcslist = s.add_tdcslist()
    # Set currents
    tdcslist.currents = [-current, current]

    # === cathode ===

    # Initialize the cathode
    cathode = tdcslist.add_electrode()
    # Connect electrode to first channel (-1e-3 mA, cathode)
    cathode.channelnr = 1
    # Electrode dimension
    cathode.dimensions = electrode_dimensions
    # Rectangular shape
    cathode.shape = electrode_shape
    # 5mm thickness
    cathode.thickness = electrode_thickness
    # Electrode Position
    cathode.centre = centres[0]
    # Electrode direction
    if pos_ydirs[0] is not None:
        cathode.pos_ydir = pos_ydirs[0]

    # === anode ===
    # Add another electrode
    anode = tdcslist.add_electrode()
    # Assign it to the second channel
    anode.channelnr = 2
    # Electrode diameter
    anode.dimensions = electrode_dimensions
    # Electrode shape
    anode.shape = electrode_shape
    # 5mm thickness
    anode.thickness = electrode_thickness

    # Electrode position
    anode.centre = centres[1]
    # Electrode direction
    if pos_ydirs[1] is not None:
        anode.pos_ydir = pos_ydirs[1]
    print('cathode: \n', cathode)
    print('anode: \n', anode)


def run_tDCS_sim(subject_dir: Path,
                 output_root: Path,
                 cathode_centre: Union[str, Tuple[float, float, float]],
                 anode_centre: Union[str, Tuple[float, float, float]],
                 current: float,
                 electrode_shape: str = 'ellipse',
                 electrode_thickness=5,
                 ):
    """_summary_

    Args:
        subject_dir (Path): Subject directory
        output_root (Path): Output root directory
        cathode_centre (str): cathode centre
        anode_centre (str): anode centre
        current (float): DC current, currents in A. Not in mA!
        electrode_shape (str, optional): electrode shape. Defaults to 'ellipse'.
        electrode_thickness (int, optional): electrode thickness. Defaults to 5.
    """
    assert subject_dir.is_dir(), 'subject_dir must be a directory'
    assert output_root.is_dir(), 'output_dir must be a directory'

    # Single-Pair tDCS simulation

    current = 1e-3
    # cathode, anode
    centres = [cathode_centre, anode_centre]

    # pos_ydirs = ['Cz', 'Cz']
    electrode_dimensions = [50, 50]

    out_dname = f'{centres[0]}-{centres[1]}-c-{current}'
    output_root = output_root / out_dname

    if output_root.is_dir():
        shutil.rmtree(output_root)

    # Initalize a session
    s = sim_struct.SESSION()
    s.subpath = str(subject_dir)
    # Output folder
    s.pathfem = str(output_root)
    add_tdcs_electrodes(s, current,
                        centres,
                        electrode_shape,
                        electrode_dimensions,
                        electrode_thickness,
                        # pos_ydirs=pos_ydirs
                        )

    t0 = time.time()
    run_simnibs(s, cpus=8)
    dt = time.time() - t0
    print('Elapsed time: %f s' % dt)


def analyze_tDCS_sim(subject_dir: Path, output_dir: Path):
    """ Analyze the tDCS simulation by calculating the mean electric field
    in the M1 ROI
    """
    msh_f = output_dir / 'ernie_TDCS_1_scalar.msh'
    head_mesh = simnibs.read_msh(msh_f)

    # Crop the mesh so we only have gray matter volume elements (tag 2 in the mesh)
    gray_matter = head_mesh.crop_mesh(2)

    # Define the ROI

    # Define M1 from MNI coordinates (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2034289/)
    # the first argument is the MNI coordinates
    # the second argument is the subject "m2m" folder
    mni_coords = [-37, -21, 58]
    r = 10.
    ernie_coords = simnibs.mni2subject_coords(mni_coords, str(subject_dir))
    # we will use a sphere of radius 10 mm

    # Electric fields are defined in the center of the elements
    # get element centers
    # this gives us a numpy array of shape (n,3).
    # get the coordinate of the element centers in the mesh
    elm_centers = gray_matter.elements_baricenters()[:]

    # determine the elements in the ROI
    roi = np.linalg.norm(elm_centers - ernie_coords,
                         axis=1) < r  # True/False array
    # get the element volumes, we will use those for averaging

    elm_vols = gray_matter.elements_volumes_and_areas()[:]

    # Plot the ROI
    gray_matter.add_element_field(roi, 'roi')
    import ipdb; ipdb.set_trace() # fmt: off
    # gray_matter.view(visible_fields='roi').show()


    # Get field and calculate the mean
    # get the field of interest
    field_name = 'magnE'
    field = gray_matter.field[field_name][:]

    # Efield = gray_matter.field['E'][:]

    # Calculate the mean
    mean_magnE = np.average(field[roi], weights=elm_vols[roi])
    print('mean ', field_name, ' in M1 ROI: ', mean_magnE)
    pass


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
    # angles = np.arccos(cos_angles)
    # return np.degrees(angles)
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


def analyze_BE_TI_from_tDCS_sims(subject_dir: Path,
                                 output_dir_1: Path,
                                 output_dir_2: Path,
                                 ):
    """Try to extract the bi-electrode  TI effect using two tDCS simulations.

    Args:
        output_dir (Path): _description_
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    head_mesh_base = simnibs.read_msh(find_msh(output_dir_1))
    elm_centers_base = torch.from_numpy(head_mesh_base.elements_baricenters()[:]).to(device)

    head_mesh_2 = simnibs.read_msh(find_msh(output_dir_2))
    elm_centers2 = torch.from_numpy(head_mesh_2.elements_baricenters()[:]).to(device)

    if not elm_centers_base.shape[0] < elm_centers2.shape[0]:
        print('swapping meshes')
        head_mesh_base, head_mesh_2 = head_mesh_2, head_mesh_base
        elm_centers_base, elm_centers2 = elm_centers2, elm_centers_base

    new_indices_2, distances = align_mesh_idx(elm_centers_base, elm_centers2, device)

    plt.hist(distances.data.cpu().numpy(), bins=100)
    plt.savefig('distances.png')



    elm_centers2 = elm_centers2[new_indices_2]

    # Now let's calculate the amplitude of the whole mesh.
    magnE_base =  torch.from_numpy(head_mesh_base.field['magnE'][:]).to(device)
    vecE_base =  torch.from_numpy(head_mesh_base.field['E'][:]).to(device)

    magnE_2 =  torch.from_numpy(head_mesh_2.field['magnE'][:]).to(device)[new_indices_2]
    vecE_2 =  torch.from_numpy(head_mesh_2.field['E'][:]).to(device)[new_indices_2]

    # Now let's calculate the amplitude of the whole mesh.
    cosine_alpha = (vecE_base * vecE_2).sum(dim=1) / (
        vecE_base.norm(dim=1) * vecE_2.norm(dim=1)
        )

    mask = magnE_base > magnE_2

    angle_mask = magnE_base * cosine_alpha > magnE_2

    # cosine_alpha

    # mni_target_coords = [-37, -21, 58]
    # target_radius = 10.
    # ernie_target_coords = simnibs.mni2subject_coords(mni_target_coords, str(subject_dir))
    # roi = np.linalg.norm(elm_centers_base - ernie_target_coords,
    #                      axis=1) < target_radius  # True/False array
    # roi_indices = np.argwhere(roi>0)
    # non_overlap_ratio = (new_indices_2<0).float().mean()
    # print(non_overlap_ratio)






    # this gives us a numpy array of shape (n,3). each row is the coordinate of the element center



    # res_elms = np.linalg.norm(elm_centers_base, elm_centers2)


    # dist_matrix = distance.cdist(elm_centers_base, elm_centers2, 'euclidean')
    # # now for each element we calculate its amplitude
    # # vecE_1 * vecE_2
    # import ipdb; ipdb.set_trace() # fmt: off
    pass





    pass




    import ipdb; ipdb.set_trace() # fmt: off
    pass



def main():
    # Input and output folders
    subject_folder = Path('data/m2m_ernie')
    output_folder = Path('data/single_sgp_sims/')
    output_folder.mkdir(exist_ok=True)
    current = 1e-3
    # cathode, anode
    cathode_centre = 'P8'
    anode_centre = 'F8'
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

    output_dir_1 = Path('data/single_sgp_sims/P8-F8-c-0.001')
    output_dir_2 = Path('data/single_sgp_sims/P7-F7-c-0.001')
    analyze_BE_TI_from_tDCS_sims(
        subject_folder,
        output_dir_1,
        output_dir_2
    )
    pass


if __name__ == '__main__':
    main()
