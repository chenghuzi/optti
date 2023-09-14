import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import simnibs
import torch
from simnibs import sim_struct, transformations
from tqdm import tqdm

from .ti_utils import align_mesh_idx, find_msh, tags_needed


def post_process_tDCS_sim(output_dir: Path,
                          size: int = 1,
                          reduce: bool = False,
                          just_gray_matter: bool = False) -> Path:
    """ Analyze the tDCS simulation by calculating the mean electric field
    in the M1 ROI
    """
    msh_f = output_dir / 'ernie_TDCS_1_scalar.msh'
    head_mesh = simnibs.read_msh(msh_f)
    # Here we only care about volumes
    head_mesh = head_mesh.crop_mesh(tags_needed)
    if just_gray_matter:
        head_mesh = head_mesh.crop_mesh(2)
    vecE = torch.from_numpy(head_mesh.field['E'][:])
    magnE = torch.from_numpy(head_mesh.field['magnE'][:])
    elm_centers = torch.from_numpy(
        head_mesh.elements_baricenters()[:])
    if not reduce:
        pass
    else:
        elm_centers_int = elm_centers.int() // size

        _, indices = np.unique(elm_centers.numpy(), return_index=True, axis=0)
        indices_tensor = torch.from_numpy(indices)

        elm_centers_int = elm_centers_int[indices_tensor]
        vecE = vecE[indices_tensor]
        magnE = magnE[indices_tensor]

        elm_centers = elm_centers_int,
    ouput_f = output_dir / 'vecE.pt'
    torch.save({
        'elm_centers': elm_centers.type(torch.float16),
        'vecE': vecE.type(torch.float16),
        'magnE': magnE.type(torch.float16),
    }, ouput_f)
    return ouput_f


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
                 electrode_dimensions: Tuple[float, float] = (50, 50),
                 electrode_thickness=5,
                 post_process_size: int = 1,
                 cpus: int = 1,
                 reduce: bool = False,
                 just_gray_matter: bool = False,
                 ) -> Dict[str, Union[str, int]]:
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

    # cathode, anode
    centres = [cathode_centre, anode_centre]

    # pos_ydirs = ['Cz', 'Cz']

    out_dname = f'{centres[0]}-{centres[1]}-c-{current}-{electrode_shape}\
-{electrode_dimensions}-{electrode_thickness}'
    output_root = output_root / out_dname

    if output_root.is_dir():
        size = 1
        ouput_f = output_root / f'vecE-size-{size}.pt'
        if ouput_f.is_file():
            return {
                'output_root': str(output_root.resolve()),
                'n_vertices': torch.load(ouput_f)['vecE'].shape[0],
                'tensor_E': str(ouput_f.resolve()),
            }
        else:
            shutil.rmtree(output_root)

    # Initalize a session
    s = sim_struct.SESSION()
    s.open_in_gmsh = False
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
    s.run(cpus=cpus, save_mat=False)
    dt = time.time() - t0
    print('Elapsed time: %f s' % dt)

    tensor_E = post_process_tDCS_sim(
        output_root, size=post_process_size, reduce=reduce,
        just_gray_matter=just_gray_matter,
    )
    n_vertices = simnibs.read_msh(
        find_msh(output_root)).elements_baricenters()[:].shape[0]
    return {
        'output_root': str(output_root.resolve()),
        'n_vertices': n_vertices,
        'tensor_E': str(tensor_E.resolve()),
    }


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

def compute_TI_max_magnitude(
        vecE_base: torch.Tensor,
        vecE_2: torch.Tensor,
        eps: float = 1e-16,
        tof16: bool = True,
) -> torch.Tensor:
    """
    Computes the maximum magnitude of the Ti simulation.
    Implementation based on the following papers:
    - 10.1016/j.cell.2017.05.024
    - 10.1016/j.brs.2018.09.010

    Args:
        vecE_base (torch.Tensor): A tensor of shape (N, 3) representing the base
            electric field vector.
        vecE_2 (torch.Tensor): A tensor of shape (N, 3) representing the second
            electric field vector.
        eps (float, optional): A small value used to avoid division by zero.
            Defaults to 1e-16.
        tof16 (bool, optional): A flag indicating whether to return the result
            as a tensor of dtype torch.float16. Defaults to True.

    Returns:
        torch.Tensor: A tensor of shape (N,) representing the maximum magnitude of the TI for each pair of electric field vectors.
    """
    magnE_base = vecE_base.norm(dim=1)
    magnE_2 = vecE_2.norm(dim=1)
    mask = magnE_base > magnE_2
    # flip according to the condition |E1| > |E2|
    tmp_magnE = magnE_base[torch.logical_not(mask)]
    magnE_base[torch.logical_not(mask)] = magnE_2[torch.logical_not(mask)]
    magnE_2[torch.logical_not(mask)] = tmp_magnE

    # assert (magnE_base > magnE_2).all()

    tmp_E = vecE_base[torch.logical_not(mask)]
    vecE_base[torch.logical_not(mask)] = vecE_2[torch.logical_not(mask)]
    vecE_2[torch.logical_not(mask)] = tmp_E

    vec_diffE = vecE_base - vecE_2

    # this is the amplitude when |E1| cos alpha <= |E2|
    tmp_amp = torch.cross(vecE_2, vec_diffE, dim=1).norm(
        dim=1) / (vec_diffE.norm(dim=1) +eps)

    cosine_alpha = torch.abs((vecE_base * vecE_2).sum(dim=1) / (
        vecE_base.norm(dim=1) * vecE_2.norm(dim=1) +eps
    ))
    # this is the condition when |E1| cos alpha > |E2|
    angle_mask = (magnE_base * cosine_alpha > magnE_2).float()

    assert torch.isnan(tmp_amp).any() == torch.tensor(False)
    assert torch.isnan(magnE_2).any() == torch.tensor(False)
    amplitude = 2 * (angle_mask * magnE_2 + (1 - angle_mask) * tmp_amp)
    if tof16:
        return amplitude.half()
    else:
        return amplitude

def cartesian2sphere(x:torch.Tensor)->torch.Tensor:
    return torch.stack(
        (x.norm(dim=-1),
         torch.acos(x[..., 2] / x.norm(dim=-1)),
         torch.atan2(x[..., 1], x[..., 0])),
        dim=-1)

def approx_TI_max_magnitude(
        vecE_base: torch.Tensor,
        vecE_2: torch.Tensor,
        eps: float = 1e-16,
        tof16: bool = True,
        k:int = 14,
        block_size = 10000,
        verbose:bool = False,
) -> torch.Tensor:
    """
    Computes the maximum magnitude of the Ti simulation.
    Implementation based on the following papers:
    - 10.1016/j.cell.2017.05.024
    - 10.1016/j.brs.2018.09.010

    Args:
        vecE_base (torch.Tensor): A tensor of shape (N, 3) representing the base
            electric field vector.
        vecE_2 (torch.Tensor): A tensor of shape (N, 3) representing the second
            electric field vector.
        eps (float, optional): A small value used to avoid division by zero.
            Defaults to 1e-16.
        tof16 (bool, optional): A flag indicating whether to return the result
            as a tensor of dtype torch.float16. Defaults to True.

    Returns:
        torch.Tensor: A tensor of shape (N,) representing the maximum magnitude of the TI for each pair of electric field vectors.
    """
    with torch.no_grad():
        eps = 1e-16
        angle_mesh = torch.stack(torch.meshgrid(
            torch.linspace(eps, 2* np.pi, k*2),
            torch.linspace(eps, np.pi/2, k//2),
            indexing=None
            ), -1).view(-1, 2) # z>0, northern hemisphere
        unit_vecs_sphere = torch.cat((
            torch.ones_like(angle_mesh[..., 0].unsqueeze(-1)),
            angle_mesh
        ), dim=1)
        xs = unit_vecs_sphere[...,0 ] * torch.sin(
            unit_vecs_sphere[..., 1]) * torch.cos(unit_vecs_sphere[..., 2])
        ys = unit_vecs_sphere[...,0 ] * torch.sin(
            unit_vecs_sphere[..., 1]) * torch.sin(unit_vecs_sphere[..., 2])
        zs = unit_vecs_sphere[...,0 ] * torch.cos(unit_vecs_sphere[..., 1])

        unit_vecs = torch.stack((xs, ys, zs), dim=-1).to(vecE_base.device)
        vecE_south_idx = torch.where(vecE_base[...,0]<0)
        vecE_base[vecE_south_idx] = -vecE_base[vecE_south_idx]

        vecE_south_idx = torch.where(vecE_2[...,0]<0)
        vecE_2[vecE_south_idx] = -vecE_2[vecE_south_idx]


    approximated_TI_amplitude = []
    pbar = range(len(vecE_base)//block_size +1)
    if verbose:
        pbar = tqdm(pbar)
    for i in pbar:
        if vecE_base[i*block_size:(i+1)*block_size].shape[0] == 0:
            continue
        vecE_base_b = vecE_base[i*block_size:(i+1)*block_size]
        vecE_2_b = vecE_2[i*block_size:(i+1)*block_size]
        tmpv = torch.stack((
            torch.mm(vecE_base_b, unit_vecs.T),
            torch.mm(vecE_2_b, unit_vecs.T)
            ), dim=-1)
        tmp_amp = tmpv.min(dim=-1).values.max(dim=-1).values

        approximated_TI_amplitude.append(tmp_amp)

    approximated_TI_amplitude = torch.cat(approximated_TI_amplitude)*2


    if tof16:
        return approximated_TI_amplitude.half()
    else:
        return approximated_TI_amplitude





def compute_BE_TI_from_tDCS_sims(subject_dir: Path,
                                 output_dir_1: Path,
                                 output_dir_2: Path,
                                 just_gray_matter: bool = False,
                                 plot_alignment: bool = False,
)-> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute the bi-electrode TI effect using two tDCS simulations.

    Args:
        subject_dir (Path): The subject directory.
        output_dir_1 (Path): The output directory of the first simulation.
        output_dir_2 (Path): The output directory of the second simulation.
        just_gray_matter (bool, optional): A flag indicating whether to just
            consider the gray matter part. Defaults to False. This will save
            some computation time if True.
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
        # print('swapping meshes')
        head_mesh_base, head_mesh_2 = head_mesh_2, head_mesh_base
        elm_centers_base, elm_centers2 = elm_centers2, elm_centers_base

    new_indices_2, distances = align_mesh_idx(
        elm_centers_base, elm_centers2, device)


    if plot_alignment:
        plt.hist(distances.data.cpu().numpy(), bins=100)
        plt.savefig('distances-analyze_BE_TI_from_tDCS_sims.png')

    elm_centers2 = elm_centers2[new_indices_2]
    vecE_base = torch.from_numpy(head_mesh_base.field['E'][:]).to(device)
    vecE_2 = torch.from_numpy(head_mesh_2.field['E'][:]).to(device)[
        new_indices_2]

    # magnE_base = torch.from_numpy(head_mesh_base.field['magnE'][:]).to(device)
    # magnE_2 = torch.from_numpy(head_mesh_2.field['magnE'][:]).to(device)[
    #     new_indices_2]
    # Now let's calculate the max magnitude across the whole model.
    # max_ti_magn = compute_TI_max_magnitude(vecE_base, vecE_2)
    max_ti_magn = approx_TI_max_magnitude(vecE_base, vecE_2)
    assert max_ti_magn.shape[0] == elm_centers_base.shape[0]

    return elm_centers_base, max_ti_magn, (vecE_base, vecE_2)



def compute_TI_focality(
    subject_dir: Path,
    elm_centers: torch.Tensor,
    max_ti_magn: torch.Tensor,
    mni_coords: Union[Tuple[float, float, float], List[Tuple[float, float, float]]],
    rs: Union[List[float], float],
    mesh_vols: torch.Tensor = None,
    mesh_mask: torch.Tensor = None,
    quantile_threshold: float = 0.99,
    )-> torch.Tensor:
    """
    Computes the focality of TI values for a given set of coordinates and radii.
    # TODO Maybe we need to take volume into account too.

    Args:
        elm_centers (torch.Tensor): A tensor of shape (n_elms, 3) representing
            the coordinates of the elements.
        max_ti_magn (torch.Tensor): A tensor of shape (n_elms,) representing the
            maximum TI magnitude for each element.
        mni_coords (Union[Tuple[float, float, float],
            List[Tuple[float, float, float]]]): A tuple or list of tuples
            representing the coordinates of the spots.
        rs (Union[List[float], float]): A float or list of floats representing
            the radii of the spots.

    Returns:
        float: Focality
    """
    if isinstance(mni_coords, tuple):
        mni_coords = [mni_coords]

    if isinstance(rs, (float, int)):
        rs = [float(rs)] * len(mni_coords)
    assert len(mni_coords) == len(rs), 'The number of coordinates and radii \
        must be the same'
    for idx, (coord, r) in enumerate(zip(mni_coords, rs)):
        subj_coords = simnibs.mni2subject_coords(coord, str(subject_dir))
        mni_coords[idx] = (subj_coords[0], subj_coords[1], subj_coords[2])

        assert len(coord) == 3, 'Each coordinate must have 3 values'
        assert r > 0, 'The radius must be positive'
    with torch.no_grad():
        mni_coords = torch.tensor(mni_coords).float().to(max_ti_magn.device)
        rs = torch.tensor(rs).float().to(max_ti_magn.device)

        dis2spots = torch.cdist(elm_centers.float(), mni_coords)
        coord_cond = dis2spots < rs # of shape  (n_elms, n_spots)
        found = coord_cond.any(dim=0).all()
        if found.float() != 1:
            raise ValueError('There are some spots that are \
                not covered by any element. Found ones: {found}')

        coord_mask = coord_cond.any(dim=1) # .float() # of shape (n_elms,)
        noncoord_mask = torch.logical_not(coord_mask)
        if mesh_mask is not None:
            coord_mask = torch.logical_and(coord_mask, mesh_mask)
        focal_coords = torch.where(coord_mask == torch.tensor(True))

        # this actually considers many other elements other than the GM and WM
        nonfocal_coords = torch.where(
            torch.logical_and(noncoord_mask, mesh_mask) == torch.tensor(True)
            ) 
    focal_magn = max_ti_magn[focal_coords]
    nonfocal_magn = max_ti_magn[nonfocal_coords]
    if mesh_vols is not None:
        # # method 1
        # focal_vol = mesh_vols[focal_coords].sum() # total volume
        # focal_magn = (focal_magn * mesh_vols[focal_coords]).sum() # total magnitude

        # focal_magn = focal_magn / focal_vol # average density

        # nonfocal_vol = mesh_vols[nonfocal_coords].sum()
        # nonfocal_magn = (nonfocal_magn * mesh_vols[nonfocal_coords]).sum() / nonfocal_vol

        # # method 2
        # eps = 1e-16
        # focal_magn = focal_magn / (mesh_vols[focal_coords] + eps)
        # focal_magn = focal_magn.mean()

        # nonfocal_magn = nonfocal_magn / (mesh_vols[nonfocal_coords] + eps)
        # nonfocal_magn = nonfocal_magn.mean()
        # mesh_weights = mesh_vols / mesh_vols.sum()

        # # method 3
        focal_magn = focal_magn.mean()
        # nonfocal_magn = nonfocal_magn.max()
        nonfocal_magn = torch.quantile(nonfocal_magn, quantile_threshold)
        # import ipdb; ipdb.set_trace() # fmt: off
        # nonfocal_magn = nonfocal_magn.min()

        # # method 4
        # focal_vol = mesh_vols[focal_coords]
        # focal_vol = focal_vol / focal_vol.sum()
        # focal_magn = (focal_magn * mesh_vols[focal_coords]).sum() # average magnitude
        # # focal_magn = focal_magn.mean()

        # # focal_magn = focal_magn / focal_vol # average density
        # # import ipdb; ipdb.set_trace() # fmt: off
        # # nonfocal_vol = mesh_vols[nonfocal_coords]
        # # nonfocal_vol = nonfocal_vol / nonfocal_vol.sum()
        # # nonfocal_magn = (nonfocal_magn * mesh_vols[nonfocal_coords]).sum()
        # nonfocal_magn = nonfocal_magn.max()

        pass
    else:
        focal_magn = focal_magn.mean()
        nonfocal_magn = nonfocal_magn.mean()

    focality = focal_magn / nonfocal_magn

    return focality



def analyze_TI_from_sims(
        subject_folder: Path,
        output_dir_1: Path,
        output_dir_2: Path,
        coords: Union[Tuple[float, float, float], List[Tuple[float, float, float]]],
        rs: Union[List[float], float],
        **kwargs,
) -> Tuple[float, dict, List[Tuple[float, float, float]], List[float]]:
    """
    Analyzes tDCS simulations to compute the TI focality index and its
        associated information.

    Args:
        subject_folder (Path): Path to the subject folder containing the tDCS
            simulation data.
        output_dir_1 (Path): Path to the first output directory.
        output_dir_2 (Path): Path to the second output directory.
        coords (Union[Tuple[float, float, float], List[Tuple[float, float, float]]]): 
            Coordinates of the region(s) of interest for computing TI.
        rs (Union[List[float], float]): Radius or radii of the region(s) of
            interest for computing TI.
        kwargs: Additional keyword arguments for `compute_BE_TI_from_tDCS_sims`.


    Returns:
        Tuple[float, dict, List[Tuple[float, float, float]], List[float]]: 
            A tuple containing the TI focality, its associated information, 
            the element centers, and the TI magnitudes.
    """

    elm_centers, ti_magn, _ = compute_BE_TI_from_tDCS_sims(
        subject_folder,
        output_dir_1,
        output_dir_2,
        **kwargs,
    )

    focality = compute_TI_focality(
        subject_folder,
        elm_centers, ti_magn,
        coords, rs)
    focality_info = None
    return focality, focality_info, elm_centers, ti_magn


def res2mask_and_vol(res_tensors: Path, device: str, just_gray_matter: bool):
    res_ref = torch.load(res_tensors)
    res_ref['elm_centers'] = res_ref['elm_centers'].to(device)

    head_mesh = simnibs.read_msh(find_msh(res_tensors.parent))
    head_mesh = head_mesh.crop_mesh(tags_needed)
    if just_gray_matter:
        mesh_mask = torch.from_numpy(
            head_mesh.elm.tag1 == 2
        ).to(device)
    else:
        # mesh_mask = torch.ones(head_mesh.elm.nr).bool().to(device)
        # import ipdb; ipdb.set_trace() # fmt: off
        mask_123 = (head_mesh.elm.tag1 == 2) | \
            (head_mesh.elm.tag1 == 1) | \
            (head_mesh.elm.tag1 == 3)
        mesh_mask = torch.from_numpy(mask_123).to(device)

    mesh_vols = torch.from_numpy(
        head_mesh.elements_volumes_and_areas()[:]
    ).to(device)
    return res_ref, mesh_mask, mesh_vols


def save_msh2mni(model_dir, mesh, amp_TI, mesh_path, nii_path, crop=True, to_mni=True):
    mesh.add_element_field(amp_TI, 'TI')
    mesh.elmdata = [ed for ed in mesh.elmdata if ed.field_name == 'TI']
    if crop:
        mesh = mesh.crop_mesh([1, 2, 3])
    mesh.write(mesh_path)
    print(f'TI mesh saved to {mesh_path}.')

    if to_mni:
        transformations.warp_volume(
            mesh_path, model_dir, nii_path,
            transformation_direction='subject2mni',
            transformation_type='nonl',
            )
    else:
        mni_mesh_path = mesh_path.split('.')[0] + '-mni'
        transformations.interpolate_to_volume(
            mni_mesh_path, model_dir, nii_path)