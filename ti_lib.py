import shutil
import time
from pathlib import Path
from typing import Tuple, Union, Dict

import numpy as np
import simnibs
import torch
from simnibs import sim_struct
from ti_utils import find_msh


def post_process_tDCS_sim(output_dir: Path,
                          size: int = 1,
                          reduce: bool = False,
                          just_gray_matter: bool = False) -> Path:
    """ Analyze the tDCS simulation by calculating the mean electric field
    in the M1 ROI
    """
    msh_f = output_dir / 'ernie_TDCS_1_scalar.msh'
    head_mesh = simnibs.read_msh(msh_f)
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
    ouput_f = output_dir / f'vecE-size-{size}.pt'
    torch.save({
        'elm_centers': elm_centers,
        'vecE': vecE,
        'magnE': magnE,
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

    out_dname = f'{centres[0]}-{centres[1]}-c-{current}'
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

