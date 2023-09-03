# import os
# import time
# from simnibs import sim_struct, run_simnibs
# import shutil

# t0 = time.time()
# # Initalize a session
# s = sim_struct.SESSION()
# # Name of head mesh
# s.subpath = 'data/m2m_ernie'
# # Output folder
# s.pathfem = 'data/tutorial/'
# shutil.rmtree(s.pathfem, ignore_errors=True)

# # Initialize a list of TMS simulations
# # tmslist = s.add_tmslist()
# # Select coil
# # tmslist.fnamecoil = os.path.join('legacy_and_other', 'Magstim_70mm_Fig8.ccd')

# # Initialize a coil position
# # pos = tmslist.add_position()
# # Select coil centre
# # pos.centre = 'C1'
# # # Select coil direction
# # pos.pos_ydir = 'CP1'

# # Initialize a tDCS simulation
# tdcslist = s.add_tdcslist()
# # Set currents
# tdcslist.currents = [-1e-3, 1e-3]

# # Initialize the cathode
# cathode = tdcslist.add_electrode()
# # Connect electrode to first channel (-1e-3 mA, cathode)
# cathode.channelnr = 1
# # Electrode dimension
# cathode.dimensions = [50, 70]
# # Rectangular shape
# cathode.shape = 'rect'
# # 5mm thickness
# cathode.thickness = 5
# # Electrode Position
# cathode.centre = 'C3'
# # Electrode direction
# cathode.pos_ydir = 'Cz'

# # Add another electrode
# anode = tdcslist.add_electrode()
# # Assign it to the second channel
# anode.channelnr = 2
# # Electrode diameter
# anode.dimensions = [30, 30]
# # Electrode shape
# anode.shape = 'ellipse'
# # 5mm thickness
# anode.thickness = 5
# # Electrode position
# anode.centre = 'C4'

# run_simnibs(s, cpus=6)

# dt = time.time() - t0
# print('Elapsed time: %f s' % dt)

from ti_utils import read_eeg_locations

read_eeg_locations()