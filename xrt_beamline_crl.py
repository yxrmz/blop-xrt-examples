import itertools
from collections import deque
from datetime import datetime
from pathlib import Path
import sys
import os
import h5py
import numpy as np
import scipy as sp
from event_model import compose_resource
from ophyd import Component as Cpt
from ophyd.device import DynamicDeviceComponent as DDC
from ophyd import EpicsSignal, Kind
from ophyd import Device, Signal
from ophyd.sim import NullStatus, new_uid
from ophyd.utils import make_dir_tree

from blop.utils import get_beam_stats
from blop.sim.handlers import ExternalFileReference
import matplotlib as mpl
import time
from collections import OrderedDict
# os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
# os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"

# import xrt.backends.raycing.run as rrun
# import xrt.backends.raycing as raycing
# import xrt.plotter as xrtplot
# import xrt.runner as xrtrun
# sys.path.append('/home/rchernikov/github/xrt/examples/withRaycing/_QookBeamlines')
# from trace_KB_elliptical import build_beamline, run_process, build_histRGB
sys.path.append(r"C:\NSLS\LINUX\github\blop-xrt-examples")
#sys.path.append("/home/rcherniko/github/blop-xrt-examples")
#from trace_KB import build_beamline, run_process, build_histRGB
#from crl_individual_3D import build_beamline
# rrun.run_process = run_process
# from matplotlib import pyplot as plt


# def plot_generator(beamLine, plots):
#     while True:
#         yield
    # print(plots[0].intensity, plots[0].total2D.shape)
# def plot_generator(beamLine, plots):
#     yield
TEST = False
epicsPrefix = "CRL:"

class Motor(Device):
    x = Cpt(EpicsSignal, ":center:x", kind="hinted")
    y = Cpt(EpicsSignal, ":center:y", kind="hinted")
    z = Cpt(EpicsSignal, ":center:z", kind="hinted")


def make_crl_array(n_crl):
    out_dict = OrderedDict()
    for lens_n in range(n_crl):
        attr = f'Lens{lens_n:02d}'
        out_dict[attr] = (Motor, attr, dict())
    return out_dict
   
    
# class LensArray(Device):
#     positions = DDC(make_crl_array(47), kind="hinted")    

class xrtEpicsScreen(Device):
    sum = Cpt(Signal, kind="hinted")
    max = Cpt(Signal, kind="normal")
    area = Cpt(Signal, kind="normal")
    cen_x = Cpt(Signal, kind="hinted")
    cen_y = Cpt(Signal, kind="hinted")
    wid_x = Cpt(Signal, kind="hinted")
    wid_y = Cpt(Signal, kind="hinted")
    image = Cpt(EpicsSignal, f'{epicsPrefix}FSM_focus:image', kind="normal")
    acquire = Cpt(EpicsSignal, f'{epicsPrefix}Acquire', kind="normal")
    acquireStatus = Cpt(EpicsSignal, f'{epicsPrefix}AcquireStatus', kind="normal")
    image_shape = Cpt(Signal, value=(256, 256), kind="normal")
    noise = Cpt(Signal, kind="normal")    

    def __init__(self,
                 root_dir: str = "c:/nsls/blop/sim",
#                 root_dir: str = "/tmp/blop/sim",
                 verbose: bool = True,
                 noise: bool = True, *args, **kwargs):
        # self.parent = kwargs.pop['parent']
        _ = make_dir_tree(datetime.now().year, base_path=root_dir)

        self._root_dir = root_dir
        self._verbose = verbose

        # Used for the emulated cameras only.
        self._img_dir = None

        # Resource/datum docs related variables.
        self._asset_docs_cache = deque()
        self._resource_document = None
        self._datum_factory = None
        super().__init__(*args, **kwargs)

    def trigger(self):
        super().trigger()
        # raw_image = self.generate_beam(noise=self.noise.get())
        self.acquire.put(1)
        while self.acquireStatus.get() > 0:
            time.sleep(0.01)
        raw_image = self.image.get()
        image = raw_image.reshape(*self.image_shape.get())
        # print(image.shape)
        # print("Reshaped image shape", image.shape)

        current_frame = next(self._counter)

        self._dataset.resize((current_frame + 1, *self.image_shape.get()))

        self._dataset[current_frame, :, :] = image

        datum_document = self._datum_factory(datum_kwargs={"frame": current_frame})
        self._asset_docs_cache.append(("datum", datum_document))

        stats = get_beam_stats(image)
        # self.image.put(datum_document["datum_id"])

        for attr in ["max", "sum", "cen_x", "cen_y", "wid_x", "wid_y"]:
            getattr(self, attr).put(stats[attr])

        # super().trigger()

        return NullStatus()

    def stage(self):
        super().stage()
        date = datetime.now()
        self._assets_dir = date.strftime("%Y/%m/%d")
        data_file = f"{new_uid()}.h5"

        self._resource_document, self._datum_factory, _ = compose_resource(
            start={"uid": "needed for compose_resource() but will be discarded"},
            spec="HDF5",
            root=self._root_dir,
            resource_path=str(Path(self._assets_dir) / Path(data_file)),
            resource_kwargs={},
        )

        self._data_file = str(Path(self._resource_document["root"]) / Path(self._resource_document["resource_path"]))

        # now discard the start uid, a real one will be added later
        self._resource_document.pop("run_start")
        self._asset_docs_cache.append(("resource", self._resource_document))

        self._h5file_desc = h5py.File(self._data_file, "x")
        group = self._h5file_desc.create_group("/entry")
        self._dataset = group.create_dataset(
            "image",
            data=np.full(fill_value=np.nan, shape=(1, *self.image_shape.get())),
            maxshape=(None, *self.image_shape.get()),
            chunks=(1, *self.image_shape.get()),
            dtype="float64",
            compression="lzf",
        )
        self._counter = itertools.count()

    def unstage(self):
        super().unstage()
        del self._dataset
        self._h5file_desc.close()
        self._resource_document = None
        self._datum_factory = None


class BeamlineEpics(Device):
    det = Cpt(xrtEpicsScreen, name="DetectorScreen")
    lenses = DDC(make_crl_array(48), kind="hinted")
    autoUpdate = Cpt(EpicsSignal, 'AutoUpdate', kind="normal")
    # kbh_ush = Cpt(Signal, kind="hinted")
    # kbh_dsh = Cpt(EpicsSignal, ':TM_HOR:R', kind="hinted")
    # kbv_usv = Cpt(Signal, kind="hinted")
    # kbv_dsv = Cpt(EpicsSignal, ':TM_VERT:R', kind="hinted")

    # ssa_inboard = Cpt(Signal, value=-5.0, kind="hinted")
    # ssa_outboard = Cpt(Signal, value=5.0, kind="hinted")
    # ssa_lower = Cpt(Signal, value=-5.0, kind="hinted")
    # ssa_upper = Cpt(Signal, value=5.0, kind="hinted")

    def __init__(self, *args, **kwargs):
#        self.beamline = build_beamline()
        super().__init__(*args, **kwargs)

