import itertools
from collections import deque
from datetime import datetime
from pathlib import Path
#import sys
#import os
import h5py
import numpy as np
from event_model import compose_resource
from ophyd import Component as Cpt
from ophyd import EpicsSignal
from ophyd import Device, Signal
from ophyd.sim import NullStatus, new_uid
from ophyd.utils import make_dir_tree

from blop.utils import get_beam_stats
#from blop.sim.handlers import ExternalFileReference
#import matplotlib as mpl
import time

TEST = False
epicsPrefix = "TST"


class xrtEpicsScreen(Device):
    sum = Cpt(Signal, kind="hinted")
    max = Cpt(Signal, kind="normal")
    area = Cpt(Signal, kind="normal")
    cen_x = Cpt(Signal, kind="hinted")
    cen_y = Cpt(Signal, kind="hinted")
    wid_x = Cpt(Signal, kind="hinted")
    wid_y = Cpt(Signal, kind="hinted")
    image = Cpt(EpicsSignal, f'{epicsPrefix}:Screen1:image', kind="normal")
    acquire = Cpt(EpicsSignal, f'{epicsPrefix}:Acquire', kind="normal")
    acquireStatus = Cpt(EpicsSignal, f'{epicsPrefix}:AcquireStatus', kind="normal")
    image_shape = Cpt(Signal, value=(300, 400), kind="normal")
    noise = Cpt(Signal, kind="normal")    

    def __init__(self, root_dir: str = "/tmp/blop/sim", verbose: bool = True,
                 noise: bool = True, *args, **kwargs):
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

        self.acquire.put(1)
        while self.acquireStatus.get() > 0:
            time.sleep(0.01)
        raw_image = self.image.get()
        image = raw_image.reshape(*self.image_shape.get())

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

        self._data_file = str(Path(self._resource_document["root"]) /
                              Path(self._resource_document["resource_path"]))

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
    autoUpdate = Cpt(EpicsSignal, ':AutoUpdate', kind="normal")

    kbh_dsh = Cpt(EpicsSignal, ':TM_HOR:R', kind="hinted")
    kbv_dsv = Cpt(EpicsSignal, ':TM_VERT:R', kind="hinted")


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
