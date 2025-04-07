#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:32:06 2024

@author: rchernikov
"""

import numpy as np
import sys, os
from bluesky.plan_stubs import mv
os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
#import matplotlib as mpl
#mpl.use("Agg")
from matplotlib import pyplot as plt
import blop
from blop.utils import prepare_re_env as pre
from xrt_beamline_epics import BeamlineEpics
from blop import DOF, Objective, Agent
from blop.digestion import beam_stats_digestion
import time


plt.ion()

kwargs_re = dict(
        db_type='temp',
        root_dir=pre.DEFAULT_ROOT_DIR)
ret = pre.re_env(**kwargs_re)

RE = ret['RE']
db = ret['db']
bec = ret['bec']

R1, dR1 = 40000, 10000
R2, dR2 = 20000, 10000

bec.disable_plots()

isEPICS = True

if isEPICS:
    beamline = BeamlineEpics('TST', name="bl") 
    
time.sleep(1)

dofs = [
    DOF(description="KBV R",
        device=beamline.kbv_dsv,
        search_domain=(R1-dR1, R1+dR1)),
    DOF(description="KBH R",
        device=beamline.kbh_dsh,
        search_domain=(R2-dR2, R2+dR2)),
]

objectives = [
    Objective(name="bl_det_sum", 
              target="max",
              transform="log",
              trust_domain=(20, 1e12)),

    Objective(name="bl_det_wid_x",
              target="min",
              transform="log",
              latent_groups=[("bl_kbh_dsh", "bl_kbv_dsv")]),
    Objective(name="bl_det_wid_y",
              target="min",
              transform="log",
              latent_groups=[("bl_kbh_dsh", "bl_kbv_dsv")]),

    # Objective(name="bl_det_cen_x",
    #           target=(190., 210.),
    #           # transform="log",
    #             # trust_domain=(50, 350),
    #           latent_groups=[("bl_kbh_dsh", "bl_kbh_ush")]),
    # Objective(name="bl_det_cen_y",
    #           target=(140., 160.),
    #           # transform="log",
    #           # trust_domain=(50, 250),
    #           latent_groups=[("bl_kbv_dsv", "bl_kbv_usv")]
    #             )
]

digestion_kwargs = {"image_key": "bl_det_image"}
if isEPICS:
    digestion_kwargs.update({"image_shape": (300, 400)})
                    

agent = Agent(
    dofs=dofs,
    objectives=objectives,
    detectors=[beamline.det],
    digestion=beam_stats_digestion,
    digestion_kwargs=digestion_kwargs,
    verbose=True,
    db=db,
    tolerate_acquisition_errors=False,
    enforce_all_objectives_valid=True,
    train_every=3,
)

RE(mv(beamline.autoUpdate, 0))
time.sleep(0.5)
RE(agent.learn("qr", n=16))
RE(agent.learn("qei", n=4, iterations=4))
RE(mv(beamline.autoUpdate, 1))
best_image = agent.best.bl_det_image
RE(agent.go_to_best())
if len(best_image.shape) < 2:
    best_image = np.reshape(best_image, (300, 400))
plt.imshow(best_image)
#plt.savefig("best_image.png")
# agent.plot_objectives(axes=(2, 3))
agent.plot_objectives(axes=(0, 1))
#plt.savefig("objectives.png")