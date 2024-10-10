#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:32:06 2024

@author: rchernikov
"""

import numpy as np
import sys, os
# import matplotlib as mpl
# mpl.use("Agg")
from matplotlib import pyplot as plt
sys.path.append('/home/rchernikov/github/blop/src')
import blop
# sys.path.append(os.path.join('..', '..', '..')) 
from blop.utils import prepare_re_env as pre
from blop.sim import Beamline
from blop import DOF, Objective, Agent
from blop.digestion import beam_stats_digestion



plt.ion()

kwargs_re = dict(db_type='temp', root_dir=pre.DEFAULT_ROOT_DIR)
ret = pre.re_env(**kwargs_re)

RE = ret['RE']
db = ret['db']
bec = ret['bec']

# h_opt = 4.375
# dh = 0.075

h_opt = 0
dh = 5

R1, dR1 = 40000, 10000
R2, dR2 = 20000, 10000

bec.disable_plots()

beamline = Beamline(name="bl")

dofs = [
    # DOF(description="KBV downstream",
    #     device=beamline.kbv_dsv,
    #     search_domain=(h_opt-dh, h_opt+dh)),
    # DOF(description="KBV upstream",
    #     device=beamline.kbv_usv,
    #     search_domain=(-h_opt-dh, -h_opt+dh)),
    # DOF(description="KBH downstream",
    #     device=beamline.kbh_dsh,
    #     search_domain=(h_opt-dh, h_opt+dh)),
    # DOF(description="KBH upstream",
    #     device=beamline.kbh_ush,
    #     search_domain=(-h_opt-dh, -h_opt+dh)),

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
              # trust_domain=(15000, 1e6)),
    # Objective(name="bl_det_wid_x",
    #           target="min",
    #           transform="log",
    #           latent_groups=[("bl_kbh_dsh", "bl_kbh_ush")]),
    # Objective(name="bl_det_wid_y",
    #           target="min",
    #           transform="log",
    #           latent_groups=[("bl_kbv_dsv", "bl_kbv_usv")]),

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

agent = Agent(
    dofs=dofs,
    objectives=objectives,
    detectors=[beamline.det],
    digestion=beam_stats_digestion,
    digestion_kwargs={"image_key": "bl_det_image"},
    verbose=True,
    db=db,
    tolerate_acquisition_errors=False,
    enforce_all_objectives_valid=True,
    train_every=3,
)

(uid,) = RE(agent.learn("qr", n=30))

RE(agent.learn("qei", n=4, iterations=5))
plt.imshow(agent.best.bl_det_image)

# agent.plot_objectives(axes=(2, 3))
agent.plot_objectives(axes=(0, 1))