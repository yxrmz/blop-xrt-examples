# blop-xrt-examples

xrt-blop interoperability tests
1. Fix the path to _xrt_ in `xrt_beamline.py` under ```sys.path.append("PATH_TO_XRT")```. The test works only with [`xrt/new_glow`](https://github.com/kklmn/xrt/tree/new_glow) branch (until merged with `master`)
2. We use [pythonSoftIOC](https://github.com/DiamondLightSource/pythonSoftIOC) to spawn PVs dynamically, so
   `pip install softioc`
3. Navigate to `xrt/examples/withRaycing/_QookBeamlines` and run `xrt_beamline_epics.py` - this will spawn a virtual IOC.
4. Open the `xrt_screen.bob` in [Phoebus](https://github.com/ControlSystemStudio/phoebus), make sure it can see the PVs, if not - try adding localhost to the epics address list in the `settings.ini`:\
`org.phoebus.pv.ca/addr_list=127.0.0.1`\
`org.phoebus.pv.ca/auto_addr_list=no`\
Once all controls can connect to PVs, try generating some images pressing "Acquire"
6. Run `my_test_bl.py` - optimization process should modify the curvatures of the mirrors and update the image on the screen.
