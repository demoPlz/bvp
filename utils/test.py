#!/usr/bin/env python3

# 1) Install dependencies:
#    pip install meshcat pinocchio

import pinocchio as pin
import sys
from os.path import dirname, join, abspath
from pinocchio.visualize import MeshcatVisualizer

if __name__ == "__main__":
    # Path to your URDF and (optional) mesh root
    urdf_file = "./assets/urdf/assembly/kinova3_robotiq_2f_85/kinova_gen3_robotiq_2f_85_left.urdf"
    mesh_root = dirname(abspath(urdf_file))

    # 2) Build the Pinocchio model from URDF
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_file,
        mesh_root,
        pin.JointModelFreeFlyer()  # or pin.JointModelRNEA() if no free-flyer
    )

    # 3) Create the MeshCat visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model)

    # 4) Launch the MeshCat server in your browser
    #    (or start `meshcat-server` in a terminal, then remove `open=True`)
    try:
        viz.initViewer(open=True)
    except ImportError as err:
        print("❌ Could not start MeshCat. Is `meshcat` installed?")
        print(err)
        sys.exit(1)

    # 5) Push the robot into the viewer
    viz.loadViewerModel()

    # 6) Display the "home" configuration so you see something
    q0 = pin.neutral(model)
    viz.display(q0)

    print("✅ URDF loaded. Check your browser for errors in the JS console.")
    input("Hit <Enter> to exit and shut down the server…")
