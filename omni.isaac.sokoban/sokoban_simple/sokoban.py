import gc
import numpy as np
import asyncio
import random
import cv2

import carb
import omni.kit.commands
import omni.kit.editor
import omni.ext
import omni.appwindow
import omni.kit.ui
import omni.kit.settings

from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.manip import _manip
from omni.physx import _physx
from omni.physx.scripts import physicsUtils, utils
from omni.isaac.utils.scripts.scene_utils import setUpZAxis, SetupPhysics, CreateBackground
import omni.syntheticdata

from pxr import Sdf, Gf, PhysicsSchema, UsdGeom, PhysicsSchemaTools
from .utils.kaya import Kaya



EXTENSION_NAME = "Sokoban_Simple"


class Extension(omni.ext.IExt):
    def on_startup(self):
        """Initialize extension and UI elements
        """
        self._editor = omni.kit.editor.get_editor_interface()
        self._usd_context = omni.usd.get_context()
        self._stage = self._usd_context.get_stage()
        self._window = omni.kit.ui.Window(
            EXTENSION_NAME,
            300,
            200,
            menu_path="Isaac Robotics/RL/" + EXTENSION_NAME,
            open=True,
            dock=omni.kit.ui.DockPreference.LEFT_BOTTOM,
        )

        self._dc = _dynamic_control.acquire_dynamic_control_interface()

        self._load_kaya_btn = self._window.layout.add_child(omni.kit.ui.Button("Load Environment"))
        self._load_kaya_btn.set_clicked_fn(self._on_environment_setup)
        self._load_kaya_btn.tooltip = omni.kit.ui.Label("Reset the stage and load the kaya environment")

        self._train_btn = self._window.layout.add_child(omni.kit.ui.Button("Train"))
        self._train_btn.set_clicked_fn(self._train)
        self._train_btn.tooltip = omni.kit.ui.Label("Train Kaya")
        self._train_btn.enabled = False

        self._capture_btn = self._window.layout.add_child(omni.kit.ui.Button("Capture"))
        self._capture_btn.set_clicked_fn(self._capture)
        self._capture_btn.tooltip = omni.kit.ui.Label("Capture Video")
        self._capture_btn.enabled = False
        
        self.kaya = None

        self._settings = omni.kit.settings.get_settings_interface()
        self._settings.set("/persistent/physics/updateToUsd", False)
        self._settings.set("/persistent/physics/useFastCache", True)

        self._manip = _manip.acquire()
        self._vel_target = np.zeros(3)

        self._is_train = False
        self._is_capture = False

        self.path = "/home/axellkir/Documents/Isaac/Captures"
        self._time = 0.0

        print("Kaya Preview Startup Complete")

    async def _create_kaya(self, task):
        done, pending = await asyncio.wait({task})
        if task in done:
            print("Loading Kaya Enviornment")
            # Create empty scene and camera
            self._editor.set_camera_position("/OmniverseKit_Persp", -300, 300, 100, True)
            self._editor.set_camera_target("/OmniverseKit_Persp", 0, 0, 0, True)
            self._stage = self._usd_context.get_stage()
            asset_path = "omni:/Isaac/Robots/Kaya"
            kaya_usd = asset_path + "/kaya.usd"
            speed_gain = 10.0


            setUpZAxis(self._stage)
            SetupPhysics(self._stage)
            PhysicsSchemaTools.addGroundPlane(self._stage, "/physics/groundPlane", "Z", 4000, Gf.Vec3f(-100, -100, -0.001), Gf.Vec3f(0.0))
            # add_ground_plane(self._stage, "/physics/ground_plane", axis='Z', size=[10000, 10000], position=[-1000, -1000, 0], color=[0, 0, 0])

            # Load Kaya model
            self.kaya = Kaya(
                stage=self._stage, dc=self._dc, usd_path=kaya_usd, prim_path="/World/kaya", speed_gain=speed_gain
            )
            UsdGeom.XformCommonAPI(self.kaya.prim).SetTranslate([0, 0, 10])
            # Create camera for Kaya
            self.realsense_camera = self._stage.DefinePrim(f"/World/kaya/base_link/Kaya_Body/RealSense_Bridge_ASM_1/realsense/realsense_camera", "Camera")
            vpi = omni.kit.viewport.get_viewport_interface()
            vpi.get_viewport_window().set_active_camera(str(self.realsense_camera.GetPath()))

            xform_api = UsdGeom.XformCommonAPI(self.realsense_camera)
            camera_api = UsdGeom.Camera(self.realsense_camera)
            camera_api.CreateFocalLengthAttr(6)
            xform_api.SetRotate((0, 180, 0))
            xform_api.SetTranslate((0, 5, 6))

            # Load Environment
            CreateBackground(
                self._stage,
                "omni:/Isaac/Environments/Simple_Warehouse/warehouse.usd",
                # "omni:/Isaac/Environments/Grid/gridroom_curved.usd",
                background_path="/World/warehouse",
                offset=Gf.Vec3d(700, -3600, 0)
            )


            # Create cube and place
            self.cube_prim = self._stage.DefinePrim(f"/World/Cube", "Cube")
            # result, path = omni.kit.commands.execute("CreateShapesPrimCommand", prim_type="Cube")
            # self.cube_prim = self._stage.GetPrimAtPath("/World/Cube")

            UsdGeom.XformCommonAPI(self.cube_prim).SetTranslate((-100, 200, 2))
            UsdGeom.XformCommonAPI(self.cube_prim).SetScale((1.5, 1.5, 1.5))
            colorAttr = UsdGeom.Gprim.Get(self._stage, "/World/Cube").GetDisplayColorAttr()
            colorAttr.Set([(1.0, 0.0, 0.0)])
            utils.setRigidBody(self.cube_prim, "convexHull", False)

            self.place_prim = self._stage.DefinePrim(f"/World/place", "Cube")
            UsdGeom.XformCommonAPI(self.place_prim).SetTranslate((-400, 500, -0.95))
            UsdGeom.XformCommonAPI(self.place_prim).SetScale((50, 50, 1))
            colorAttr = UsdGeom.Gprim.Get(self._stage, "/World/place").GetDisplayColorAttr()
            colorAttr.Set([(1.0, 0.0, 0.0)])
            #
            # Synthetic Data Generatorscc
            self.sd = omni.syntheticdata._syntheticdata
            self.sdi = self.sd.acquire_syntheticdata_interface()

            # Unlock Buttons
            self._train_btn.enabled = True
            self._capture_btn.enabled = True

            # start stepping after kaya is created
            self._editor_event_subscription = self._editor.subscribe_to_update_events(self._on_editor_step)

    
    def _train(self, widget):
        self._editor.play()
        if self._is_train:
            self._is_train = False
        else:
            self._is_train = True

    def _capture(self, widget):
        self._editor.play()
        self.video_data = []
        self.depth_data = []
        self._time = 0
        if self._is_capture:
            self._is_capture= False
        else:
            self._is_capture = True
        


    def _save_video(self):
        out = cv2.VideoWriter(self.path + '/sokoban_test.avi',cv2.VideoWriter_fourcc(*'XVID'), 60, (1280, 720))
        for i in range(len(self.video_data)):
            out.write(self.video_data[i])
        out.release()
        return 0
    
    def _save_depth(self):
        out = cv2.VideoWriter(self.path + '/sokoban_depth.avi',cv2.VideoWriter_fourcc(*'XVID'), 60, (1280, 720), 0)
        for i in range(len(self.depth_data)):
            frame = (self.depth_data[i] - np.min(self.depth_data[i]))/(np.max(self.depth_data[i]) - np.min(self.depth_data[i]))
            out.write((frame * 255).astype(np.uint8).reshape((720, 1280, 1)))
        out.release()
        return 0

    def _get_sensor_data(self, sensor, dtype):
        width = self.sdi.get_sensor_width(sensor)
        height = self.sdi.get_sensor_height(sensor)
        row_size = self.sdi.get_sensor_row_size(sensor)

        get_sensor = {
                "uint32": self.sdi.get_sensor_host_uint32_texture_array,
                "float": self.sdi.get_sensor_host_float_texture_array,
        }
        return get_sensor[dtype](sensor, width, height, row_size)
            
    def _on_editor_step(self, step):
        # """Update kaya physics once per step"""
        
        # change_direction_steps = 1
        if self._is_capture:
            print(self._time)
            print(self.cube_prim.GetAttribute("xformOp:translate").Get())
            data = self._get_sensor_data(self.sd.SensorType.Rgb, "uint32")
            depth_data = self._get_sensor_data(self.sd.SensorType.Depth, "float")
            image = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape, -1)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.depth_data.append(depth_data)
            self.video_data.append(image)
            self._time += 1.0 / 60.0
            if self._time < change_direction_steps:
                self.kaya.move((4, 0, 0))
            elif self._time < 1.8*change_direction_steps:
                self.kaya.move((0, 4, 0))
            elif self._time < 4.85*change_direction_steps:
                self.kaya.move((4, 0, 0))
            elif self._time < 5*change_direction_steps:
                self.kaya.move((0, 0, 1))    
            elif self._time < 7.2*change_direction_steps:
                self.kaya.move((4, 0, 0))
            else:
                self.kaya.move((0, 0, 0))
                data = self._get_sensor_data(self.sd.SensorType.Rgb, "uint32")
                image = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape, -1)
                depth_data = self._get_sensor_data(self.sd.SensorType.Depth, "float")
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.depth_data.append(depth_data)
                self.video_data.append(image)
                self._save_video()
                self._save_depth()
                print("Video Saved")
                self._is_capture = False




    def _on_environment_setup(self, widget):
        # wait for new stage before creating kaya
        task = asyncio.ensure_future(omni.kit.asyncapi.new_stage())
        asyncio.ensure_future(self._create_kaya(task))


    def on_shutdown(self):
        """Cleanup objects on extension shutdown
        """
        print("Shutting down Kaya Preview")

        self._manip.unbind_gamepad()
        self._editor.stop()
        self.kaya = None
        self._window = None
        gc.collect()

