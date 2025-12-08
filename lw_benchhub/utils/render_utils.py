# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def optimize_rendering(env, args_cli):
    import carb
    settings = carb.settings.get_settings()
    # enable async rendering
    settings.set_bool("/app/asyncRendering", True)
    settings.set_bool("/app/asyncRenderingLowLatency", True)
    settings.set_bool("/app/asyncRendering", False)
    settings.set_bool("/app/asyncRenderingLowLatency", False)
    settings.set_bool("/app/asyncRendering", True)
    settings.set_bool("/app/asyncRenderingLowLatency", True)

    # use the USD / Fabric only for poses
    if not args_cli.disable_fabric:
        settings.set_bool("/physics/updateToUsd", False)
        settings.set_bool("/physics/updateParticlesToUsd", True)
        settings.set_bool("/physics/updateVelocitiesToUsd", False)
        settings.set_bool("/physics/updateForceSensorsToUsd", False)
        settings.set_bool("/physics/updateResidualsToUsd", False)
        settings.set_bool("/physics/outputVelocitiesLocalSpace", False)
        settings.set_bool("/physics/fabricUpdateTransformations", True)
        settings.set_bool("/physics/fabricUpdateVelocities", False)
        settings.set_bool("/physics/fabricUpdateForceSensors", False)
        settings.set_bool("/physics/fabricUpdateJointStates", False)
        settings.set_bool("/physics/fabricUpdateResiduals", False)
        settings.set_bool("/physics/fabricUseGPUInterop", True)

    # enable DLSS and performance optimization
    settings.set_bool("/rtx-transient/dlssg/enabled", True)
    settings.set_int("/rtx/post/dlss/execMode", 0)  # "Performance"
    settings.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)

    # TODO this option affects the rendering of dynamic spawned objects
    settings.set_bool("/app/renderer/skipMaterialLoading", False)
    settings.set_bool("/app/renderer/skipTextureLoading", False)

    # Setup timeline
    import omni.timeline as timeline
    timeline = timeline.get_timeline_interface()
    # Configure Kit to not wait for wall clock time to catch up between updates
    # This setting is effective only with Fixed time stepping
    timeline.set_play_every_frame(True)

    # enable fast mode and ensure fixed time stepping
    settings.set_bool("/app/player/useFastMode", True)
    settings.set_bool("/app/player/useFixedTimeStepping", True)

    # configure all run loops, disable rate limiting
    for run_loop in ["present", "main", "rendering_0"]:
        settings.set_bool(f"/app/runLoops/{run_loop}/rateLimitEnabled", False)
        settings.set_int(f"/app/runLoops/{run_loop}/rateLimitFrequency", 120)
        settings.set_bool(f"/app/runLoops/{run_loop}/rateLimitUseBusyLoop", False)

    # disable vertical sync to improve frame rate
    settings.set_bool("/app/vsync", False)
    settings.set_bool("/exts/omni.kit.renderer.core/present/enabled", False)

    # enable gpu dynamics
    if args_cli.device != "cpu":
        physics_context = env.sim.get_physics_context()
        physics_context.enable_gpu_dynamics(True)
        physics_context.set_broadphase_type("GPU")

    # settings.set_int("/persistent/physics/numThreads", 0)
    settings.set_float("/rtx/sceneDb/ambientLightIntensity", 0.0)
