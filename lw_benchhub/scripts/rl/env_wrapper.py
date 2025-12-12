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

"""Wrapper to configure an environment instance to skrl environment.

The following example shows how to wrap an environment for skrl:

.. code-block:: python

    from policy.skrl.env_wrapper import SkrlVecEnvWrapper

    env = SkrlVecEnvWrapper(env, ml_framework="torch")  # or ml_framework="jax"

Or, equivalently, by directly calling the skrl library API as follows:

.. code-block:: python

    from skrl.envs.torch.wrappers import wrap_env  # for PyTorch, or...
    from skrl.envs.jax.wrappers import wrap_env    # for JAX

    env = wrap_env(env, wrapper="isaaclab")

"""

# needed to import for type hinting: Agent | list[Agent]
from __future__ import annotations

from typing import Literal

from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedRLEnv

"""
Vectorized environment wrapper.
"""


def SkrlVecEnvWrapper(
    env: ManagerBasedRLEnv | DirectRLEnv | DirectMARLEnv,
    ml_framework: Literal["torch", "jax", "jax-numpy"] = "torch",
    wrapper: Literal["auto", "isaaclab", "isaaclab-single-agent", "isaaclab-multi-agent"] = "isaaclab",
):
    """Wraps around Isaac Lab environment for skrl.

    This function wraps around the Isaac Lab environment. Since the wrapping
    functionality is defined within the skrl library itself, this implementation
    is maintained for compatibility with the structure of the extension that contains it.
    Internally it calls the :func:`wrap_env` from the skrl library API.

    Args:
        env: The environment to wrap around.
        ml_framework: The ML framework to use for the wrapper. Defaults to "torch".
        wrapper: The wrapper to use. Defaults to "isaaclab": leave it to skrl to determine if the environment
            will be wrapped as single-agent or multi-agent.

    Raises:
        ValueError: When the environment is not an instance of any Isaac Lab environment interface.
        ValueError: If the specified ML framework is not valid.

    Reference:
        https://skrl.readthedocs.io/en/latest/api/envs/wrapping.html
    """
    # check that input is valid
    if (
        not isinstance(env.unwrapped, ManagerBasedRLEnv)
        and not isinstance(env.unwrapped, DirectRLEnv)
        and not isinstance(env.unwrapped, DirectMARLEnv)
    ):
        raise ValueError(
            "The environment must be inherited from ManagerBasedRLEnv, DirectRLEnv or DirectMARLEnv. Environment type:"
            f" {type(env)}"
        )

    # import statements according to the ML framework
    if ml_framework.startswith("torch"):
        from skrl.envs.wrappers.torch import wrap_env
    elif ml_framework.startswith("jax"):
        from skrl.envs.wrappers.jax import wrap_env
    else:
        ValueError(
            f"Invalid ML framework for skrl: {ml_framework}. Available options are: 'torch', 'jax' or 'jax-numpy'"
        )

    # wrap and return the environment
    return wrap_env(env, wrapper)
