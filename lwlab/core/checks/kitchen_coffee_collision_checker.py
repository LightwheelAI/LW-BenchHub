from lwlab.core.checks.base_checker import BaseChecker
import torch
from lwlab.core.models.fixtures.fixture_types import FixtureType
from lwlab.utils.object_utils import check_contact


class KitchenCoffeeCollisionChecker(BaseChecker):
    type = "kitchen_coffee_collision"

    def __init__(self, warning_on_screen=False):
        super().__init__(warning_on_screen)
        self._init_state()

    def _init_state(self):
        self._coffee_collision_warning_frame_count = 0
        self._coffee_collision_warning_text = ""
        self._coffee_collision_counts = 0

    def reset(self):
        self._init_state()

    def _check(self, env):
        return self._check_collision(env)

    def _check_collision(self, env):
        """
        Check if the coffee cup collides with the coffee machine.
        Calculates the collision status and the z-axis position.

        Returns:
            dict: Dictionary containing total collision times.
        """
        self.coffee_machine = env.cfg.isaac_arena_env.task.get_fixture(FixtureType.COFFEE_MACHINE)

        if self._coffee_collision_counts is None:
            self._coffee_collision_counts = 0

        if self._coffee_collision_warning_frame_count is not None and 50 > self._coffee_collision_warning_frame_count > 0:
            self._coffee_collision_warning_frame_count += 1
        else:
            self._coffee_collision_warning_frame_count = 0
            self._coffee_collision_warning_text = ""

        cup_coffee_collision = False

        cup_coffee_collision = check_contact(env, env.cfg.isaac_arena_env.task.objects["obj"], self.coffee_machine)
        z_checks = env.cfg.isaac_arena_env.task.coffee_machine.get_z_checks()
        for z_check in z_checks:
            if not z_check and cup_coffee_collision and self._coffee_collision_warning_frame_count == 0:
                self._coffee_collision_warning_text = "kitchen_coffee_collision Warning: Collision between <<Mug>> and <<Coffee_Machine>> happens"
                self._coffee_collision_counts += 1
                self._coffee_collision_warning_frame_count += 1
                break

        cup_coffee_collision = False

        if self._coffee_collision_counts > 0:
            success = False
        else:
            success = True

        metrics = {}

        metrics["mug_coffeemachine_collision_times"] = self._coffee_collision_counts
        metrics["success"] = success

        result = {
            "success": success,
            "warning_text": self._coffee_collision_warning_text,
            "metrics": metrics
        }

        return result
