import gymnasium as gym
from lwlab.core.tasks.base import LwLabTaskBase

gym.register(
    id="Robocasa-Task-L90K3PutTheFryingPanOnTheStove",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90_put_on_stove:L90K3PutTheFryingPanOnTheStove",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90K3PutTheMokaPotOnTheStove",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90_put_on_stove:L90K3PutTheMokaPotOnTheStove",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90K8PutTheRightMokaPotOnTheStove",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90_put_on_stove:L90K8PutTheRightMokaPotOnTheStove",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S2PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_book_in_caddy:L90S2PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S2PickUpTheBookAndPlaceItInTheBackCompartmentOfTheCaddy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_book_in_caddy:L90S2PickUpTheBookAndPlaceItInTheBackCompartmentOfTheCaddy",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S4PickUpTheBookOnTheLeftAndPlaceItOnTopOfTheShelf",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_book_in_caddy:L90S4PickUpTheBookOnTheLeftAndPlaceItOnTopOfTheShelf",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S4PickUpTheBookOnTheRightAndPlaceItOnTheCabinetShelf",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_book_in_caddy:L90S4PickUpTheBookOnTheRightAndPlaceItOnTheCabinetShelf",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L10S1PickUpTheBookAndPlaceItInTheBackCompartmentOfTheCaddy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_book_in_caddy:L10S1PickUpTheBookAndPlaceItInTheBackCompartmentOfTheCaddy",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S3PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_book_in_caddy:L90S3PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-LGPutTheBowlOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_goal_tasks:LGPutTheBowlOnThePlate"},
)

gym.register(
    id="Robocasa-Task-LGPutTheCreamCheeseInTheBowl",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_goal_tasks:LGPutTheCreamCheeseInTheBowl"},
)

gym.register(
    id="Robocasa-Task-L10L2PutBothTheCreamCheeseBoxAndTheButterInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_10_put_in_basket:L10L2PutBothTheCreamCheeseBoxAndTheButterInTheBasket"},
)

gym.register(
    id="Robocasa-Task-L10L2PutBothTheAlphabetSoupAndTheTomatoSauceInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_10_put_in_basket:L10L2PutBothTheAlphabetSoupAndTheTomatoSauceInTheBasket"},
)

gym.register(
    id="Robocasa-Task-L10L1PutBothTheAlphabetSoupAndTheCreamCheeseBoxInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_10_put_in_basket:L10L1PutBothTheAlphabetSoupAndTheCreamCheeseBoxInTheBasket"},
)

gym.register(
    id="Robocasa-Task-L90S3PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_book_in_caddy:L90S3PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S2PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_book_in_caddy:L90S2PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90L3PickUpTheCreamCheeseAndPutItInTheTray",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_cheese_to_tray:L90L3PickUpTheCreamCheeseAndPutItInTheTray",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S1PickUpTheYellowAndWhiteMugAndPlaceItToTheRightOfTheCaddy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy:L90S1PickUpTheYellowAndWhiteMugAndPlaceItToTheRightOfTheCaddy",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90K10PutTheButterAtTheBackInTheTopDrawerOfTheCabinetAndCloseIt",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it:L90K10PutTheButterAtTheBackInTheTopDrawerOfTheCabinetAndCloseIt",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90K1PutTheBlackBowlOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_the_black_bowl_on_the_plate:L90K1PutTheBlackBowlOnThePlate",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90K2StackTheBlackBowlAtTheFrontOnTheBlackBowlInTheMiddle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle:L90K2StackTheBlackBowlAtTheFrontOnTheBlackBowlInTheMiddle",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90L5PutTheYellowAndWhiteMugOnTheRightPlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_the_yellow_and_white_mug_on_the_right_plate:L90L5PutTheYellowAndWhiteMugOnTheRightPlate",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90L5PutTheWhiteMugOnTheLeftPlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_the_yellow_and_white_mug_on_the_right_plate:L90L5PutTheWhiteMugOnTheLeftPlate",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90K4PutTheWineBottleOnTheWineRack",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_the_wine_bottle_on_the_wine_rack:L90K4PutTheWineBottleOnTheWineRack",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90L4StackTheLeftBowlOnTheRightBowlAndPlaceThemInTheTray",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray:L90L4StackTheLeftBowlOnTheRightBowlAndPlaceThemInTheTray",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90L4PickUpTheChocolatePuddingAndPutItInTheTray",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_chocolate_to_tray:L90L4PickUpTheChocolatePuddingAndPutItInTheTray",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90L1PickUpTheKetchupAndPutItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_put_in_basket:L90L1PickUpTheKetchupAndPutItInTheBasket"},
)

gym.register(
    id="Robocasa-Task-L90L1PickUpTheCreamCheeseBoxAndPutItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_put_in_basket:L90L1PickUpTheCreamCheeseBoxAndPutItInTheBasket"},
)

gym.register(
    id="Robocasa-Task-L90L2PickUpTheTomatoSauceAndPutItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_put_in_basket:L90L2PickUpTheTomatoSauceAndPutItInTheBasket"},
)

gym.register(
    id="Robocasa-Task-L90L2PickUpTheAlphabetSoupAndPutItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_put_in_basket:L90L2PickUpTheAlphabetSoupAndPutItInTheBasket"},
)

gym.register(
    id="Robocasa-Task-L90L2PickUpTheOrangeJuiceAndPutItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_put_in_basket:L90L2PickUpTheOrangeJuiceAndPutItInTheBasket"},
)

gym.register(
    id="Robocasa-Task-L90L1PickUpTheTomatoSauceAndPutItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_put_in_basket:L90L1PickUpTheTomatoSauceAndPutItInTheBasket"},
)

gym.register(
    id="Robocasa-Task-L90K5PutTheBlackBowlInTheTopDrawerOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_drawer_tasks:L90K5PutTheBlackBowlInTheTopDrawerOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-L90K1OpenTheTopDrawerOfTheCabinetAndPutTheBowlInIt",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_drawer_tasks:L90K1OpenTheTopDrawerOfTheCabinetAndPutTheBowlInIt"},
)

gym.register(
    id="Robocasa-Task-L90K5CloseTheTopDrawerOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_drawer_tasks:L90K5CloseTheTopDrawerOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-L90K10CloseTheTopDrawerOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_drawer_tasks:L90K10CloseTheTopDrawerOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-L90K4CloseTheBottomDrawerOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_drawer_tasks:L90K4CloseTheBottomDrawerOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-L90K4PutTheBlackBowlInTheBottomDrawerOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_drawer_tasks:L90K4PutTheBlackBowlInTheBottomDrawerOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-L90K4PutTheWineBottleInTheBottomDrawerOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_drawer_tasks:L90K4PutTheWineBottleInTheBottomDrawerOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-L90K4PutTheBlackBowlOnTopOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_drawer_tasks:L90K4PutTheBlackBowlOnTopOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-L10K4PutTheBlackBowlInTheBottomDrawerOfTheCabinetAndCloseIt",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_drawer_tasks:L10K4PutTheBlackBowlInTheBottomDrawerOfTheCabinetAndCloseIt"},
)

gym.register(
    id="Robocasa-Task-L90K4CloseTheBottomDrawerOfTheCabinetAndOpenTheTopDrawer",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_drawer_tasks:L90K4CloseTheBottomDrawerOfTheCabinetAndOpenTheTopDrawer"},
)


gym.register(
    id="Robocasa-Task-L90K10CloseTheTopDrawerOfTheCabinetAndPutTheBlackBowlOnTopOfIt",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_drawer_tasks:L90K10CloseTheTopDrawerOfTheCabinetAndPutTheBlackBowlOnTopOfIt"},
)

gym.register(
    id="Robocasa-Task-L90K10PutTheBlackBowlInTheTopDrawerOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_drawer_tasks:L90K10PutTheBlackBowlInTheTopDrawerOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-L90K10PutTheButterAtTheFrontInTheTopDrawerOfTheCabinetAndCloseIt",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_drawer_tasks:L90K10PutTheButterAtTheFrontInTheTopDrawerOfTheCabinetAndCloseIt"},
)

gym.register(
    id="Robocasa-Task-L90K10PutTheChocolatePuddingInTheTopDrawerOfTheCabinetAndCloseIt",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_drawer_tasks:L90K10PutTheChocolatePuddingInTheTopDrawerOfTheCabinetAndCloseIt"},
)

gym.register(
    id="Robocasa-Task-LGPutTheWineBottleOnTopOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_goal_tasks:LGPutTheWineBottleOnTopOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-LGOpenTheTopDrawerAndPutTheBowlInside",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_goal_tasks:LGOpenTheTopDrawerAndPutTheBowlInside"},
)

gym.register(
    id="Robocasa-Task-LGOpenTheMiddleDrawerOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_goal_tasks:LGOpenTheMiddleDrawerOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-LGPutTheBowlOnTopOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_goal_tasks:LGPutTheBowlOnTopOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-LGOpenTopDrawerOfCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_goal_tasks:LGOpenTopDrawerOfCabinet"},
)

gym.register(
    id="Robocasa-Task-LGPushThePlateToTheFrontOfTheStove",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_goal_tasks:LGPushThePlateToTheFrontOfTheStove"},
)

gym.register(
    id="Robocasa-Task-LGPutTheBowlOnTheStove",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_goal_tasks:LGPutTheBowlOnTheStove"},
)

gym.register(
    id="Robocasa-Task-LGPutTheWineBottleOnTheRack",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_goal_tasks:LGPutTheWineBottleOnTheRack"},
)

gym.register(
    id="Robocasa-Task-LGTurnOnTheStove",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_goal_tasks:LGTurnOnTheStove"},
)

gym.register(
    id="Robocasa-Task-L10L6PutWhiteMugOnPlateAndPutChocolatePuddingToRightPlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_10_MugOnAndChocolateRightPlate:L10L6PutWhiteMugOnPlateAndPutChocolatePuddingToRightPlate"},
)

gym.register(
    id="Robocasa-Task-L90L6PutTheRedMugOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_MugOnPlate:L90L6PutTheRedMugOnThePlate"},
)

gym.register(
    id="Robocasa-Task-L90L5PutTheRedMugOnTheRightPlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_MugOnPlate:L90L5PutTheRedMugOnTheRightPlate"},
)

gym.register(
    id="Robocasa-Task-L90L6PutTheChocolatePuddingToTheLeftOfThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_living_room_put_chocolate_pudding_relative_to_plate:L90L6PutTheChocolatePuddingToTheLeftOfThePlate"},
)

gym.register(
    id="Robocasa-Task-L90L6PutTheChocolatePuddingToTheRightOfThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_living_room_put_chocolate_pudding_relative_to_plate:L90L6PutTheChocolatePuddingToTheRightOfThePlate"},
)

gym.register(
    id="Robocasa-Task-L90K5PutTheBlackBowlOnTopOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_black_bowl_and_plate:L90K5PutTheBlackBowlOnTopOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-L90K5PutTheKetchupInTheTopDrawerOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_black_bowl_and_plate:L90K5PutTheKetchupInTheTopDrawerOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-L90K5PutTheBlackBowlOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_black_bowl_and_plate:L90K5PutTheBlackBowlOnThePlate"},
)

gym.register(
    id="Robocasa-Task-L90S4PickUpTheBookInTheMiddleAndPlaceItOnTheCabinetShelf",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf:L90S4PickUpTheBookInTheMiddleAndPlaceItOnTheCabinetShelf"},
)
gym.register(
    id="Robocasa-Task-L90L3PickUpTheAlphabetSoupAndPutItInTheTray",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_up_the_alphabet_soup_and_put_it_in_the_tray:L90L3PickUpTheAlphabetSoupAndPutItInTheTray",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-LOPickUpTheAlphabetSoupAndPlaceItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_object_in_basket.put_alphabet_soup_in_basket:LOPickUpTheAlphabetSoupAndPlaceItInTheBasket",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-LOPutCreamCheeseInBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_object_in_basket.put_cream_cheese_in_basket:LOPutCreamCheeseInBasket",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-LOPickUpTheBbqSauceAndPlaceItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_object_in_basket.put_bbq_sauce_in_basket:LOPickUpTheBbqSauceAndPlaceItInTheBasket",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-LOPickUpTheChocolatePuddingAndPlaceItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_object_in_basket.put_chocolate_pudding_in_basket:LOPickUpTheChocolatePuddingAndPlaceItInTheBasket",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-LOPickUpTheButterAndPlaceItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_object_in_basket.put_butter_in_basket:LOPickUpTheButterAndPlaceItInTheBasket",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-LOPickUpTheKetchupAndPlaceItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_object_in_basket.put_ketchup_in_basket:LOPickUpTheKetchupAndPlaceItInTheBasket",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-LOPickUpTheMilkAndPlaceItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_object_in_basket.put_milk_in_basket:LOPickUpTheMilkAndPlaceItInTheBasket",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-LOPickUpTheOrangeJuiceAndPlaceItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_object_in_basket.put_orange_juice_in_basket:LOPickUpTheOrangeJuiceAndPlaceItInTheBasket",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-LOPickUpTheSaladDressingAndPlaceItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_object_in_basket.put_salad_dressing_in_basket:LOPickUpTheSaladDressingAndPlaceItInTheBasket",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-LOPickUpTheTomatoSauceAndPlaceItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_object_in_basket.put_tomato_sauce_in_basket:LOPickUpTheTomatoSauceAndPlaceItInTheBasket",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-L90L2PickUpTheButterAndPutItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_object_in_basket.put_butter_in_basket:L90L2PickUpTheButterAndPutItInTheBasket",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PutBlackBowlOnPlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_black_bowl_on_plate.put_black_bowl_on_plate:PutBlackBowlOnPlate",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-LSPickUpBlackBowlInTopDrawerOfWoodenCabinetAndPlaceItOnPlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_black_bowl_on_plate.pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate:LSPickUpBlackBowlInTopDrawerOfWoodenCabinetAndPlaceItOnPlate",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-LSPickUpTheBlackBowlOnTheCookieBoxAndPlaceItOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_black_bowl_on_plate.pick_up_black_bowl_on_cookitbox_and_place_it_on_the_plate:LSPickUpTheBlackBowlOnTheCookieBoxAndPlaceItOnThePlate",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-LSPickUpTheBlackBowlOnTheRamekinAndPlaceItOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_black_bowl_on_plate.pick_up_black_bowl_on_cookitbox_and_place_it_on_the_plate:LSPickUpTheBlackBowlOnTheRamekinAndPlaceItOnThePlate",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-LSPickUpTheBlackBowlOnTheStoveAndPlaceItOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_black_bowl_on_plate.pick_up_black_bowl_on_cookitbox_and_place_it_on_the_plate:LSPickUpTheBlackBowlOnTheStoveAndPlaceItOnThePlate",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-LSPickUpTheBlackBowlOnTheWoodenCabinetAndPlaceItOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_black_bowl_on_plate.pick_up_black_bowl_on_cookitbox_and_place_it_on_the_plate:LSPickUpTheBlackBowlOnTheWoodenCabinetAndPlaceItOnThePlate",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S4PickUpTheBookOnTheRightAndPlaceItUnderTheCabinetShelf",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf:L90S4PickUpTheBookOnTheRightAndPlaceItUnderTheCabinetShelf",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90L1PickUpTheAlphabetSoupAndPutItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_up_the_alphabet_soup_and_put_it_in_the_basket:L90L1PickUpTheAlphabetSoupAndPutItInTheBasket",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90L4PickUpTheBlackBowlOnTheLeftAndPutItInTheTray",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_black_bowl_to_tray:L90L4PickUpTheBlackBowlOnTheLeftAndPutItInTheTray",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S3PickUpTheRedMugAndPlaceItToTheRightOfTheCaddy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy:L90S3PickUpTheRedMugAndPlaceItToTheRightOfTheCaddy",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-LSPickUpBlackBowlBetweenPlateAndRamekinAndPlaceItOnPlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_black_bowl_on_plate.pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate:LSPickUpBlackBowlBetweenPlateAndRamekinAndPlaceItOnPlate",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-L10L5PutWhiteMugOnLeftPlateAndPutYellowAndWhiteMugOnRightPlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_the_yellow_and_white_mug_on_the_right_plate:L10L5PutWhiteMugOnLeftPlateAndPutYellowAndWhiteMugOnRightPlate",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-L90L3PickUpTheTomatoSauceAndPutItInTheTray",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_tomoto_sauce_on_tray:L90L3PickUpTheTomatoSauceAndPutItInTheTray"},
)

gym.register(
    id="Robocasa-Task-L90L3PickUpTheKetchupAndPutItInTheTray",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_ketchup_on_tray:L90L3PickUpTheKetchupAndPutItInTheTray"},
)

gym.register(
    id="Robocasa-Task-L90K2PutTheMiddleBlackBowlOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_black_bowl_and_plate:L90K2PutTheMiddleBlackBowlOnThePlate"},
)

gym.register(
    id="Robocasa-Task-L90K2PutTheBlackBowlAtTheFrontOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_black_bowl_and_plate:L90K2PutTheBlackBowlAtTheFrontOnThePlate"},
)

gym.register(
    id="Robocasa-Task-L90K2StackTheMiddleBlackBowlOnTheBackBlackBowl",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_black_bowl_and_plate:L90K2StackTheMiddleBlackBowlOnTheBackBlackBowl"},
)

gym.register(
    id="Robocasa-Task-L90K2PutTheBlackBowlAtTheBackOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_black_bowl_and_plate:L90K2PutTheBlackBowlAtTheBackOnThePlate"},
)

gym.register(
    id="Robocasa-Task-L90K2OpenTheTopDrawerOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_black_bowl_and_plate:L90K2OpenTheTopDrawerOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-L90K7PutTheWhiteBowlOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_white_bowl_and_plate:L90K7PutTheWhiteBowlOnThePlate"},
)

gym.register(
    id="Robocasa-Task-L90K7PutTheWhiteBowlToTheRightOfThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_white_bowl_and_plate:L90K7PutTheWhiteBowlToTheRightOfThePlate"},
)

gym.register(
    id="Robocasa-Task-L90K7OpenTheMicrowave",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_white_bowl_and_plate:L90K7OpenTheMicrowave"},
)

gym.register(
    id="Robocasa-Task-L90L4PickUpTheSaladDressingAndPutItInTheTray",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_PnPSaladDressingOnTray:L90L4PickUpTheSaladDressingAndPutItInTheTray"},
)

gym.register(
    id="Robocasa-Task-L90K3TurnOnTheStove",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.trun_on_stove:L90K3TurnOnTheStove"},
)

gym.register(
    id="Robocasa-Task-L90K9TurnOnTheStove",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.trun_on_stove:L90K9TurnOnTheStove"},
)

gym.register(
    id="Robocasa-Task-L90K3TurnOnTheStoveAndPutTheFryingPanOnIt",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.trun_on_stove:L90K3TurnOnTheStoveAndPutTheFryingPanOnIt"},
)

gym.register(
    id="Robocasa-Task-L90K9TurnOnTheStoveAndPutTheFryingPanOnIt",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.trun_on_stove:L90K9TurnOnTheStoveAndPutTheFryingPanOnIt"},
)


gym.register(
    id="Robocasa-Task-L10K3TurnOnTheStoveAndPutTheMokaPotOnIt",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.trun_on_stove:L10K3TurnOnTheStoveAndPutTheMokaPotOnIt"},
)

gym.register(
    id="Robocasa-Task-L90K1OpenTheBottomDrawerOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.open_the_bottom_drawer_of_the_cabinet:L90K1OpenTheBottomDrawerOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-L90K1OpenTheTopDrawerOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.open_the_bottom_drawer_of_the_cabinet:L90K1OpenTheTopDrawerOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-L90K2PutTheMiddleBlackBowlOnTopOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.put_the_middle_black_bowl_on_top_of_the_cabinet:L90K2PutTheMiddleBlackBowlOnTopOfTheCabinet"},
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90K6PutTheYellowAndWhiteMugToTheFrontOfTheWhiteMug",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_mug_placement:L90K6PutTheYellowAndWhiteMugToTheFrontOfTheWhiteMug"},
)

gym.register(
    id="Robocasa-Task-L90K6CloseTheMicrowave",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_mug_placement:L90K6CloseTheMicrowave"},
)

gym.register(
    id="Robocasa-Task-L10K6PutTheYellowAndWhiteMugInTheMicrowaveAndCloseIt",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_mug_placement:L10K6PutTheYellowAndWhiteMugInTheMicrowaveAndCloseIt"},
)

gym.register(
    id="Robocasa-Task-L90L5PutTheRedMugOnTheLeftPlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.libero_90_mug_placement:L90L5PutTheRedMugOnTheLeftPlate"},
)
gym.register(
    id="Robocasa-Task-LSPickUpTheBlackBowlFromTableCenterAndPlaceItOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_black_bowl_on_plate.pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate:LSPickUpTheBlackBowlFromTableCenterAndPlaceItOnThePlate",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-LSPickUpTheBlackBowlNextToTheCookieBoxAndPlaceItOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_black_bowl_on_plate.pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate:LSPickUpTheBlackBowlNextToTheCookieBoxAndPlaceItOnThePlate",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-LSPickUpTheBlackBowlNextToThePlateAndPlaceItOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_black_bowl_on_plate.pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate:LSPickUpTheBlackBowlNextToThePlateAndPlaceItOnThePlate",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-LSPickUpTheBlackBowlNextToTheRamekinAndPlaceItOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.put_black_bowl_on_plate.pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate:LSPickUpTheBlackBowlNextToTheRamekinAndPlaceItOnThePlate",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90K9PutTheWhiteBowlOnTopOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.put_the_white_bowl_on_top_of_the_cabinet:L90K9PutTheWhiteBowlOnTopOfTheCabinet"},
)

gym.register(
    id="Robocasa-Task-L90L6PutTheWhiteMugOnThePlate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_put_the_white_mug_on_the_plate:L90L6PutTheWhiteMugOnThePlate",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90L3PickUpTheButterAndPutItInTheTray",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_pick_up_the_butter_and_put_it_in_the_tray:L90L3PickUpTheButterAndPutItInTheTray",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90L4StackTheRightBowlOnTheLeftBowlAndPlaceThemInTheTray",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray:L90L4StackTheRightBowlOnTheLeftBowlAndPlaceThemInTheTray",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S1PickUpTheBookAndPlaceItInTheFrontCompartmentOfTheCaddy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy:L90S1PickUpTheBookAndPlaceItInTheFrontCompartmentOfTheCaddy",
    },
    disable_env_checker=True,
)
gym.register(
    id="Robocasa-Task-L10K8PutBothMokaPotsOnTheStove",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_10.put_both_moka_pots_on_the_stove:L10K8PutBothMokaPotsOnTheStove",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90K9PutTheFryingPanOnTheCabinetShelf",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_put_the_frying_pan_on_the_cabinet_shelf:L90K9PutTheFryingPanOnTheCabinetShelf",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90K9PutTheFryingPanOnTopOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_put_the_frying_pan_on_top_of_the_cabinet:L90K9PutTheFryingPanOnTopOfTheCabinet",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90K9PutTheFryingPanUnderTheCabinetShelf",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_put_the_frying_pan_under_the_cabinet_shelf:L90K9PutTheFryingPanUnderTheCabinetShelf",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90L2PickUpTheMilkAndPutItInTheBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_pick_up_the_milk_and_put_it_in_basket:L90L2PickUpTheMilkAndPutItInTheBasket",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S1PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy:L90S1PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S1PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy:L90S1PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S2PickUpTheBookAndPlaceItInTheFrontCompartmentOfTheCaddy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy:L90S2PickUpTheBookAndPlaceItInTheFrontCompartmentOfTheCaddy",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S3PickUpTheBookAndPlaceItInTheFrontCompartmentOfTheCaddy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy:L90S3PickUpTheBookAndPlaceItInTheFrontCompartmentOfTheCaddy",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90S3PickUpTheWhiteMugAndPlaceItToTheRightOfTheCaddy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_pick_up_the_white_mug_and_place_it_in_the_right_compartment_of_the_caddy:L90S3PickUpTheWhiteMugAndPlaceItToTheRightOfTheCaddy",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90K1PutTheBlackBowlOnTopOfTheCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_put_the_black_bowl_on_top_of_the_cabinet:L90K1PutTheBlackBowlOnTopOfTheCabinet",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-L90K8TurnOffTheStove",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero_90.libero_90_turn_off_the_stove:L90K8TurnOffTheStove",
    },
    disable_env_checker=True,
)
