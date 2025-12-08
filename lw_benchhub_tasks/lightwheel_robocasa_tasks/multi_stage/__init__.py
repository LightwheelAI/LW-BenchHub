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

import gymnasium as gym

gym.register(
    id="Robocasa-Task-DryDrinkware",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.washing_dishes.dry_drinkware:DryDrinkware"},
)

gym.register(
    id="Robocasa-Task-FoodCleanup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.clearing_table.food_cleanup:FoodCleanup"},
)

gym.register(
    id="Robocasa-Task-CheesyBread",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.making_toast.cheesy_bread:CheesyBread"},
)

gym.register(
    id="Robocasa-Task-SizeSorting",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.setting_the_table.size_sorting:SizeSorting"},
)

gym.register(
    id="Robocasa-Task-PrewashFoodAssembly",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.washing_fruits_and_vegetables.prewash_food_assembly:PrewashFoodAssembly"},
)

gym.register(
    id="Robocasa-Task-ArrangeTea",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.brewing.arrange_tea:ArrangeTea"},
)

gym.register(
    id="Robocasa-Task-SimmeringSauce",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.reheating_food.simmering_sauce:SimmeringSauce"},
)

gym.register(
    id="Robocasa-Task-HeatMultipleWater",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.boiling.heat_multiple_water:HeatMultipleWater"},
)

gym.register(
    id="Robocasa-Task-FryingPanAdjustment",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.frying.frying_pan_adjustment:FryingPanAdjustment"},
)

gym.register(
    id="Robocasa-Task-ArrangeBreadBasket",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.setting_the_table.arrange_bread_basket:ArrangeBreadBasket"},
)

gym.register(
    id="Robocasa-Task-PrepareToast",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.making_toast.prepare_toast:PrepareToast"},
)

gym.register(
    id="Robocasa-Task-StackBowlsInSink",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.washing_dishes.stack_bowls:StackBowlsInSink"},
)

gym.register(
    id="Robocasa-Task-StackBowlsInSink-Mimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.washing_dishes.stack_bowls:StackBowlsInSinkMimic"},
)

gym.register(
    id="Robocasa-Task-DryDishes",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.washing_dishes.dry_dishes:DryDishes"},
)

gym.register(
    id="Robocasa-Task-SteamInMicrowave",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.steaming_food.steam_in_microwave:SteamInMicrowave"},
)

gym.register(
    id="Robocasa-Task-SearingMeat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.frying.searing_meat:SearingMeat"},
)

gym.register(
    id="Robocasa-Task-WaffleReheat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reheating_food.waffle_reheat:WaffleReheat",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PrepForSanitizing",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.sanitize_surface.prep_for_sanitizing:PrepForSanitizing"},
)

gym.register(
    id="Robocasa-Task-PrepMarinatingMeat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.meat_preparation.prep_marinating_meat:PrepMarinatingMeat"},
)

gym.register(
    id="Robocasa-Task-PrepareCoffee",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.brewing.prepare_coffee:PrepareCoffee"}
)

gym.register(
    id="Robocasa-Task-PreSoakPan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.washing_dishes.pre_soak_pan:PreSoakPan"},
)

gym.register(
    id="Robocasa-Task-MicrowaveThawing",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.defrosting_food.microwave_thawing:MicrowaveThawing"}
)

gym.register(
    id="Robocasa-Task-ArrangeVegetables",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.chopping_food.arrange_vegetables:ArrangeVegetables"}
)

gym.register(
    id="Robocasa-Task-ArrangeVegetablesSimple",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.chopping_food.arrange_vegetables:ArrangeVegetablesSimple"}
)

gym.register(
    id="Robocasa-Task-CupcakeCleanup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.baking.cupcake_cleanup:CupcakeCleanup"},
)

gym.register(
    id="Robocasa-Task-SpicyMarinade",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.mixing_and_blending.spicy_marinade:SpicyMarinade"},
)

gym.register(
    id="Robocasa-Task-BowlAndCup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.clearing_table.bowl_and_cup:BowlAndCup"},
)

gym.register(
    id="Robocasa-Task-PanTransfer",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.serving_food.pan_transfer:PanTransfer"},
)

gym.register(
    id="Robocasa-Task-DefrostByCategory",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.defrosting_food.defrost_by_category:DefrostByCategory"},
)

gym.register(
    id="Robocasa-Task-RestockPantry",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.restocking_supplies.restock_pantry:RestockPantry"},
)

gym.register(
    id="Robocasa-Task-StockingBreakfastFoods",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.restocking_supplies.stocking_breakfast_foods:StockingBreakfastFoods"},
)

gym.register(
    id="Robocasa-Task-SetBowlsForSoup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.setting_the_table.set_bowls_for_soup:SetBowlsForSoup"},
)

gym.register(
    id="Robocasa-Task-ServeSteak",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.serving_food.serve_steak:ServeSteak"},
)

gym.register(
    id="Robocasa-Task-SnackSorting",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.tidying_cabinets_and_drawers.snack_sorting:SnackSorting"},
)

gym.register(
    id="Robocasa-Task-VeggieDipPrep",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.snack_preparation.veggie_dip_prep:VeggieDipPrep"},
)

gym.register(
    id="Robocasa-Task-BeverageSorting",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.restocking_supplies.beverage_sorting:BeverageSorting"},
)

gym.register(
    id="Robocasa-Task-FillKettle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.boiling.fill_kettle:FillKettle"}
)

gym.register(
    id="Robocasa-Task-OrganizeVegetables",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.chopping_food.organize_vegetables:OrganizeVegetables"}
)

gym.register(
    id="Robocasa-Task-BreadSetupSlicing",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.chopping_food.bread_setup_slicing:BreadSetupSlicing"}
)

gym.register(
    id="Robocasa-Task-MeatTransfer",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.chopping_food.meat_transfer:MeatTransfer"}
)

gym.register(
    id="Robocasa-Task-CondimentCollection",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.clearing_table.condiment_collection:CondimentCollection"}
)

gym.register(
    id="Robocasa-Task-DessertAssembly",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.clearing_table.dessert_assembly:DessertAssembly"}
)

gym.register(
    id="Robocasa-Task-CandleCleanup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.clearing_table.candle_cleanup:CandleCleanup"}
)

gym.register(
    id="Robocasa-Task-ThawInSink",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.defrosting_food.thaw_in_sink:ThawInSink",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-AssembleCookingArray",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.frying.assemble_cooking_array:AssembleCookingArray",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-BreadSelection",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.making_toast.bread_selection:BreadSelection",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-SweetSavoryToastSetup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.making_toast.sweet_savory_toast_setup:SweetSavoryToastSetup",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PrepForTenderizing",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.meat_preparation.prep_for_tenderizing:PrepForTenderizing",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-ColorfulSalsa",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixing_and_blending.colorful_salsa:ColorfulSalsa",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-DrinkwareConsolidation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.clearing_table.drinkware_consolidation:DrinkwareConsolidation"}
)

gym.register(
    id="Robocasa-Task-ClearClutter",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.washing_fruits_and_vegetables.clear_clutter:ClearClutter"},
)

gym.register(
    id="Robocasa-Task-DrainVeggies",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.washing_fruits_and_vegetables.drain_veggies:DrainVeggies"},
)

gym.register(
    id="Robocasa-Task-AfterwashSorting",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.washing_fruits_and_vegetables.afterwash_sorting:AfterwashSorting"},
)

gym.register(
    id="Robocasa-Task-SortingCleanup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.washing_dishes.sorting_cleanup:SortingCleanup"},
)

gym.register(
    id="Robocasa-Task-OrganizeCleaningSupplies",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.tidying_cabinets_and_drawers.organize_cleaning_supplies:OrganizeCleaningSupplies"},
)

gym.register(
    id="Robocasa-Task-MakeFruitBowl",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.snack_preparation.make_fruit_bowl:MakeFruitBowl"}
)

gym.register(
    id="Robocasa-Task-BreadAndCheese",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.snack_preparation.bread_and_cheese:BreadAndCheese"}
)

gym.register(
    id="Robocasa-Task-SetupFrying",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.frying.setup_frying:SetupFrying",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-MakeLoadedPotato",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reheating_food.make_loaded_potato:MakeLoadedPotato",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-WarmCroissant",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reheating_food.warm_croissant:WarmCroissant",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PrepareSoupServing",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.serving_food.prepare_soup_serving:PrepareSoupServing"},
)

gym.register(
    id="Robocasa-Task-DessertUpgrade",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.serving_food.dessert_upgrade:DessertUpgrade"},
)

gym.register(
    id="Robocasa-Task-CleanMicrowave",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.sanitize_surface.clean_microwave:CleanMicrowave"},
)

gym.register(
    id="Robocasa-Task-PastryDisplay",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.baking.pastry_display:PastryDisplay"}
)

gym.register(
    id="Robocasa-Task-OrganizeBakingIngredients",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.baking.organize_baking_ingredients:OrganizeBakingIngredients"},
)

gym.register(
    id="Robocasa-Task-KettleBoiling",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.brewing.kettle_boiling:KettleBoiling"}
)

gym.register(
    id="Robocasa-Task-QuickThaw",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.defrosting_food.quick_thaw:QuickThaw",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-RestockBowls",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.restocking_supplies.restock_bowls:RestockBowls"},
)

gym.register(
    id="Robocasa-Task-PlaceFoodInBowls",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.serving_food.place_food_in_bowls:PlaceFoodInBowls"},
)

gym.register(
    id="Robocasa-Task-DateNight",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.setting_the_table.date_night:DateNight"},
)

gym.register(
    id="Robocasa-Task-HeatMug",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reheating_food.heat_mug:HeatMug",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-CerealAndBowl",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.snack_preparation.cereal_and_bowl:CerealAndBowl"}
)

gym.register(
    id="Robocasa-Task-BeverageOrganization",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.setting_the_table.beverage_organization:BeverageOrganization"},
)

gym.register(
    id="Robocasa-Task-YogurtDelightPrep",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.snack_preparation.yogurt_delight_prep:YogurtDelightPrep"}
)

gym.register(
    id="Robocasa-Task-MultistepSteaming",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.steaming_food.multistep_steaming:MultistepSteaming",
    },
)

gym.register(
    id="Robocasa-Task-SeasoningSpiceSetup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.setting_the_table.seasoning_spice_setup:SeasoningSpiceSetup"},
)

gym.register(
    id="Robocasa-Task-VeggieBoil",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.boiling.veggie_boil:VeggieBoil"}
)

gym.register(
    id="Robocasa-Task-ClearingTheCuttingBoard",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.chopping_food.clearing_the_cutting_board:ClearingTheCuttingBoard"}
)

gym.register(
    id="Robocasa-Task-ClearingCleaningReceptacles",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.clearing_table.clearing_cleaning_receptacles:ClearingCleaningReceptacles"}
)

gym.register(
    id="Robocasa-Task-MealPrepStaging",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.frying.meal_prep_staging:MealPrepStaging"},
)

gym.register(
    id="Robocasa-Task-SetupJuicing",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixing_and_blending.setup_juicing:SetupJuicing",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-CountertopCleanup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.sanitize_surface.countertop_cleanup:CountertopCleanup"},
)

gym.register(
    id="Robocasa-Task-PushUtensilsToSink",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.sanitize_surface.push_utensils_to_sink:PushUtensilsToSink"},
)

gym.register(
    id="Robocasa-Task-WineServingPrep",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.serving_food.wine_serving_prep:WineServingPrep"},
)

gym.register(
    id="Robocasa-Task-SteamVegetables",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.steaming_food.steam_vegetables:SteamVegetables"},
)

gym.register(
    id="Robocasa-Task-DrawerUtensilSort",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.tidying_cabinets_and_drawers.drawer_utensil_sort:DrawerUtensilSort"},
)

gym.register(
    id="Robocasa-Task-PantryMishap",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.tidying_cabinets_and_drawers.pantry_mishap:PantryMishap"},
)

gym.register(
    id="Robocasa-Task-ShakerShuffle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.tidying_cabinets_and_drawers.shaker_shuffle:ShakerShuffle"},
)
