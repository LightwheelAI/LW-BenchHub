from autosim import register_pipeline


register_pipeline(
    id="LWBenchhub-Autosim-CoffeeSetupMugPipeline-v0",
    entry_point=f"{__name__}.pipelines.coffee_setup_mug:CoffeeSetupMugPipeline",
    cfg_entry_point=f"{__name__}.pipelines.coffee_setup_mug:CoffeeSetupMugPipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-OpenFridgePipeline-v0",
    entry_point=f"{__name__}.pipelines.open_fridge:OpenFridgePipeline",
    cfg_entry_point=f"{__name__}.pipelines.open_fridge:OpenFridgePipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-CheesyBreadPipeline-v0",
    entry_point=f"{__name__}.pipelines.cheesy_bread:CheesyBreadPipeline",
    cfg_entry_point=f"{__name__}.pipelines.cheesy_bread:CheesyBreadPipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-CloseOvenPipeline-v0",
    entry_point=f"{__name__}.pipelines.close_oven:CloseOvenPipeline",
    cfg_entry_point=f"{__name__}.pipelines.close_oven:CloseOvenPipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-KettleBoilingPipeline-v0",
    entry_point=f"{__name__}.pipelines.kettle_boiling:KettleBoilingPipeline",
    cfg_entry_point=f"{__name__}.pipelines.kettle_boiling:KettleBoilingPipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-DessertUpgradePipeline-v0",
    entry_point=f"{__name__}.pipelines.dessert_upgrade:DessertUpgradePipeline",
    cfg_entry_point=f"{__name__}.pipelines.dessert_upgrade:DessertUpgradePipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-OpenToasterOvenDoorPipeline-v0",
    entry_point=f"{__name__}.pipelines.open_toaster_oven_door:OpenToasterOvenDoorPipeline",
    cfg_entry_point=f"{__name__}.pipelines.open_toaster_oven_door:OpenToasterOvenDoorPipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-CloseDrawerPipeline-v0",
    entry_point=f"{__name__}.pipelines.close_drawer:CloseDrawerPipeline",
    cfg_entry_point=f"{__name__}.pipelines.close_drawer:CloseDrawerPipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-PnPCounterToStovePipeline-v0",
    entry_point=f"{__name__}.pipelines.pnp_counter_to_stove:PnPCounterToStovePipeline",
    cfg_entry_point=f"{__name__}.pipelines.pnp_counter_to_stove:PnPCounterToStovePipelineCfg",
)
