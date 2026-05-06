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

# G1 pipelines
register_pipeline(
    id="LWBenchhub-Autosim-G1OpenFridgePipeline-v0",
    entry_point=f"{__name__}.pipelines.open_fridge:OpenFridgePipeline",
    cfg_entry_point=f"{__name__}.pipelines.open_fridge:G1OpenFridgePipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-G1CoffeeSetupMugPipeline-v0",
    entry_point=f"{__name__}.pipelines.coffee_setup_mug:CoffeeSetupMugPipeline",
    cfg_entry_point=f"{__name__}.pipelines.coffee_setup_mug:G1CoffeeSetupMugPipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-G1CheesyBreadPipeline-v0",
    entry_point=f"{__name__}.pipelines.cheesy_bread:CheesyBreadPipeline",
    cfg_entry_point=f"{__name__}.pipelines.cheesy_bread:G1CheesyBreadPipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-G1DessertUpgradePipeline-v0",
    entry_point=f"{__name__}.pipelines.dessert_upgrade:DessertUpgradePipeline",
    cfg_entry_point=f"{__name__}.pipelines.dessert_upgrade:G1DessertUpgradePipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-G1KettleBoilingPipeline-v0",
    entry_point=f"{__name__}.pipelines.kettle_boiling:KettleBoilingPipeline",
    cfg_entry_point=f"{__name__}.pipelines.kettle_boiling:G1KettleBoilingPipelineCfg",
)
