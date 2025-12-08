from enum import IntEnum


class FixtureType(IntEnum):
    """
    Enum for fixture types in lw_benchhub kitchen environments.
    """

    MICROWAVE = 1
    STOVE = 2
    OVEN = 3
    SINK = 4
    COFFEE_MACHINE = 5
    TOASTER = 6
    TOASTER_OVEN = 7
    FRIDGE = 8
    DISHWASHER = 9
    BLENDER = 10
    STAND_MIXER = 11
    ELECTRIC_KETTLE = 12
    STOOL = 13
    COUNTER = 14
    ISLAND = 15
    COUNTER_NON_CORNER = 16
    DINING_COUNTER = 17
    CABINET = 18
    CABINET_WITH_DOOR = 19
    CABINET_SINGLE_DOOR = 20
    CABINET_DOUBLE_DOOR = 21
    SHELF = 22
    DRAWER = 23
    TOP_DRAWER = 24
    WINDOW = 25
    DISH_RACK = 26
    COUNTER_NON_DINING = 27
    TABLE = 28
    YCB = 29
    DOOR = 30
    SIDETABLE = 31

    # AI2 FIXTURES
    BUTTON = 32
    SWITCH = 33
    BOOK = 34
    HAND_SOAP = 35
    DISH_SOAP = 36
    CART = 37
    COFFEE_POT = 38
    SOFA = 39
    TELEVISION = 40
    COFFEE_TABLE = 41
    WALL_ART = 42
    CARPET = 43
    TABLE_LAMP = 44
    FLOOR_LAMP = 45

    # libro fixtures
    BBQ_SOURCE = 46
    BOTTLE = 47
    DRAINER = 48
    MOKA_POT = 49
    SALAD_DRESSING = 50
    STORAGE_FURNITURE = 51
    KETCHUP = 52
    FAUCET = 53
    WINE_RACK = 54
    DECORATION = 55
    CABINET_MESH = 56

    # wall accessories
    SOCKET = 57
    UTENSIL_RACK = 58
    FLOOR_OBJ = 59
    FLOOR_LAYOUT = 60
    HOOD = 61
    DISHTOWEL = 62
    BATTERY = 63


# Mapping from FixtureType to layout registry names (strings used by floorplan_loader)
FIXTURE_TYPE_TO_REGISTRY_NAME = {
    FixtureType.MICROWAVE: "microwave",
    FixtureType.STOVE: "stove",
    FixtureType.OVEN: "oven",
    FixtureType.SINK: "sink",
    FixtureType.COFFEE_MACHINE: "coffee_machine",
    FixtureType.TOASTER: "toaster",
    FixtureType.TOASTER_OVEN: "toaster_oven",
    FixtureType.FRIDGE: "fridge_side_by_side",
    FixtureType.DISHWASHER: "dishwasher",
    FixtureType.BLENDER: "blender",
    FixtureType.STAND_MIXER: "stand_mixer",
    FixtureType.ELECTRIC_KETTLE: "electric_kettle",
    FixtureType.STOOL: "stool",
    FixtureType.COUNTER: "counter",
    FixtureType.COUNTER_NON_CORNER: "counter",
    FixtureType.DINING_COUNTER: "counter",
    FixtureType.COUNTER_NON_DINING: "counter",
    FixtureType.CABINET: "single_cabinet",
    FixtureType.CABINET_WITH_DOOR: "hinge_cabinet",
    FixtureType.CABINET_SINGLE_DOOR: "single_cabinet",
    FixtureType.CABINET_DOUBLE_DOOR: "hinge_cabinet",
    FixtureType.WINDOW: "window",
    FixtureType.DISH_RACK: "dish_rack",
    FixtureType.SOCKET: "socket",
    FixtureType.UTENSIL_RACK: "utensil_rack",
    FixtureType.HOOD: "hood",
}
