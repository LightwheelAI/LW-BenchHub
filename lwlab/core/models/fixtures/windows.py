from .fixture import Fixture
from .accessories import WallAccessory


class WindowProcBase(Fixture):

    @property
    def nat_lang(self):
        return "windows"


class WindowProc(WindowProcBase):
    pass


class Window(WallAccessory):

    @property
    def nat_lang(self):
        return "window"
