"""sakura.dispatch — Compute URI tag + Dispatcher abstractions."""
from sakura.dispatch.base import Dispatcher, Future, Result
from sakura.dispatch.compute import Compute
from sakura.dispatch.in_thread import InThreadDispatcher
from sakura.dispatch.local import LocalDispatcher
from sakura.dispatch.remote import RemoteDispatcher
from sakura.dispatch.zakuro import ZakuroDispatcher

__all__ = [
    "Compute",
    "Dispatcher",
    "Future",
    "InThreadDispatcher",
    "LocalDispatcher",
    "RemoteDispatcher",
    "Result",
    "ZakuroDispatcher",
]
