from typing import ContextManager
from . import keops as keops

def backend(name: str) -> ContextManager[None]: ...
