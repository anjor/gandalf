"""Shared pytest fixtures."""

from typing import Callable, Iterator

import jax
import pytest

from krmhd import hermite


@pytest.fixture
def hermite_in_place(request, monkeypatch) -> Iterator[Callable[[bool], None]]:
    """Pin hermite.advance_in_place to one path: in place, or all at once with
    indirect parametrization to False. Returns a function that switches path.

    jit reads HERMITE_IN_PLACE_ENTRIES at trace time, so each switch and the
    teardown clear the caches: no call reuses a trace made under the other path.
    """

    def use(in_place: bool) -> None:
        monkeypatch.setattr(hermite, "HERMITE_IN_PLACE_ENTRIES", 0 if in_place else 2**62)
        jax.clear_caches()

    use(getattr(request, "param", True))
    yield use
    jax.clear_caches()
