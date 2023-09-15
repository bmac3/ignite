from dataclasses import dataclass, fields
from typing import Optional

import equinox as eqx
import jax
import optax


def unpack(dc) -> dict:
    return {field.name: getattr(dc, field.name) for field in fields(dc)}


@dataclass
class EngineState:
    model: eqx.Module
    opt_state: optax.OptState
    rng: jax.random.PRNGKey
    step: int = 0
    epoch: int = 0
    iteration: int = 0
    loss: Optional[float] = None
    best_val_metric: Optional[float] = None
    
    def save_to_disk(self, path):
        eqx.tree_serialise_leaves(path, EqxEngineState.from_engine_state(self))

    @staticmethod
    def load_from_disk(path, like):
        like = EqxEngineState.from_engine_state(like)
        eqx_state = eqx.tree_deserialise_leaves(path, like)
        return eqx_state.to_engine_state()


class EqxEngineState(eqx.Module, EngineState):
    
    @classmethod
    def from_engine_state(cls, state):
        return cls(**unpack(state))

    def to_engine_state(self):
        return EngineState(**unpack(self))
