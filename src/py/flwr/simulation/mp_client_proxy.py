# Copyright 2020 Adap GmbH. All Rights Reserved.
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
# ==============================================================================
"""Ray-based Flower ClientProxy implementation."""


import random
from logging import DEBUG
from typing import Any, Callable, Dict, Optional, Type, cast

import pathos.multiprocessing as mp

from flwr import common
from flwr.client import ClientLike, to_client
from flwr.client.client import (
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

ClientFn = Callable[[str], ClientLike]

MAX_SEED_SIZE = 10000000

def execute_with_timeout(fn,ins,timeout):
    with mp.Pool(processes=1) as p:
        res = p.apply_async(fn, args=(ins,))
        return res.get(timeout)

class MPClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(
        self,
        client_fn: ClientFn,
        seed_fn: Optional[Callable[[int], None]],
        cid: str,
        resources: Dict[str, float],
    ):
        super().__init__(cid)
        seed = random.randint(0, MAX_SEED_SIZE) if seed_fn is not None else None
        self.remote_client_proxy = _RemoteMPClientProxy(
            client_fn,
            cid,
            seed_fn=seed_fn,
            seed=seed,
        )

    def get_properties(self, ins: common.GetPropertiesIns, timeout: Optional[float]) -> common.GetPropertiesRes:
        """Returns client's properties."""
        return execute_with_timeout(self.remote_client_proxy.get_properties,ins, timeout)

    def get_parameters(self, ins: common.GetParametersIns, timeout: Optional[float]) -> common.GetParametersRes:
        """Return the current local model parameters."""
        return execute_with_timeout(self.remote_client_proxy.get_parameters,ins, timeout)

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        return execute_with_timeout(self.remote_client_proxy.fit,ins, timeout)

    def evaluate(self, ins: common.EvaluateIns, timeout: Optional[float]) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        return execute_with_timeout(self.remote_client_proxy.evaluate,ins, timeout)

    def reconnect(self, ins: common.ReconnectIns, timeout: Optional[float]) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return execute_with_timeout(self.remote_client_proxy.reconnect,ins, timeout)


class _RemoteMPClientProxy(ClientProxy):
    """The remote part of the RayClientProxy.

    Keeps track of it's rng for reproducibility.
    """

    def __init__(
        self,
        client_fn: ClientFn,
        cid: str,
        seed_fn: Optional[Callable[[int], None]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(cid)
        self.client_fn = client_fn
        self.seed_fn = seed_fn
        if self.seed_fn is not None:
            # Create a RNG for this client, seeded using the global RNG
            self.rng = random.Random()
            self.rng.seed(seed)
        else:
            self.rng = None

    def _get_client(self):
        return to_client(client_like=self.client_fn(self.cid))

    def _set_seeds(self):
        if self.seed_fn is not None:
            seed = self.rng.randint(0, MAX_SEED_SIZE)
            self.seed_fn(seed)

    def get_properties(self, ins: common.GetPropertiesIns) -> common.GetPropertiesRes:
        """Returns client's properties."""
        self._set_seeds()
        return maybe_call_get_properties(self._get_client(), ins)

    def get_parameters(self, ins: common.GetParametersIns) -> common.GetParametersRes:
        """Return the current local model parameters."""
        self._set_seeds()
        return maybe_call_get_parameters(self._get_client(), ins)

    def fit(self, ins: common.FitIns) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        self._set_seeds()
        return maybe_call_fit(self._get_client(), ins)

    def evaluate(self, ins: common.EvaluateIns) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        self._set_seeds()
        return maybe_call_evaluate(self._get_client(), ins)

    def reconnect(self, ins: common.ReconnectIns) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return common.DisconnectRes(reason="")  # Nothing to do here (yet)
