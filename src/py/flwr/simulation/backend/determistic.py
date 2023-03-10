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
"""DeterministicClientProxy implementation."""


import random
from typing import Callable, Optional

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


class DeterministicClientProxy(ClientProxy):
    """This client can be run in any thread in a deterministic way, given it's
    seed.

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

    def get_properties(
        self, ins: common.GetPropertiesIns, timeout: Optional[float]
    ) -> common.GetPropertiesRes:
        """Returns client's properties."""
        self._set_seeds()
        return maybe_call_get_properties(self._get_client(), ins)

    def get_parameters(
        self, ins: common.GetParametersIns, timeout: Optional[float]
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        self._set_seeds()
        return maybe_call_get_parameters(self._get_client(), ins)

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        self._set_seeds()
        return maybe_call_fit(self._get_client(), ins)

    def evaluate(
        self, ins: common.EvaluateIns, timeout: Optional[float]
    ) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        self._set_seeds()
        return maybe_call_evaluate(self._get_client(), ins)

    def reconnect(
        self, ins: common.ReconnectIns, timeout: Optional[float]
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return common.DisconnectRes(reason="")  # Nothing to do here (yet)
