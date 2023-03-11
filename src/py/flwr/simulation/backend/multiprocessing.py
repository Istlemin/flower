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
from typing import Callable, Optional

import dill
import torch.multiprocessing as mp

from flwr import common
from flwr.client import ClientLike
from flwr.server.client_proxy import ClientProxy
from flwr.simulation.backend import Backend
from flwr.simulation.backend.deterministic import DeterministicClientProxy

ClientFn = Callable[[str], ClientLike]

MAX_SEED_SIZE = 10000000


def run_dilled_fn(fn, args):
    fn = dill.loads(fn)
    args = dill.loads(args)
    return dill.dumps(fn(*args))


class MultiProcessingBackend(Backend):
    def __init__(self) -> None:
        self.processing_pool = None

    def init(self) -> None:
        self.processing_pool = mp.Pool()

    def shutdown(self) -> None:
        self.processing_pool.close()
        self.processing_pool.join()

    def get_client_proxy(
        self,
        client_fn: Callable[[str], ClientLike],
        cid: str,
        seed_fn: Optional[Callable[[int], None]] = None,
    ) -> ClientProxy:
        return MultiProcessingClientProxy(client_fn, seed_fn, cid, self.processing_pool)


class MultiProcessingClientProxy(ClientProxy):
    """Flower client proxy which delegates work using torch.multiprocessing."""

    def __init__(
        self,
        client_fn: ClientFn,
        seed_fn: Optional[Callable[[int], None]],
        cid: str,
        processing_pool: mp.Pool,
    ):
        super().__init__(cid)
        seed = random.randint(0, MAX_SEED_SIZE) if seed_fn is not None else None
        self.deterministic_client_proxy = DeterministicClientProxy(
            client_fn,
            cid,
            seed_fn=seed_fn,
            seed=seed,
        )
        self.processing_pool = processing_pool

    def _run_dilled_fn(self, fn, *args, timeout=None):
        fn = dill.dumps(fn)
        args = dill.dumps(args)
        return dill.loads(
            self.processing_pool.apply_async(run_dilled_fn, args=(fn, args)).get(
                timeout
            )
        )

    def get_properties(
        self, ins: common.GetPropertiesIns, timeout: Optional[float]
    ) -> common.GetPropertiesRes:
        """Returns client's properties."""
        return self._run_dilled_fn(
            self.deterministic_client_proxy.get_properties,
            ins,
            timeout,
            timeout=timeout,
        )

    def get_parameters(
        self, ins: common.GetParametersIns, timeout: Optional[float]
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        return self._run_dilled_fn(
            self.deterministic_client_proxy.get_parameters,
            ins,
            timeout,
            timeout=timeout,
        )

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        return self._run_dilled_fn(
            self.deterministic_client_proxy.fit,
            ins,
            timeout,
            timeout=timeout,
        )

    def evaluate(
        self, ins: common.EvaluateIns, timeout: Optional[float]
    ) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        return self._run_dilled_fn(
            self.deterministic_client_proxy.evaluate,
            ins,
            timeout,
            timeout=timeout,
        )

    def reconnect(
        self, ins: common.ReconnectIns, timeout: Optional[float]
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return self._run_dilled_fn(
            self.deterministic_client_proxy.reconnect,
            ins,
            timeout,
            timeout=timeout,
        )
