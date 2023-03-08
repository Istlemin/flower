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
from typing import Callable, Dict, Optional, cast

import ray
from flwr import common
from flwr.client import Client, ClientLike, to_client
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


class RayClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(
        self,
        client_fn: ClientFn,
        seed_fn: Optional[Callable[[int], None]],
        cid: str,
        resources: Dict[str, float],
    ):
        super().__init__(cid)
        self.client_fn = client_fn
        self.seed_fn = seed_fn
        self.resources = resources

        if self.seed_fn is not None:
            # Create a RNG for this client, seeded using the global RNG
            seed = random.randint(0, MAX_SEED_SIZE)
            self.rng = random.Random()
            self.rng.seed(seed)
        else:
            self.rng = None

    def _get_current_seed_fn(self):
        if self.seed_fn is None:
            return None
        else:
            seed = self.rng.randint(0, MAX_SEED_SIZE)
            return lambda: self.seed_fn(seed)

    def _get_client_fn(self):
        return lambda: self.client_fn(self.cid)

    def get_properties(
        self, ins: common.GetPropertiesIns, timeout: Optional[float]
    ) -> common.GetPropertiesRes:
        """Returns client's properties."""
        res = self._execute_deterministically_using_ray(
            maybe_call_get_properties, timeout, get_properties_ins=ins
        )
        return cast(
            common.GetPropertiesRes,
            res,
        )

    def get_parameters(
        self, ins: common.GetParametersIns, timeout: Optional[float]
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        res = self._execute_deterministically_using_ray(
            maybe_call_get_parameters, timeout, get_parameters_ins=ins
        )
        return cast(
            common.GetParametersRes,
            res,
        )

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        res = self._execute_deterministically_using_ray(
            maybe_call_fit, timeout, fit_ins=ins
        )
        return cast(
            common.FitRes,
            res,
        )

    def evaluate(
        self, ins: common.EvaluateIns, timeout: Optional[float]
    ) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        res = self._execute_deterministically_using_ray(
            maybe_call_evaluate, timeout, evaluate_ins=ins
        )
        return cast(
            common.EvaluateRes,
            res,
        )

    def reconnect(
        self, ins: common.ReconnectIns, timeout: Optional[float]
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return common.DisconnectRes(reason="")  # Nothing to do here (yet)

    def _execute_deterministically_using_ray(
        self,
        fn: Callable[[], None],
        timeout: Optional[float],
        *args,
        **kwargs,
    ):
        """Execute the given function deterministically using Ray."""
        future_fn_res = call_ray_with_seeded_client.options(  # type: ignore
            **self.resources,
        ).remote(
            self._get_client_fn(), self._get_current_seed_fn(), fn, *args, **kwargs
        )
        try:
            return ray.get(future_fn_res, timeout=timeout)
        except Exception as ex:
            log(DEBUG, ex)
            raise ex


@ray.remote
def call_ray_with_seeded_client(
    client_fn: ClientFn,
    seed_fn: Optional[Callable[[], None]],
    fn: Callable[[Client], common.FitRes],
    *args,
    **kwargs,
) -> common.FitRes:
    """Execute the given function remotely."""
    if seed_fn is not None:
        seed_fn()
    client: Client = to_client(client_like=client_fn())
    return fn(client, *args, **kwargs)
