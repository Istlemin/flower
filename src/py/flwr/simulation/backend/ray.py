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

import ray
from flwr import common
from flwr.client import ClientLike
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.simulation.backend.determistic import MAX_SEED_SIZE, DeterministicClientProxy

ClientFn = Callable[[str], ClientLike]


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
        seed = random.randint(0, MAX_SEED_SIZE) if seed_fn is not None else None
        self.remote_client_proxy = _RayRemoteProxyClient.options(**resources).remote(
            client_fn,
            cid,
            seed_fn=seed_fn,
            seed=seed,
        )

    def _get_ray_future(
        self,
        ray_future: ray.ObjectRef,
        timeout: Optional[float] = None,
        type: Optional[Type] = None,
    ) -> Any:
        try:
            res = ray.get(ray_future, timeout=timeout)
            if type is not None:
                res = cast(type, res)
            return res
        except Exception as e:
            log(DEBUG, e)  # logging error to reproduce previous behaviour
            raise e

    def get_properties(
        self, ins: common.GetPropertiesIns, timeout: Optional[float]
    ) -> common.GetPropertiesRes:
        """Returns client's properties."""
        return self._get_ray_future(
            self.remote_client_proxy.get_properties.remote(ins, timeout),
            timeout=timeout,
            type=common.GetPropertiesRes,
        )

    def get_parameters(
        self, ins: common.GetParametersIns, timeout: Optional[float]
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        return self._get_ray_future(
            self.remote_client_proxy.get_parameters.remote(ins, timeout),
            timeout=timeout,
            type=common.GetParametersRes,
        )

    def fit(self, ins: common.FitIns, timeout: Optional[float]) -> common.FitRes:
        """Train model parameters on the locally held dataset."""
        return self._get_ray_future(
            self.remote_client_proxy.fit.remote(ins, timeout),
            timeout=timeout,
            type=common.FitRes,
        )

    def evaluate(
        self, ins: common.EvaluateIns, timeout: Optional[float]
    ) -> common.EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        return self._get_ray_future(
            self.remote_client_proxy.evaluate.remote(ins, timeout),
            timeout=timeout,
            type=common.EvaluateRes,
        )

    def reconnect(
        self, ins: common.ReconnectIns, timeout: Optional[float]
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return self._get_ray_future(
            self.remote_client_proxy.reconnect.remote(ins, timeout),
            timeout=timeout,
            type=common.DisconnectRes,
        )


@ray.remote
class _RayRemoteProxyClient(DeterministicClientProxy):
    ...
