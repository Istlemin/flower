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
"""Flower simulation app."""


import random
from logging import ERROR, INFO
from typing import Any, Callable, Dict, List, Optional

import ray
from flwr.client.client import Client
from flwr.common import EventType, event
from flwr.common.logger import log
from flwr.server import Server
from flwr.server.app import ServerConfig, _fl, _init_defaults
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.strategy import Strategy
from flwr.simulation.backend import Backend
from flwr.simulation.backend.ray_backend import RayBackend

INVALID_ARGUMENTS_START_SIMULATION_CLIENTS = """
INVALID ARGUMENTS ERROR

Invalid Arguments in method:

`start_simulation(
    *,
    client_fn: Callable[[str], Client],
    num_clients: Optional[int] = None,
    clients_ids: Optional[List[str]] = None,
    client_resources: Optional[Dict[str, float]] = None,
    server: Optional[Server] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,
    keep_initialised: Optional[bool] = False,
    seed_fn: Optional[Callable[[int], None]] = None,
    seed: Optional[int] = None,
) -> None:`

REASON:
    Method requires:
        - Either `num_clients`[int] or `clients_ids`[List[str]]
        to be set exclusively.
        OR
        - `len(clients_ids)` == `num_clients`

"""
INVALID_ARGUMENTS_START_SIMULATION_SEED = """
INVALID ARGUMENTS ERROR

Invalid Arguments in method:

`start_simulation(
    *,
    client_fn: Callable[[str], Client],
    num_clients: Optional[int] = None,
    clients_ids: Optional[List[str]] = None,
    client_resources: Optional[Dict[str, float]] = None,
    server: Optional[Server] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,
    keep_initialised: Optional[bool] = False,
    seed_fn: Optional[Callable[[int], None]] = None,
    seed: Optional[int] = None,
) -> None:`

REASON:
    Method requires:
        - `seed`[int] and `seed_fn`[Callable[[int], None]]
        to be either both set or both None.
"""


def start_simulation(  # pylint: disable=too-many-arguments
    *,
    client_fn: Callable[[str], Client],
    num_clients: Optional[int] = None,
    clients_ids: Optional[List[str]] = None,
    client_resources: Optional[Dict[str, float]] = None,
    server: Optional[Server] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,
    keep_initialised: Optional[bool] = False,
    seed_fn: Optional[Callable[[int], None]] = None,
    seed: Optional[int] = None,
) -> History:
    """Start a Ray-based Flower simulation server.

    Parameters
    ----------
    client_fn : Callable[[str], Client]
        A function creating client instances. The function must take a single
        str argument called `cid`. It should return a single client instance.
        Note that the created client instances are ephemeral and will often be
        destroyed after a single method invocation. Since client instances are
        not long-lived, they should not attempt to carry state over method
        invocations. Any state required by the instance (model, dataset,
        hyperparameters, ...) should be (re-)created in either the call to
        `client_fn` or the call to any of the client methods (e.g., load
        evaluation data in the `evaluate` method itself).
    num_clients : Optional[int]
        The total number of clients in this simulation. This must be set if
        `clients_ids` is not set and vice-versa.
    clients_ids : Optional[List[str]]
        List `client_id`s for each client. This is only required if
        `num_clients` is not set. Setting both `num_clients` and `clients_ids`
        with `len(clients_ids)` not equal to `num_clients` generates an error.
    client_resources : Optional[Dict[str, float]] (default: None)
        CPU and GPU resources for a single client. Supported keys are
        `num_cpus` and `num_gpus`. Example: `{"num_cpus": 4, "num_gpus": 1}`.
        To understand the GPU utilization caused by `num_gpus`, consult the Ray
        documentation on GPU support.
    server : Optional[flwr.server.Server] (default: None).
        An implementation of the abstract base class `flwr.server.Server`. If no
        instance is provided, then `start_server` will create one.
    config: ServerConfig (default: None).
        Currently supported values are `num_rounds` (int, default: 1) and
        `round_timeout` in seconds (float, default: None).
    strategy : Optional[flwr.server.Strategy] (default: None)
        An implementation of the abstract base class `flwr.server.Strategy`. If
        no strategy is provided, then `start_server` will use
        `flwr.server.strategy.FedAvg`.
    client_manager : Optional[flwr.server.ClientManager] (default: None)
        An implementation of the abstract base class `flwr.server.ClientManager`.
        If no implementation is provided, then `start_simulation` will use
        `flwr.server.client_manager.SimpleClientManager`.
    ray_init_args : Optional[Dict[str, Any]] (default: None)
        Optional dictionary containing arguments for the call to `ray.init`.
        If ray_init_args is None (the default), Ray will be initialized with
        the following default args:

        { "ignore_reinit_error": True, "include_dashboard": False }

        An empty dictionary can be used (ray_init_args={}) to prevent any
        arguments from being passed to ray.init.
    keep_initialised: Optional[bool] (default: False)
        Set to True to prevent `ray.shutdown()` in case `ray.is_initialized()=True`.

    Returns
    -------
        hist : flwr.server.history.History.
            Object containing metrics from training.
    """
    # pylint: disable-msg=too-many-locals
    event(
        EventType.START_SIMULATION_ENTER,
        {"num_clients": len(clients_ids) if clients_ids is not None else num_clients},
    )

    # Initialize server and server config
    initialized_server, initialized_config = _init_defaults(
        server=server,
        config=config,
        strategy=strategy,
        client_manager=client_manager,
    )
    log(
        INFO,
        "Starting Flower simulation, config: %s",
        initialized_config,
    )

    # clients_ids takes precedence
    cids: List[str]
    if clients_ids is not None:
        if (num_clients is not None) and (len(clients_ids) != num_clients):
            log(ERROR, INVALID_ARGUMENTS_START_SIMULATION_CLIENTS)
            raise ValueError(INVALID_ARGUMENTS_START_SIMULATION_CLIENTS)
        else:
            cids = clients_ids
    else:
        if num_clients is None:
            log(ERROR, INVALID_ARGUMENTS_START_SIMULATION_CLIENTS)
            raise ValueError(INVALID_ARGUMENTS_START_SIMULATION_CLIENTS)
        else:
            cids = [str(x) for x in range(num_clients)]

    if seed is not None:
        if seed_fn is None:
            log(ERROR, INVALID_ARGUMENTS_START_SIMULATION_SEED)
            raise ValueError(INVALID_ARGUMENTS_START_SIMULATION_SEED)

    elif seed_fn is not None:
        log(ERROR, INVALID_ARGUMENTS_START_SIMULATION_SEED)
        raise ValueError(INVALID_ARGUMENTS_START_SIMULATION_SEED)

    # Default arguments for Ray initialization
    if not ray_init_args:
        ray_init_args = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }

    # Register one RayClientProxy object for each client with the ClientManager
    client_resources = client_resources if client_resources is not None else {}
    backend: Backend = RayBackend(
        client_resources=client_resources,
        ray_init_args=ray_init_args,
        keep_initialised=keep_initialised,
    )
    backend.init()

    if seed is not None:
        # Set seed for everything running in main thread
        seed_fn(seed)

        # Set seed for random sampling
        random.seed(seed)

    log(
        INFO,
        "Flower VCE: Ray initialized with resources: %s",
        ray.cluster_resources(),
    )

    for cid in cids:
        client_proxy = backend.get_client_proxy(client_fn, cid, seed_fn)
        initialized_server.client_manager().register(client=client_proxy)

    # Start training
    hist = _fl(
        server=initialized_server,
        config=initialized_config,
    )

    event(EventType.START_SIMULATION_LEAVE)
    backend.shutdown()
    return hist
