import os
from typing import Callable, Optional

import torch.multiprocessing as mp

from flwr.client import ClientLike
from flwr.server.client_proxy import ClientProxy
from flwr.simulation.backend import Backend

from flwr.simulation.backend.multiprocessing import MultiProcessingClientProxy

class TorchMultiProcessingBackend(Backend):
    def __init__(self, num_processes=None) -> None:
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        self.processing_pool = None

    def init(self) -> None:
        self.processing_pool = mp.Pool(self.num_processes)

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