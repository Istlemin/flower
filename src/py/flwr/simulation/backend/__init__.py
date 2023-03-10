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
"""Backend implementation for simulations."""


from abc import ABCMeta, abstractmethod
from typing import Callable, Optional

from flwr.client.client import Client
from flwr.server.client_proxy import ClientProxy


class Backend(object, metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    def init(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    @abstractmethod
    def get_client_proxy(
        self,
        client_fn: Callable[[str], Client],
        cid: str,
        seed_fn: Optional[Callable[[int], None]] = None,
    ) -> ClientProxy:
        pass
