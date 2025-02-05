#!/usr/bin/env python3

import numpy as np
import os
from derecho.cascade.external_client import ServiceClientAPI




if __name__ == "__main__":
    capi = ServiceClientAPI()
    key = "/print/1"
    value = ["hello", "world"]
    subgroup_type = "VolatileCascadeStoreWithStringKey"
    subgroup_index = 0
    shard_index = 0

    res = capi.put(key, bytes(value,'utf-8'),subgroup_type=subgroup_type,
                subgroup_index=subgroup_index,shard_index=shard_index, message_id=1)



