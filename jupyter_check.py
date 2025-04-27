import sys
import json
import os
from jupyter_client.kernelspec import find_kernel_specs, get_kernel_spec

print("🔍 Python:", sys.executable)
print("🔍 Python Version:", sys.version)
print("\n🧠 Jupyter Kernels:")
for name, path in find_kernel_specs().items():
    spec = get_kernel_spec(name)
    print(f"  - {name}: {spec.argv[0]} ({spec.resource_dir})")

try:
    import websocket
    print("\n✅ WebSocket available in Python.")
except ImportError:
    print("\n❌ websocket-client not found.")
