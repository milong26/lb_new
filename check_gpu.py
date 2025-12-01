import sys
from pynvml import *

try:
    nvmlInit()
    count = nvmlDeviceGetCount()
    print(f"发现 GPU 数量: {count}")

    for i in range(count):
        handle = nvmlDeviceGetHandleByIndex(i)
        name = nvmlDeviceGetName(handle)
        pci = nvmlDeviceGetPciInfo(handle)
        print(f"\nGPU {i}:")
        print(f"  名称: {name}")
        print(f"  PCI Bus ID: {pci.busId}")
    
    nvmlShutdown()

except NVMLError as err:
    print(f"NVML 错误: {err}")
    sys.exit(1)
