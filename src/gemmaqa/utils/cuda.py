"""
CUDA availability check utility.
"""

import torch


def check_cuda() -> dict:
    """
    Check CUDA availability and return device info.
    
    Returns:
        Dict with CUDA availability info.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count()
        info["devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_gb": torch.cuda.get_device_properties(i).total_memory / 1e9,
            }
            for i in range(torch.cuda.device_count())
        ]
    
    return info


def main():
    """CLI entry point for CUDA check."""
    info = check_cuda()
    
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info["cuda_available"]:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"Device Count: {info['device_count']}")
        for device in info["devices"]:
            print(f"  [{device['index']}] {device['name']} ({device['memory_gb']:.1f} GB)")
    else:
        print("No CUDA devices found. Using CPU.")


if __name__ == "__main__":
    main()
