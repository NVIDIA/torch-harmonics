#!/usr/bin/env python3
"""
Build wheels following PyTorch ecosystem naming convention.
This script creates wheels with proper +cuXXX or +cpu suffixes.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# PyTorch/CUDA combinations following PyTorch ecosystem conventions
BUILD_MATRIX = [
    # PyTorch 2.4.x with multiple CUDA versions
    {"torch": "2.4.0", "cuda": "cu118", "cuda_version": "11.8"},
    {"torch": "2.4.0", "cuda": "cu121", "cuda_version": "12.1"},
    {"torch": "2.4.0", "cuda": "cu124", "cuda_version": "12.4"},
    {"torch": "2.4.0", "cuda": "cpu", "cuda_version": "none"},

    # PyTorch 2.5.x with multiple CUDA versions
    {"torch": "2.5.0", "cuda": "cu118", "cuda_version": "11.8"},
    {"torch": "2.5.0", "cuda": "cu121", "cuda_version": "12.1"},
    {"torch": "2.5.0", "cuda": "cu124", "cuda_version": "12.4"},
    {"torch": "2.5.0", "cuda": "cpu", "cuda_version": "none"},

    # PyTorch 2.6.x with multiple CUDA versions
    {"torch": "2.6.0", "cuda": "cu118", "cuda_version": "11.8"},
    {"torch": "2.6.0", "cuda": "cu121", "cuda_version": "12.1"},
    {"torch": "2.6.0", "cuda": "cu124", "cuda_version": "12.4"},
    {"torch": "2.6.0", "cuda": "cpu", "cuda_version": "none"},

]

def get_version():
    """Get the current version from pyproject.toml or setup.py."""
    try:
        # Try to get version from git tag
        result = subprocess.run(['git', 'describe', '--tags', '--dirty'],
                              capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        # Remove 'v' prefix if present
        if version.startswith('v'):
            version = version[1:]
        return version
    except:
        return "0.8.1"  # Fallback version

def build_wheel_for_config(config):
    """Build wheel for a specific PyTorch/CUDA configuration."""
    torch_version = config["torch"]
    cuda_suffix = config["cuda"]
    cuda_version = config["cuda_version"]

    print(f"\n{'='*80}")
    print(f"Building wheel for PyTorch {torch_version} + {cuda_suffix}")
    print(f"{'='*80}")

    # Create isolated environment
    env_name = f"build_env_{torch_version.replace('.', '_')}_{cuda_suffix}"

    try:
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", env_name], check=True)

        # Determine pip path
        if os.name == 'nt':  # Windows
            pip_cmd = f"{env_name}\\Scripts\\pip"
            python_cmd = f"{env_name}\\Scripts\\python"
        else:  # Unix/Linux/macOS
            pip_cmd = f"{env_name}/bin/pip"
            python_cmd = f"{env_name}/bin/python"

        # Install build dependencies
        subprocess.run([
            pip_cmd, "install", "--upgrade",
            "pip", "setuptools", "wheel", "cibuildwheel", "setuptools-scm"
        ], check=True)

        # Install PyTorch
        if cuda_suffix == "cpu":
            subprocess.run([
                pip_cmd, "install",
                f"torch=={torch_version}+cpu",
                "--index-url", "https://download.pytorch.org/whl/cpu",
                "numpy"
            ], check=True)
        else:
            subprocess.run([
                pip_cmd, "install",
                f"torch=={torch_version}+{cuda_suffix}",
                "--index-url", f"https://download.pytorch.org/whl/{cuda_suffix}",
                "numpy"
            ], check=True)

        # Build manylinux wheels using cibuildwheel
        env = os.environ.copy()
        env.update({
            "CIBW_BUILD": "cp39-* cp310-* cp311-* cp312-*",
            "CIBW_PLATFORM": "linux",
            "CIBW_BEFORE_BUILD": f"pip install torch=={torch_version}+{cuda_suffix} --index-url https://download.pytorch.org/whl/{cuda_suffix} numpy",
            "CIBW_ENVIRONMENT": f"TORCH_CUDA_VERSION={cuda_version}",
            "CIBW_REPAIR_WHEEL_COMMAND": "auditwheel repair -w {dest_dir} {wheel}",
            "CIBW_TEST_COMMAND": "python -c 'import torch_harmonics; print(\"Import successful\")'"
        })

        subprocess.run([
            python_cmd, "-m", "cibuildwheel", "--platform", "linux"
        ], check=True, env=env)

        # Rename wheels to follow PyTorch ecosystem convention
        version = get_version()
        wheel_pattern = f"torch_harmonics-{version}-*.whl"
        wheel_files = list(Path("wheelhouse").glob(wheel_pattern))

        # Rename all wheels in wheelhouse
        for wheel_file in wheel_files:
            # Extract the wheel filename parts
            parts = wheel_file.stem.split('-')
            if len(parts) >= 4:
                # Format: torch_harmonics-0.8.1-cp310-cp310-linux_x86_64.whl
                # Convert to: torch_harmonics-0.8.1+cu121-cp310-cp310-linux_x86_64.whl
                new_parts = [
                    parts[0],  # torch_harmonics
                    f"{parts[1]}+{cuda_suffix}",  # 0.8.1+cu121
                ] + parts[2:]  # cp310-cp310-linux_x86_64

                new_wheel_name = '-'.join(new_parts) + '.whl'
                new_wheel_path = Path("wheelhouse") / new_wheel_name

                # Rename the wheel
                shutil.move(str(wheel_file), str(new_wheel_path))
                print(f"✅ Renamed wheel to: {new_wheel_name}")

        print(f"✅ Successfully built wheel for PyTorch {torch_version} + {cuda_suffix}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build wheel for PyTorch {torch_version} + {cuda_suffix}: {e}")
        return False
    finally:
        # Clean up environment
        if Path(env_name).exists():
            shutil.rmtree(env_name)

def main():
    """Build wheels for all configurations."""
    print("Building torch-harmonics wheels following PyTorch ecosystem naming convention...")

    successful_builds = []
    failed_builds = []

    for config in BUILD_MATRIX:
        if build_wheel_for_config(config):
            successful_builds.append(f"{config['torch']}+{config['cuda']}")
        else:
            failed_builds.append(f"{config['torch']}+{config['cuda']}")

    # Summary
    print(f"\n{'='*80}")
    print("BUILD SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful builds: {len(successful_builds)}")
    for build in successful_builds:
        print(f"   - PyTorch {build}")

    if failed_builds:
        print(f"❌ Failed builds: {len(failed_builds)}")
        for build in failed_builds:
            print(f"   - PyTorch {build}")

    print(f"\nWheels are available in the 'wheelhouse/' directory")
    print("Example wheel names:")
    print("  - torch_harmonics-0.8.1+cu121-cp310-cp310-linux_x86_64.whl")
    print("  - torch_harmonics-0.8.1+cpu-cp310-cp310-linux_x86_64.whl")
    print("\nUpload to PyPI with: python -m twine upload wheelhouse/*")

if __name__ == "__main__":
    main()
