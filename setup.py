from __future__ import annotations

import os
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def run(self) -> None:
        if os.environ.get("EPS_SKIP_CMAKE") != "1":
            self._run_cmake_build()
        super().run()

    def _run_cmake_build(self) -> None:
        root = Path(__file__).resolve().parent
        build_dir = Path(os.environ.get("EPS_BUILD_DIR", root / "build" / "pip"))
        build_dir.mkdir(parents=True, exist_ok=True)

        build_type = os.environ.get("EPS_BUILD_TYPE", "Release")
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={build_type}",
            "-DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF",
            "-DMSCCLPP_BYPASS_GPU_CHECK=ON",
            "-DMSCCLPP_USE_CUDA=ON",
            "-DWITH_NVSHMEM=OFF",
        ]

        extra_args = os.environ.get("EPS_CMAKE_ARGS")
        if extra_args:
            cmake_args.extend(extra_args.split())

        env = os.environ.copy()
        arch_list = env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a;10.0a")
        if "10.0a" in arch_list or "10.0" in arch_list:
            env.setdefault("EPS_SM_NUM", "100a")
        elif "9.0a" in arch_list or "9.0" in arch_list:
            env.setdefault("EPS_SM_NUM", "90a")

        subprocess.check_call(
            ["cmake", "-S", str(root), "-B", str(build_dir)] + cmake_args,
            env=env,
        )
        build_cmd = ["cmake", "--build", str(build_dir), "--parallel"]
        jobs = os.environ.get("EPS_BUILD_JOBS")
        if jobs:
            build_cmd.append(jobs)
        subprocess.check_call(build_cmd, env=env)


packages = find_packages(where="python")

setup(
    name="eps",
    version="0.0.0",
    description="EPS CUDA extensions and Python bindings",
    package_dir={"": "python"},
    packages=packages,
    package_data={
        "eps": ["py.typed"],
        "eps.fast_ep": ["*.so", "py.typed"],
        "eps.fast_oep": ["*.so", "py.typed"],
        "eps.executor": ["*.so", "py.typed"],
        "eps.scheduler": ["*.so", "py.typed"],
        "eps.communication": ["*.so", "py.typed"],
        "eps.utils": ["*.so", "py.typed"],
    },
    cmdclass={"build_py": build_py},
    zip_safe=False,
)
