"""EPS unit tests for Blackwell (SM100) and CUDA 13 compatibility.

Validates module imports, shared library linking, and basic API construction.
Functional kernel tests require multi-GPU distributed setup and are not
covered here.

Run: python3 -m pytest tests/test_eps.py -v
Or:  python3 tests/test_eps.py
"""

import subprocess
import sys

import pytest
import torch

# Skip all tests if no GPU
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda:0"
capability = torch.cuda.get_device_capability(0)
IS_HOPPER_OR_BLACKWELL = capability >= (9, 0)


# ── Module import tests ─────────────────────────────────────────────


class TestImports:
    """Verify all EPS modules load without missing .so errors."""

    def test_import_communication(self):
        from eps.communication import MscclppCommunicator, MscclppCommunicatorParams
        assert MscclppCommunicator is not None

    def test_import_executor(self):
        from eps import executor
        assert executor is not None

    @pytest.mark.skipif(not IS_HOPPER_OR_BLACKWELL, reason="SM >= 9.0 required")
    def test_import_aok_grouped_gemm(self):
        from eps.executor import AokGroupedGemm
        assert AokGroupedGemm is not None

    def test_import_scheduler(self):
        from eps import scheduler
        assert scheduler is not None

    def test_import_utils(self):
        from eps import utils
        assert utils is not None

    def test_import_fast_ep(self):
        from eps import fast_ep
        assert fast_ep is not None

    def test_import_fast_oep(self):
        from eps import fast_oep
        assert fast_oep is not None


# ── Static linking validation ────────────────────────────────────────


class TestStaticLinking:
    """Verify shared libs don't have unresolved dependencies.

    This catches the libexecutor.so missing issue that occurs when
    EPS_STATIC_INTERNAL_LIBS is not properly applied.
    """

    @staticmethod
    def _check_ldd(module_name, lib_name):
        """Run ldd on a .so and check for 'not found' entries."""
        import importlib
        mod = importlib.import_module(module_name)
        mod_dir = mod.__path__[0]
        lib_path = f"{mod_dir}/{lib_name}"
        result = subprocess.run(
            ["ldd", lib_path], capture_output=True, text=True
        )
        not_found = [
            line.strip()
            for line in result.stdout.splitlines()
            if "not found" in line
        ]
        assert not not_found, (
            f"{lib_path} has unresolved deps:\n" + "\n".join(not_found)
        )

    def test_executor_no_missing_deps(self):
        self._check_ldd("eps.executor", "libexecutor_.so")

    def test_communication_no_missing_deps(self):
        self._check_ldd("eps.communication", "libcommunication_.so")

    def test_scheduler_no_missing_deps(self):
        self._check_ldd("eps.scheduler", "libscheduler_.so")

    def test_utils_no_missing_deps(self):
        self._check_ldd("eps.utils", "libutils_.so")

    def test_fast_ep_no_missing_deps(self):
        self._check_ldd("eps.fast_ep", "libfast_ep.so")

    def test_fast_oep_no_missing_deps(self):
        self._check_ldd("eps.fast_oep", "libfast_oep.so")


# ── Communication tests ─────────────────────────────────────────────


class TestCommunication:
    """Test MscclppCommunicator basic operations (single-rank)."""

    def test_create_unique_id(self):
        from eps.communication import MscclppCommunicator
        uid = MscclppCommunicator.createUniqueId()
        assert uid is not None

    def test_constructor_single_rank(self):
        from eps.communication import (
            MscclppCommunicator,
            MscclppCommunicatorParams,
        )
        uid = MscclppCommunicator.createUniqueId()
        comm = MscclppCommunicator(
            uid,
            MscclppCommunicatorParams(
                rank=0, ep_world_size=1, num_ranks_per_node=1
            ),
        )
        assert comm is not None
        assert comm.data_ptr() != 0

    def test_data_ptr_castable(self):
        from eps.communication import (
            MscclppCommunicator,
            MscclppCommunicatorParams,
        )
        uid = MscclppCommunicator.createUniqueId()
        comm = MscclppCommunicator(
            uid,
            MscclppCommunicatorParams(
                rank=0, ep_world_size=1, num_ranks_per_node=1
            ),
        )
        ptr = int(comm)
        assert ptr == comm.data_ptr()
        assert ptr != 0

    def test_multiple_communicators(self):
        """Ensure we can create multiple independent communicators."""
        from eps.communication import (
            MscclppCommunicator,
            MscclppCommunicatorParams,
        )
        comms = []
        for _ in range(3):
            uid = MscclppCommunicator.createUniqueId()
            comm = MscclppCommunicator(
                uid,
                MscclppCommunicatorParams(
                    rank=0, ep_world_size=1, num_ranks_per_node=1
                ),
            )
            comms.append(comm)

        ptrs = [c.data_ptr() for c in comms]
        # All should be valid and distinct
        assert all(p != 0 for p in ptrs)
        assert len(set(ptrs)) == 3


# ── Executor constructor tests ───────────────────────────────────────


class TestExecutorConstructors:
    """Test executor object creation (doesn't run kernels)."""

    @pytest.mark.skipif(not IS_HOPPER_OR_BLACKWELL, reason="SM >= 9.0 required")
    def test_aok_grouped_gemm_constructor(self):
        from eps.executor import AokGroupedGemm

        gemm = AokGroupedGemm(num_local_experts=8)
        assert gemm is not None
        gemm.destroy()

    @pytest.mark.skipif(not IS_HOPPER_OR_BLACKWELL, reason="SM >= 9.0 required")
    def test_aok_grouped_gemm_workspace_size(self):
        from eps.executor import AokGroupedGemm

        gemm = AokGroupedGemm(num_local_experts=8)
        ws = gemm.get_workspace_size(cols=7168, max_num_tokens=256)
        assert ws >= 0
        gemm.destroy()

    @pytest.mark.skipif(not IS_HOPPER_OR_BLACKWELL, reason="SM >= 9.0 required")
    def test_aok_grouped_gemm_different_expert_counts(self):
        from eps.executor import AokGroupedGemm

        for n in [1, 4, 8, 16]:
            gemm = AokGroupedGemm(num_local_experts=n)
            assert gemm is not None
            gemm.destroy()


# ── Fast EP / Fast OEP constructor tests ─────────────────────────────


class TestFastEPConstructors:
    """Test fast_ep and fast_oep AllToAll construction."""

    def _make_comm(self):
        from eps.communication import (
            MscclppCommunicator,
            MscclppCommunicatorParams,
        )
        uid = MscclppCommunicator.createUniqueId()
        return MscclppCommunicator(
            uid,
            MscclppCommunicatorParams(
                rank=0, ep_world_size=1, num_ranks_per_node=1
            ),
        )

    def test_fast_ep_alltoall_constructor(self):
        from eps.fast_ep.all_to_all import AllToAll

        comm = self._make_comm()
        a2a = AllToAll(
            top_k=2,
            num_experts=8,
            hidden_size=7168,
            max_num_global_tokens=256,
            comm=comm,
        )
        assert a2a is not None
        a2a.destroy()

    def test_fast_oep_alltoall_constructor(self):
        from eps.fast_oep.all_to_all import AllToAll

        comm = self._make_comm()
        a2a = AllToAll(
            embed_dim=7168,
            size_per_rank=256,
            max_num_global_tokens=256,
            comm=comm,
        )
        assert a2a is not None
        a2a.destroy()


# ── CUDA error check ────────────────────────────────────────────────


class TestCudaErrorCheck:
    """Verify CUDA error checking utility works."""

    def test_sync_check_no_error(self):
        from eps.communication import sync_check_cuda_error_eps

        # Should not raise after clean operations
        torch.zeros(1, device=DEVICE)
        sync_check_cuda_error_eps()


# ── GPU capability reporting ─────────────────────────────────────────


class TestEnvironment:
    """Report GPU environment for diagnostics."""

    def test_gpu_info(self):
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"\nGPU: {name}, SM {cap[0]}.{cap[1]}")
        print(f"torch: {torch.__version__}, CUDA: {torch.version.cuda}")
        assert True


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
