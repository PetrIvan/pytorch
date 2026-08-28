# Owner(s): ["module: inductor"]
import types
from unittest import mock

import torch
import torch.fx
from torch._inductor.codegen import common
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    init_backend_registration,
    register_backend_for_device,
)
from torch._inductor.test_case import TestCase


class DeviceBackendInitTest(TestCase):
    """
    Tests the optional ``_inductor_backend_init`` hook: a no-arg callable on a
    privateuse1 device module (the one registered via
    ``torch._register_device_module``).  Inductor invokes it once per process
    at the first compile, before the decomposition table is built; the hook
    must register the device via ``register_backend_for_device`` itself.
    """

    device = "fakedevice"

    def setUp(self) -> None:
        super().setUp()
        self._orig_codegen = dict(common.device_codegens)
        self._orig_custom_passes = dict(common.custom_backend_passes)
        self._orig_custom_configs = dict(common.custom_backend_codegen_configs)
        init_backend_registration.cache_clear()

    def tearDown(self) -> None:
        # Drop whatever the hooks registered, remove the fake device module
        # from torch, and clear the cache so later tests start clean.
        for registry, orig in [
            (common.device_codegens, self._orig_codegen),
            (common.custom_backend_passes, self._orig_custom_passes),
            (common.custom_backend_codegen_configs, self._orig_custom_configs),
        ]:
            for device in [d for d in registry if d not in orig]:
                del registry[device]
        if hasattr(torch, self.device):
            delattr(torch, self.device)
        init_backend_registration.cache_clear()
        super().tearDown()

    def _install_device_module(self, **attrs) -> None:
        # rename_privateuse1_backend() is once-per-process, so mock the name
        # lookup instead of renaming for real.  Both references must be
        # patched: the probe in common.py and _get_custom_mod_func in
        # backend_registration, which imported its own copy.
        setattr(torch, self.device, types.SimpleNamespace(**attrs))
        for target in (
            "torch._C._get_privateuse1_backend_name",
            "torch.utils.backend_registration._get_privateuse1_backend_name",
        ):
            patcher = mock.patch(target, return_value=self.device)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _install_hook_registering_device(self) -> dict:
        calls = {"n": 0}

        def hook() -> None:
            calls["n"] += 1
            register_backend_for_device(self.device, object, object)

        self._install_device_module(_inductor_backend_init=hook)
        return calls

    # ------------------------------------------------------------------
    # Hook semantics
    # ------------------------------------------------------------------

    def test_backend_init_called_once(self) -> None:
        calls = self._install_hook_registering_device()
        init_backend_registration()
        self.assertIsNotNone(get_scheduling_for_device(self.device))
        # init_backend_registration is cached, so the hook is not re-fired.
        init_backend_registration()
        self.assertEqual(calls["n"], 1)

    def test_backend_init_takes_precedence_over_class_probe(self) -> None:
        # When both the hook and the legacy class attributes are present,
        # only the hook runs.
        calls = {"n": 0}

        class LegacyScheduling:
            pass

        class LegacyWrapper:
            pass

        class LegacyCppWrapper:
            pass

        class LegacyFxWrapper:
            pass

        def hook() -> None:
            calls["n"] += 1
            register_backend_for_device(self.device, object, object)

        self._install_device_module(
            _inductor_backend_init=hook,
            Scheduling=LegacyScheduling,
            PythonWrapperCodegen=LegacyWrapper,
            CppWrapperCodegen=LegacyCppWrapper,
            WrapperFxCodegen=LegacyFxWrapper,
        )
        init_backend_registration()
        self.assertEqual(calls["n"], 1)
        self.assertIs(get_scheduling_for_device(self.device), object)

    def test_class_probe_fallback_without_hook(self) -> None:
        # A device module without the hook still goes through the legacy
        # four-class probe.  All four attributes are required: a missing one
        # makes _get_custom_mod_func raise inside the probe.
        class Scheduling:
            pass

        class Wrapper:
            pass

        class CppWrapper:
            pass

        class FxWrapper:
            pass

        self._install_device_module(
            Scheduling=Scheduling,
            PythonWrapperCodegen=Wrapper,
            CppWrapperCodegen=CppWrapper,
            WrapperFxCodegen=FxWrapper,
        )
        init_backend_registration()
        self.assertIs(get_scheduling_for_device(self.device), Scheduling)

    def test_backend_init_failure_propagates_and_retries(self) -> None:
        calls = {"n": 0}

        def hook() -> None:
            calls["n"] += 1
            raise RuntimeError("boom")

        self._install_device_module(_inductor_backend_init=hook)
        # The vendor's error propagates instead of being swallowed by the
        # probe, and functools.cache does not cache failures, so the next
        # compile retries the hook.
        with self.assertRaisesRegex(RuntimeError, "boom"):
            init_backend_registration()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            init_backend_registration()
        self.assertEqual(calls["n"], 2)

    def test_backend_init_noop_not_retried(self) -> None:
        # A hook that returns without registering is a deterministic vendor
        # bug; it must not be re-fired on every subsequent compile.
        calls = {"n": 0}

        def hook() -> None:
            calls["n"] += 1

        self._install_device_module(_inductor_backend_init=hook)
        init_backend_registration()
        self.assertIsNone(get_scheduling_for_device(self.device))
        init_backend_registration()
        self.assertEqual(calls["n"], 1)

    # ------------------------------------------------------------------
    # Placement on the compile path
    # ------------------------------------------------------------------

    def test_init_fires_before_decomp_table(self) -> None:
        # init_backend_registration must run at the top of _compile_fx_main,
        # before the decomposition table is snapshotted: a vendor hook may
        # register decompositions, and the first compile must see them.
        order = []
        compile_fx_mod = torch._inductor.compile_fx
        real_init = compile_fx_mod.init_backend_registration
        real_select_decomp = compile_fx_mod.select_decomp_table

        def record_init() -> None:
            order.append("init")
            real_init()

        def record_decomp():
            order.append("decomp")
            return real_select_decomp()

        gm = torch.fx.symbolic_trace(lambda x: (x + 1,))
        x = torch.randn(8)
        with (
            mock.patch.object(compile_fx_mod, "init_backend_registration", record_init),
            mock.patch.object(compile_fx_mod, "select_decomp_table", record_decomp),
        ):
            compile_fx_mod.compile_fx(gm, [x])

        self.assertEqual(order[0], "init")
        self.assertLess(order.index("init"), order.index("decomp"))

    def test_backend_init_fires_on_torch_compile(self) -> None:
        calls = self._install_hook_registering_device()

        def fn(x):
            return x + 1

        x = torch.randn(8)
        self.assertEqual(torch.compile(fn)(x), fn(x))
        self.assertEqual(calls["n"], 1)
        self.assertIsNotNone(get_scheduling_for_device(self.device))

    def test_backend_init_fires_on_compile_fx(self) -> None:
        calls = self._install_hook_registering_device()
        gm = torch.fx.symbolic_trace(lambda x: (x + 1,))
        torch._inductor.compile_fx.compile_fx(gm, [torch.randn(8)])
        self.assertEqual(calls["n"], 1)
        self.assertIsNotNone(get_scheduling_for_device(self.device))

    def test_backend_init_fires_on_aot_compile(self) -> None:
        calls = self._install_hook_registering_device()

        class Model(torch.nn.Module):
            def forward(self, x):
                return (x + 1,)

        x = torch.randn(8)
        exported = torch.export.export(Model(), (x,))
        torch._inductor.aot_compile(exported.module(), (x,))
        self.assertEqual(calls["n"], 1)
        self.assertIsNotNone(get_scheduling_for_device(self.device))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
