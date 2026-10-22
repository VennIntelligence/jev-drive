"""Run AutoVLA's nuPlan preprocessing (tools/preprocessing/nocot_sample_generation.py, i.e. scripts/run_nuplan_preprocessing.sh
with INCLUDE_COT=false) unchanged, with the Waymo-only imports stubbed.

nocot_sample_generation.py imports the Waymo E2E dataset module at top level, which imports tensorflow and
waymo_open_dataset (TF 2.12 wheels, python <= 3.10; envs/autovla is python 3.12 for Blackwell). The nuPlan path never touches
them, so they are registered as empty stub modules here (the same move as experiments/baselines_latency/archive/autovla.py
stubbing models.utils.score); their code runs as shipped via runpy. A stub attribute is another empty stub module, so a Waymo code path
that ran by mistake would fail loudly (calling a module raises TypeError).

    cd <autovla repo> && python autovla_prep.py --config <rel cfg> --output_dir DIR --num_workers N
"""
import importlib.abc
import importlib.machinery
import runpy
import sys
import types

STUBBED = ("tensorflow", "waymo_open_dataset")


class _Stub(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        name = f"{self.__name__}.{k}"
        return sys.modules.get(name) or _Loader().create_module(importlib.machinery.ModuleSpec(name, None))


class _Loader(importlib.abc.Loader, importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        return importlib.machinery.ModuleSpec(name, self, is_package=True) if name.split(".")[0] in STUBBED else None

    def create_module(self, spec):
        m = _Stub(spec.name)
        m.__path__ = []
        sys.modules[spec.name] = m
        return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Loader())
sys.argv = ["tools/preprocessing/nocot_sample_generation.py", *sys.argv[1:]]
sys.path[:0] = [".", "./navsim"]
runpy.run_path("tools/preprocessing/nocot_sample_generation.py", run_name="__main__")
