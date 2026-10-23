import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class _Annotation:
    def __class_getitem__(cls, item):
        return object


class _FakeModel:
    def __init__(self):
        self.tokenizer = types.SimpleNamespace(eos_token="eos")
        self.cfg = types.SimpleNamespace(d_model=1)

    def requires_grad_(self, enabled):
        return self

    def state_dict(self):
        return {}


class ModelDtypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bfloat16 = object()
        cls.float32 = object()
        cls.calls = []

        hooked_transformer = type(
            "HookedTransformer",
            (),
            {
                "from_pretrained_no_processing": classmethod(
                    lambda _cls, *args, **kwargs: cls._load_model(args, kwargs)
                )
            },
        )
        torch = types.ModuleType("torch")
        torch.Tensor = type("Tensor", (), {})
        torch.dtype = type
        torch.bfloat16 = cls.bfloat16
        torch.float32 = cls.float32
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.set_grad_enabled = lambda enabled: None
        torch.nn = types.ModuleType("torch.nn")
        torch.nn.functional = types.ModuleType("torch.nn.functional")

        transformer_lens = types.ModuleType("transformer_lens")
        transformer_lens.HookedTransformer = hooked_transformer
        transformer_lens.utils = object()
        transformer_lens.ActivationCache = type("ActivationCache", (), {})
        transformer_lens.loading = object()
        hook_points = types.ModuleType("transformer_lens.hook_points")
        hook_points.HookPoint = type("HookPoint", (), {})

        jaxtyping = types.ModuleType("jaxtyping")
        jaxtyping.Float = _Annotation
        jaxtyping.Int = _Annotation

        stubs = {
            "torch": torch,
            "torch.nn": torch.nn,
            "torch.nn.functional": torch.nn.functional,
            "einops": types.ModuleType("einops"),
            "datasets": types.SimpleNamespace(load_dataset=lambda *args, **kwargs: None),
            "sklearn": types.ModuleType("sklearn"),
            "sklearn.model_selection": types.SimpleNamespace(train_test_split=lambda data, **kwargs: (data, data)),
            "tqdm": types.SimpleNamespace(tqdm=lambda value, *args, **kwargs: value),
            "transformer_lens": transformer_lens,
            "transformer_lens.hook_points": hook_points,
            "transformers": types.SimpleNamespace(AutoTokenizer=object, AutoModelForCausalLM=object),
            "jaxtyping": jaxtyping,
        }
        cls.modules_patch = patch.dict(sys.modules, stubs)
        cls.modules_patch.start()
        path = Path(__file__).resolve().parents[1] / "abliterator.py"
        cls.abliterator = types.ModuleType("abliterator_under_test")
        source = "from __future__ import annotations\n" + path.read_text()
        exec(compile(source, str(path), "exec"), cls.abliterator.__dict__)

    @classmethod
    def tearDownClass(cls):
        cls.modules_patch.stop()

    @classmethod
    def _load_model(cls, args, kwargs):
        cls.calls.append((args, kwargs))
        return _FakeModel()

    def _make_abliterator(self, **kwargs):
        return self.abliterator.ModelAbliterator(
            "model",
            ((["harmful train"], ["harmful test"]), (["harmless train"], ["harmless test"])),
            **kwargs,
        )

    def test_custom_dtype_is_forwarded_to_transformer_lens(self):
        self.calls.clear()

        self._make_abliterator(device="cpu", dtype=self.float32)

        self.assertIs(self.calls[-1][1]["dtype"], self.float32)

    def test_default_dtype_remains_bfloat16(self):
        self.calls.clear()

        self._make_abliterator(device="cpu")

        self.assertIs(self.calls[-1][1]["dtype"], self.bfloat16)


if __name__ == "__main__":
    unittest.main()
