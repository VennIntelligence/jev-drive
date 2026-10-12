"""op_torch LoRA on Conv kernels (pointwise, k x k, depthwise trainable) and write_merged, on a tiny synthetic graph.
python -m unittest tests.test_op_torch_lora -v   (needs torch + onnx; CPU)"""
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _graph(path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(0)
    w = {"stem": rng.normal(size=(8, 3, 2, 2)), "dw": rng.normal(size=(8, 1, 3, 3)), "pw": rng.normal(size=(4, 8, 1, 1)),
         "mm": rng.normal(size=(4, 5))}
    inits = [numpy_helper.from_array(v.astype(np.float16), k) for k, v in w.items()]
    nodes = [helper.make_node("Cast", ["x"], ["xh"], to=TensorProto.FLOAT16),
             helper.make_node("Conv", ["xh", "stem"], ["a"], strides=[2, 2]),
             helper.make_node("Conv", ["a", "dw"], ["b"], group=8, pads=[1, 1, 1, 1]),
             helper.make_node("Conv", ["b", "pw"], ["c"]),
             helper.make_node("Transpose", ["c"], ["d"], perm=[0, 2, 3, 1]),
             helper.make_node("MatMul", ["d", "mm"], ["y"])]
    g = helper.make_graph(nodes, "t", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 4, 4, 5])], inits)
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), str(path))


class ConvLora(unittest.TestCase):
    def test_conv_lora_and_merge(self):
        import torch
        from jevdrive.op_torch import OnnxTorch, write_merged
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "m.onnx", Path(d) / "merged.onnx"
            _graph(src)
            x = torch.randn(1, 3, 8, 8)
            ref = OnnxTorch(src).run({"x": x}, ["y"])["y"]
            m = OnnxTorch(src, trainable=["dw"], lora={"stem": 2, "pw": 2, "mm": 2}, lora_alpha=4)
            self.assertTrue(torch.equal(m.run({"x": x}, ["y"])["y"], ref))          # B = 0: bitwise the original
            with torch.no_grad():
                for k, v in m.lora.items():
                    if k.endswith("__B"):
                        v.normal_()
                m.params["dw"].add_(0.1)
            y = m.run({"x": x}, ["y"])["y"]
            self.assertFalse(torch.allclose(y, ref))
            y.float().sum().backward()                                              # gradients reach A, B and the depthwise master
            self.assertTrue(all(p.grad is not None for p in m.lora.values()) and m.params["dw"].grad is not None)
            w = m.merged()
            self.assertEqual(set(w), {"stem", "pw", "mm", "dw"})
            write_merged(src, dst, w)
            m2 = OnnxTorch(dst)
            # fp32 run of both: unmerged LoRA vs the merged file differ only by the fp16 storage of the merged weights
            y2 = m2.run({"x": x}, ["y"])["y"].float()
            self.assertLess(float((y2 - y.detach().float()).abs().max()), 0.05 * float(y.abs().max()))
            with self.assertRaises(AssertionError):
                write_merged(src, src, w)


if __name__ == "__main__":
    unittest.main()
