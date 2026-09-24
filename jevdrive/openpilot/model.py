"""openpilot driving models (ONNX) on onnxruntime, with modeld's queue logic and output decoding.

Two interfaces exist (see research/lit notes on openpilot models):
  queued    small / cinque: temporal queues live inside the ONNX (state_* in, next_state_* out),
            inputs new_img (2,6,128,256) u8 = [road, wide] current frame, desire (8,), traffic_convention,
            action_t; we keep the states on the GPU and ping-pong them between two buffers.
  external  lebowski (PR #38268): img / big_img (1,12,128,256) = frames [t-4, t], desire_pulse (1,25,8),
            features_buffer (1,24,512) = hidden states [t-96 .. t-4] step 4, all fp16; queues on the host,
            ported from compile_modeld.py at openpilot 516ec1e6.
Output decoding is ported from parse_model_outputs.py, drive_helpers.py and modeld.get_action_from_model.
"""
import base64, ctypes, os, pickle
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

MODELS_DIR = Path.home() / "data/models/openpilot"
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
FRAME_SKIP = 4  # MODEL_RUN_FREQ 20 Hz / MODEL_CONTEXT_FREQ 5 Hz
DT_MDL, LONG_SMOOTH_S, MIN_STABLE_DELAY, MIN_SPEED = 0.05, 0.3, 0.3, 1.0
BACKENDS = ("cpu", "cuda", "cuda-iob", "cuda-graph", "trt", "trt-fp32", "trt-graph")


def _preload_libs():
    """CUDA/cuDNN from the nvidia-* wheels, and TensorRT 10 from tensorrt_libs (ORT's TRT EP dlopens
    libnvinfer.so.10 by name and does not search the wheel directory)."""
    ort.preload_dlls()
    try:
        import tensorrt_libs
    except ImportError:
        return
    d = os.path.dirname(tensorrt_libs.__file__)
    for lib in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
        ctypes.CDLL(os.path.join(d, lib), mode=ctypes.RTLD_GLOBAL)


_preload_libs()


def prepare_onnx(name: str) -> Path:
    """Path of an onnxruntime-loadable copy: tinygrad's custom Contiguous op (a no-op layout hint) -> Identity."""
    src = MODELS_DIR / f"{name}.onnx"
    m = onnx.load(str(src), load_external_data=False)
    custom = [n for n in m.graph.node if n.domain]
    if not custom:
        return src
    dst = MODELS_DIR / f"{name}.ort.onnx"
    if not dst.exists():
        assert {(n.domain, n.op_type) for n in custom} == {("org.tinygrad", "Contiguous")}, custom
        for n in custom:
            n.op_type, n.domain = "Identity", ""
            del n.attribute[:]
        del m.opset_import[:]
        m.opset_import.append(onnx.helper.make_opsetid("", 20))
        onnx.save(m, str(dst))
    return dst


def providers(backend: str, cache: Path):
    cuda = ("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "EXHAUSTIVE",
                                      "enable_cuda_graph": backend == "cuda-graph"})
    if backend == "cpu":
        return ["CPUExecutionProvider"]
    if backend.startswith("cuda"):
        return [cuda, "CPUExecutionProvider"]
    cache.mkdir(parents=True, exist_ok=True)
    trt = ("TensorrtExecutionProvider", {
        "device_id": 0, "trt_fp16_enable": backend != "trt-fp32", "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache), "trt_timing_cache_enable": True, "trt_timing_cache_path": str(cache),
        "trt_max_workspace_size": 8 << 30, "trt_builder_optimization_level": 3,
        "trt_cuda_graph_enable": backend == "trt-graph"})
    return [trt, ("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]


class OPModel:
    """One openpilot model stepped at 20 Hz. step() takes the packed current frames and returns the raw
    output vector (float32); decode() turns it into plan / action."""

    def __init__(self, name: str, backend: str = "cuda-iob", cache: Path | None = None, threads: int = 0):
        assert backend in BACKENDS, backend
        self.name, self.backend = name, backend
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        # GPU sessions only need a host thread to launch kernels; ORT's default pool (all cores, spinning)
        # makes concurrent sessions fight over the CPU
        so.intra_op_num_threads = threads or (0 if backend == "cpu" else 1)
        if backend != "cpu":
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(prepare_onnx(name)), so,
                                         providers=providers(backend, cache or MODELS_DIR / "trt_cache" / f"{name}-{backend}"))
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.slices = pickle.loads(base64.b64decode(meta["output_slices"]))
        self.inputs = {i.name: (tuple(i.shape), np.float16 if "float16" in i.type else
                                np.uint8 if "uint8" in i.type else np.float32) for i in self.sess.get_inputs()}
        self.queued = "new_img" in self.inputs
        self.device = backend not in ("cpu", "cuda")  # inputs/outputs/states bound on the GPU
        self.state_names = [n for n in self.inputs if n.startswith("state_")]
        self.reset()

    # ---- state ----
    def reset(self):
        self.prev_desire = np.zeros(8, np.float32)
        self.n = 0
        zeros = {n: np.zeros(s, d) for n, (s, d) in self.inputs.items()}
        if not self.queued:  # host queues, as compile_modeld.make_input_queues
            self.img_q = np.zeros((2, FRAME_SKIP + 1, 6, 128, 256), np.uint8)
            self.desire_q = np.zeros((FRAME_SKIP * 25, 8), np.float32)
            self.feat_q = np.zeros((FRAME_SKIP * 24, 512), np.float32)
            self.prev_feat = np.zeros(512, np.float32)
        if not self.device:
            self.state = {n: zeros[n] for n in self.state_names}
        elif hasattr(self, "bindings"):  # zero in place: CUDA graphs captured the buffer addresses
            for v in [*self.dev_in.values(), *(v for st in self.sets for v in st.values())]:
                v.update_inplace(np.zeros(v.shape(), v.numpy().dtype))
        else:
            self._bind(zeros)

    def _bind(self, zeros):
        ov = lambda a: ort.OrtValue.ortvalue_from_numpy(a, "cuda", 0)  # noqa: E731
        self.dev_in = {n: ov(zeros[n]) for n in self.inputs if n not in self.state_names}
        self.dev_out = ov(np.zeros(self.sess.get_outputs()[0].shape, np.float32 if self.queued else np.float16))
        # two state buffer sets; binding k reads set k and writes set 1-k, so steps alternate k = 0, 1
        self.sets = [{n: ov(zeros[n]) for n in self.state_names} for _ in range(2)]
        self.bindings = []
        for k in range(2 if self.queued else 1):
            io = self.sess.io_binding()
            for n, v in self.dev_in.items():
                io.bind_ortvalue_input(n, v)
            io.bind_ortvalue_output(self.sess.get_outputs()[0].name, self.dev_out)
            for n in self.state_names:
                io.bind_ortvalue_input(n, self.sets[k][n])
                io.bind_ortvalue_output("next_" + n, self.sets[1 - k][n])
            ro = ort.RunOptions()
            if self.backend.endswith("graph"):
                ro.add_run_config_entry("gpu_graph_id", str(k + 1))
            self.bindings.append((io, ro))

    # ---- one 20 Hz step ----
    def feeds(self, img2, desire, traffic, action_t):
        """Model inputs other than the ONNX-internal states, following modeld's rising-edge desire pulse."""
        desire = np.asarray(desire, np.float32).copy()
        desire[0] = 0
        pulse = np.where(desire - self.prev_desire > .99, desire, 0).astype(np.float32)
        self.prev_desire = desire
        tc = np.asarray(traffic, np.float32).reshape(1, 2)
        at = np.asarray(action_t, np.float32).reshape(1, 2)
        if self.queued:
            return {"new_img": img2, "desire": pulse, "traffic_convention": tc, "action_t": at}
        self.img_q = np.concatenate([self.img_q[:, 1:], img2[:, None]], 1)
        self.desire_q = np.concatenate([self.desire_q[1:], pulse[None]])
        self.feat_q = np.concatenate([self.feat_q[1:], self.prev_feat[None]])
        f16 = np.float16
        return {"img": self.img_q[0, ::FRAME_SKIP].reshape(1, 12, 128, 256),
                "big_img": self.img_q[1, ::FRAME_SKIP].reshape(1, 12, 128, 256),
                "desire_pulse": self.desire_q.reshape(25, FRAME_SKIP, 8).max(1)[None].astype(f16),
                "traffic_convention": tc.astype(f16), "action_t": at.astype(f16),
                "features_buffer": self.feat_q[::FRAME_SKIP][None].astype(f16)}

    def step(self, img2, desire=np.zeros(8), traffic=(1, 0), action_t=(0.275, 0.525)):
        f = self.feeds(img2, desire, traffic, action_t)
        if self.device:
            for n, a in f.items():
                self.dev_in[n].update_inplace(np.ascontiguousarray(a))
            io, ro = self.bindings[self.n % len(self.bindings)]
            self.sess.run_with_iobinding(io, ro)
            out = self.dev_out.numpy()[0].astype(np.float32)
        else:
            names = [o.name for o in self.sess.get_outputs()]
            res = dict(zip(names, self.sess.run(None, f | self.state)))
            out = res[names[0]][0].astype(np.float32)
            self.state = {n: res["next_" + n] for n in self.state_names}
        if not self.queued:
            self.prev_feat = out[self.slices["hidden_state"]]
        self.n += 1
        return out


# ---- output decoding ----
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -11, np.inf)))


def mdn_mu(raw, shape):
    return raw[: raw.shape[-1] // 2].reshape(shape)


def decode(raw, slices, v_ego, action_t=(0.275, 0.525)):
    """Raw output vector -> the heads we use. desired curvature / accel follow modeld (without smoothing)."""
    s = lambda k: raw[slices[k]]  # noqa: E731
    plan = mdn_mu(s("plan"), (33, 15))
    lat_t, long_t = action_t
    if "action" in slices:
        act = mdn_mu(s("action"), (2,))
        curv, accel = act[0] / max(1.0, v_ego) ** 2, act[1]
    else:
        curv = curvature_from_plan(plan[:, 11], plan[:, 14], v_ego, lat_t)
        accel = accel_from_plan(plan[:, 3], plan[:, 6], long_t)
    return dict(plan_pos=plan[:, 0:3], plan_vel=plan[:, 3:6], plan_acc=plan[:, 6:9], plan_yaw=plan[:, 11],
                curvature=float(curv), accel=float(accel),
                lead=mdn_mu(s("lead"), (3, 6, 4)), lead_prob=sigmoid(s("lead_prob")),
                lane_lines=mdn_mu(s("lane_lines"), (4, 33, 2)), lane_prob=sigmoid(s("lane_lines_prob"))[1::2],
                engaged=float(sigmoid(s("meta"))[0]))


def accel_from_plan(speeds, accels, action_t):
    v_now, a_now = speeds[0], accels[0]
    if action_t < MIN_STABLE_DELAY:
        v_target = v_now + action_t / MIN_STABLE_DELAY * (np.interp(MIN_STABLE_DELAY, T_IDXS, speeds) - v_now)
    else:
        v_target = np.interp(action_t, T_IDXS, speeds)
    return 2 * (v_target - v_now) / action_t - a_now


def curvature_from_plan(yaws, yaw_rates, v_ego, action_t):
    if action_t < MIN_STABLE_DELAY:
        psi = action_t / MIN_STABLE_DELAY * np.interp(MIN_STABLE_DELAY, T_IDXS, yaws)
    else:
        psi = np.interp(action_t, T_IDXS, yaws)
    v = max(v_ego, MIN_SPEED)
    return 2 * psi / (v * action_t) - yaw_rates[0] / v


def smooth(val, prev, tau, dt=DT_MDL):
    alpha = 1 - np.exp(-dt / tau) if tau > 0 else 1
    return alpha * val + (1 - alpha) * prev
