"""Training-side audit: are ap2_prep's cold tokens (rule zero) the backwarp tokens of the same real slots, except the oldest real slot?
Compares against the encoder's own batch-size noise (the same pairs encoded at batch 128 and 32). Prints JSON."""
import sys, os, json, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import prep_bench as B  # noqa: E402
import numpy as np  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402


def main(n=32):
    import torch
    import ap2_prep as AP
    import pp_prep as PP
    import sh30_core as C
    from jevdrive import op_adapt as A
    sel, ents, pose, vel, W, cam, _ = B.rows(n, seed=1)
    net, enc = PP.encoder(torch.device("cuda"))
    with ProcessPoolExecutor(max(1, len(os.sched_getaffinity(0)) - 1)) as pool:
        res = list(pool.map(AP._job, [(ents[i], pose[i], vel[i], W[i], cam[i], True) for i in range(n)]))
    cf, bf = np.stack([r[0] for r in res]), np.stack([r[1] for r in res])
    f32 = lambda x: x.astype(np.float32)  # noqa: E731
    out = {}
    tb = {bs: enc(*AP.pairs(bf.reshape(n * 3, 8, *C.FRAME)), bs=bs).reshape(n, 3, 8, *A.H_SHAPE) for bs in (128, 32)}
    tc = {bs: np.concatenate([enc(*AP.pairs(cf[:, sl]), bs=bs).reshape(n, AP.AI.N_SLOT[m], *A.H_SHAPE) for m, sl in AP.COLD_AT.items()], 1) for bs in (128, 32)}
    first = np.stack([cf[:, AP.COLD_AT[m]][:, 0] for m in (1, 2, 3)], 1).reshape(-1, *C.FRAME)
    hf = enc(np.zeros_like(first), first, bs=128).reshape(n, 3, *A.H_SHAPE)
    comp = np.concatenate([np.concatenate([hf[:, m - 1][:, None], tb[128][:, m - 1, 8 - AP.AI.N_SLOT[m] + 1:]], 1) for m in (1, 2, 3)], 1)
    rms = float(np.sqrt((f32(tc[128]) ** 2).mean()))
    d = lambda x, y: dict(max_abs=float(np.abs(f32(x) - f32(y)).max()), rel=float(np.linalg.norm(f32(x) - f32(y)) / np.linalg.norm(f32(x))),  # noqa: E731
                          identical_values_share=float((x == y).mean()))
    out = dict(n=n, token_rms=rms, cold_bs128_vs_bs32=d(tc[128], tc[32]), bw_bs128_vs_bs32=d(tb[128], tb[32]), cold_today_vs_dedupe=d(tc[128], comp),
               frames_real_slots_equal=bool(all(np.array_equal(cf[:, AP.COLD_AT[m]], bf[:, m - 1, 8 - AP.AI.N_SLOT[m]:]) for m in (1, 2, 3))),
               per_slot_max_abs_today_vs_dedupe=np.abs(f32(tc[128]) - f32(comp)).max((0, 2, 3)).round(4).tolist())
    print(json.dumps(out, indent=1), flush=True)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
