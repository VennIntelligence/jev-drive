# P3 GPU enable — 2026-09-27

User explicitly authorizes P3 build and execution on GPU1/6, overriding the old T6b role split. Scientific thresholds, registered ten scenes and human ghosting review remain unchanged.

CPU prep remains exclusively owned by840478 on182–189; no duplicate prep. Scene000 has198frames/990camera images but initially zero sky masks. Initial GPU1 sky owner866463 tried the registered SegFormer-B5 checkpoint; huggingface.co direct access failed before CUDA work. That owner exited and the continuation uses hf-mirror.com for the same repository checkpoint. The observed model config revision is2c6f153e4c23c229e2fa2b188eb250607e030cd8; no weights or model substituted. GPU1 Alpamayo587198 remains untouched.

New independent scripts/p3/gpu_enable.py owners: sky GPU1 CPU180, scene0 GPU6 CPU181, all OMP/BLAS/TF threads1, every child admission requires1280 spare tasks (1024 floor plus256 launch margin). sky processes scenes000–009 as prep arrives; scene0 waits for complete sky000 then runs isolated3000-iteration train/render/index/openpilot/exam/report, followed by default30000 scene0 and the same readout. No chain GO or visual PASS is written. Errors persist per owner and dependency waits terminate on failure or after four hours. Default chain scripts are unchanged.

Short artifacts: ckpt/nq4_p3_short3000 and processed/nq4_p3_short3000, separate done markers. Formal artifacts: ckpt/nq4_p3 and processed/nq4_p3. Registry records exact independent owners, prep, train and output roots; controller owns only the resource claim, never starts another P3 writer.

Validation: Python compile; controller15tests passed. Live pipeline proof and PID/ETA will be appended after launch. Technical run may expose untested upstream rendering/node-mapping/pivot bugs; do not claim any gate passed in advance.
