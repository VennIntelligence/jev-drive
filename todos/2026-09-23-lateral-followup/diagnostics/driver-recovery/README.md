# NVIDIA 580.159.03 用户态库隔离恢复

本方案仅适用于内核驱动 580.159.03；2026-09-23 检查时系统内核模块和用户态已匹配为 580.173.02，不得用于当前启动脚本。

内核驱动仍为 580.159.03，而自动升级后的系统用户态库为 580.173.02。本目录保存恢复辅助脚本；不改变控制器源码、系统驱动、系统库链接或 Vulkan 配置。根代理负责下载、提取及真实 NVML/CARLA 探测。

官方完整 runfile 已通过 HTTP HEAD 确认可访问，大小 398,016,015 字节；[官方 SHA256 文件](https://download.nvidia.com/XFree86/Linux-x86_64/580.159.03/NVIDIA-Linux-x86_64-580.159.03.run.sha256sum) 声明：

```text
32c85d99b0f640c9501f61b39ddad208fd0288d015c4fbc5fd0435c07783fa77  NVIDIA-Linux-x86_64-580.159.03.run
```

脚本只消费已提取目录，不下载、安装或执行驱动。需要 Python 3 和 `readelf`。以下构建命令由根代理在校验归档并完成 `--extract-only` 后执行；默认输出必须不存在。

```bash
python3 todos/2026-09-23-lateral-followup/diagnostics/driver-recovery/build_isolated.py \
  --source /data/tools/nvidia-userspace-580.159.03/extracted \
  --out /data/tools/nvidia-userspace-580.159.03/isolated

python3 todos/2026-09-23-lateral-followup/diagnostics/driver-recovery/build_isolated.py \
  --verify-only --out /data/tools/nvidia-userspace-580.159.03/isolated

source /data/tools/nvidia-userspace-580.159.03/isolated/env.sh
```

实际提取目录以根代理操作结果为准。若原始 Vulkan JSON 不在顶层，可用 `--icd-template` 指定归档内真实文件；脚本保留其中 `api_version`，只将库路径改为隔离目录绝对路径。EGL JSON 按 GLVND 1.0.0 格式生成。已有输出目录会被拒绝；失败后保留现场，后续使用新的输出目录。

构建仅复制顶层 ELF64、little-endian、x86-64 的 NVIDIA vendor 库，按 ELF SONAME 建相对软链，不复制 compat32 或中性 `libGL`、`libEGL`、`libGLX`、`libGLdispatch`、Vulkan loader。私有 `bin/nvidia-smi` 链接到提取包内同版工具，记录工具哈希与依赖；提取目录因此必须保留。各库原文件名、SHA256、SONAME、DT_NEEDED、RPATH/RUNPATH 和系统依赖均进入 manifest；NVIDIA DT_NEEDED 缺失会在创建输出前失败。复制后立即核对哈希、ELF 元数据、链接及目录内容，并产生 `static-verification.json`。此验证不调用任何 GPU API。

NVML 需要 `libnvidia-ml.so.1`；图形路径还需要同版本 GLX/EGL 以及 glcore、glsi、tls、gpucomp、glvkspirv 等库，不能用单个 NVML 库的成功推断 CARLA 已恢复。[NVIDIA 官方组件说明](https://download.nvidia.com/XFree86/Linux-x86_64/580.159.03/README/installedcomponents.html)支持保留系统已有中性 GLVND，同时使用版本匹配的 vendor 组件。DT_NEEDED 检查不能穷尽运行期 `dlopen`，因此脚本保留提取包内所有符合条件的 NVIDIA vendor 库。

`env.sh` 只影响 source 它的进程及子进程：私有 bin 优先于既有 `PATH`，私有 lib 优先于既有 `LD_LIBRARY_PATH`，设置私有 Vulkan/EGL JSON，不修改 GPU 选择或渲染参数。现有 Server.start 会重设 `VK_ICD_FILENAMES` 为系统 JSON，但其中裸名 `libGLX_nvidia.so.0` 仍通过该库路径解析。较新 loader 优先使用 `VK_DRIVER_FILES`，见 [Khronos 官方说明](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md)。EGL 的 `__EGL_VENDOR_LIBRARY_FILENAMES` 行为见 [NVIDIA GLVND 说明](https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)。

真实恢复验收由根代理进行：核对下载归档 SHA256、静态验证结果、NVML/CARLA 探测退出状态，并保存目标 CARLA 进程加载库的路径证据。应确认 NVIDIA 库来自隔离 580.159.03，不能仅凭环境变量存在或静态检查通过宣称真实加载成功。初版脚本尚未执行，随后根代理已完成下述真实恢复验收。


## 已完成的恢复与限制

官方完整包 SHA256逐字匹配，仅执行 `--extract-only`，没有运行安装器。私有目录包含37个vendor库、22条SONAME软链；静态验证通过。NVML恢复读取两个GPU的UUID和580.159.03驱动版本。真实CARLA PID384002的9个vendor/loader映射已保存，GLX等NVIDIA组件确实来自隔离目录，未见580.173.02混入；实际使用GPU1 UUID `GPU-b90dd90e-394b-7800-f23f-5892a8e3d0f1`。

`aim-g2-v2`六例闭环已全部completed且G2通过，start→end49.767866s。它仍需独立逐窗候选验收，不能把恢复成功或G2通过当作控制性能通过。`aim-g2-v1`为启动0例超时182.182474s；root随后想发送SIGTERM时进程已经退出，原intervention只记录意图，补充intervention-result明确没有实际信号交付。v2首次tmux多行命令引用失败也保留launcher-error，之后使用固定launch.sh成功，未丢弃驾驶case。

系统本身仍保留自动升级后的580.173.02库，未source私有env的普通nvidia-smi仍报告不匹配。没有重启、卸载内核模块、改桌面或系统库；后续同内核实验须使用上面的env.sh。若机器将来重启加载其他驱动版本，需重新核对匹配关系，不能盲用旧私有目录。

[完整恢复证据索引](evidence/manifest.json)包括下载中断/补块日志、官方校验、库清单、真实进程maps与apt升级记录。400MB原包、提取目录和库均留在 `/data/tools/nvidia-userspace-580.159.03`，不进Git。
