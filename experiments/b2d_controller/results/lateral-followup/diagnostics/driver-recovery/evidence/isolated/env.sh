# Source in the shell launching the server; no system files are changed.
export PATH=/data/tools/nvidia-userspace-580.159.03/isolated/bin:"$PATH"
export LD_LIBRARY_PATH=/data/tools/nvidia-userspace-580.159.03/isolated/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export VK_DRIVER_FILES=/data/tools/nvidia-userspace-580.159.03/isolated/nvidia_icd.json
export VK_ICD_FILENAMES=/data/tools/nvidia-userspace-580.159.03/isolated/nvidia_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/data/tools/nvidia-userspace-580.159.03/isolated/10_nvidia.json
