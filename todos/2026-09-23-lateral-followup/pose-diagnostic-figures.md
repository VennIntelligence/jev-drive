# 定位传播的中间诊断图

优先使用[文章图版 article-v2：中文图注、PNG/PDF、逐点CSV](results/pose-regression-article-v2/README.md)。本版按research共享样式使用6.875in双栏、STIXGeneral serif、Okabe–Ito/灰baseline与300dpi；两张PNG均小于500KB。

[历史v1完整图解与原图](results/pose-regression-figures-v1/README.md)原样保留，[旧索引冻结副本](results/pose-regression-article-v2/v1-index-snapshot.md)供旧hash复核。两版657曲线样本和逐窗RMS数据逐字节相同：328 baseline拟合、329 short留出；没有删瞬态或重新拟合。

本组图独立于冻结Hermite闭环，只描述传播残差的离线拟合。预测特征为速度/gyro，但系数使用baseline真值标签，且诊断目标使用真值中点航向分离；不主张因果、定位已改善或跨车泛化。共享样式模块见[research/plot_style.py](../../research/plot_style.py)，后续图可复用。
