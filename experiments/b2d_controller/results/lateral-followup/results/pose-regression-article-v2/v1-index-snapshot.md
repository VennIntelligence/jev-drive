# 定位传播的中间诊断图

[完整中文图解、PNG/PDF与逐点CSV](results/pose-regression-figures-v1/README.md)。四窗全部657样本保留：328 baseline拟合、329 short留出；单个k只拟合baseline，预测特征为速度与gyro。真值中点航向仅用于定义/分离诊断目标，不可把此图当运行时无需真值的整体定位结果。

本组图独立于冻结Hermite闭环，展示描述性拟合及其残差，未部署补偿，不主张因果或跨车泛化。全部原始来源与绘图脚本hash已归档，旧结果不改写。
