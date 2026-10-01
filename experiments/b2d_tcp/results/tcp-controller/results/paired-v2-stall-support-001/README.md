# 1773 native，paired-v2中间失败审计

源为闭合4000帧attempt；保留collision3674前后各20帧的原始行（相对首控t36.2s、原timestamp36.25s），不能以附近actor中心距离推定碰撞责任。

首次collision前724帧：speed RMS1.4401，纵向|jerk|p95335.287；模型desired<.4的150帧全部在此prefix。第一次>=5s低速区间t20.55–28.30，156帧中84帧模型desired<.4，体现模型停/慢意图和执行的混合。

collision及之后3276帧：实际平均速度.00651m/s、desired平均4.463m/s，desired<.4为0；3208帧selected throttle>0且brake0，3275帧低速。连续低速从collision后一帧3675延续163.70s至结束。68个reverse guard全部在collision后；这一阶段不能再归因为旧微负速误guard，或模型持续请求停车。API applied_control与上一条selected精确吻合只支持命令交付，不代表物理力/通行空间已实现。碰撞/阻挡相关且责任仍未知。

全程|jerk|p95仅38.504，明显被长停滞稀释；prefix与全程必须并列。target仲裁3854/3999可用帧，按vendor冻结规则重算与angle_final全部吻合；平均绝对角减小13.673°。这是实现行为证据，不能证明该仲裁导致碰撞，也不是可以跳过安全回归的调参依据。后续真实转弯阶段会单独判断模型轨迹与执行误差。

完整逐帧、5s固定箱和全部低速区间见../paired-v2-stall-edition-001；六例主比较等待PI闭合，不以本段选择赢家。
