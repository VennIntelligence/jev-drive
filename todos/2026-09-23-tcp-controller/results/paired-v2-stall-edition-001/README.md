# TCP停滞与原生目标点仲裁审计

全程、严格首次collision前、collision及之后、从首受控tick起每5s固定箱，以及全部实际跨度>=5s低速(|truth longitudinal speed|<.5m/s)连续区间。区间仅诊断，不替代完整路线主指标；边界规则不依结果择优。表保留各箱的实际跨度与全部低速样本，停滞可稀释全程jerk，必须连同prefix看。

model_stop_intent为desired<.4m/s；command_drive为selected throttle>0且brake0。两者与实际低速分别计数。原生target仲裁按已归档vendor source重算：|angle_target|<|angle|，或|angle_target-angle_last|>.3且target forward<10m；归一化angle乘90转degree，与metadata.angle_final核对。attenuation正数表示绝对角减小，负数表示增大；不是控制效果因果证据。

附近actor仅5Hz中心距离，不能据此判bbox碰撞/责任。模型物理原点未确认；局部waypoint/aim/target不能伪装精确监督真值。applied_control对上一条命令仅检验API控制值交付，不证明物理驱动力或可通行空间。
