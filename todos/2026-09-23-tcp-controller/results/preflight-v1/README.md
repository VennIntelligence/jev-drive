# 运行前验证

真实GPU1加载固定TCP权重：strict检查0 missing/extra，26593444参数；合成图像的一次真实网络forward输出1×4×2且finite。这是模型接口检查，不是驾驶成绩。

CPU包装器/纵向测试最终16/16，运行器回归9/9。较早13项通过日志也保留，后补三项异常输入测试。原始来源及SHA见manifest.json。
