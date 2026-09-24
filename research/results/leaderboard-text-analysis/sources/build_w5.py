import csv
from pathlib import Path
from urllib.parse import quote

root=Path('/data/hack_audit')
papers={
'LinkVLA':'2603.01441','SteerVLA':'2602.08440','FIVE-VLA':'2609.18623','RoG-DAgger':'2608.24525',
'CarLLaVA':'2406.10165','SimLingo-BASE':'2503.09594','Kyber-E2E':'2405.01394','DriveFuture':'2605.09701',
'Senna':'2410.22313','OmniSpace':'2606.22617','LVLDrive':'2512.24331','NTR':'2605.31116','DrivoR':'2601.05083'}
code={
'LinkVLA':('未检出对应官方方法仓库；仅论文可核','https://github.com/search?q=LinkVLA&type=repositories'),
'SteerVLA':('项目页标注 Code (Coming Soon)，仅项目网页仓库','https://steervla.github.io/'),
'FIVE-VLA':('未检出对应官方方法仓库；仅论文可核','https://github.com/search?q=FIVE-VLA&type=repositories'),
'RoG-DAgger':('未检出对应官方方法仓库；仅论文可核','https://github.com/search?q=RoG-DAgger&type=repositories'),
'CarLLaVA':('同仓库公开 SimLingo-Base 训练目录及共用闭环 agent；未找到能对应旧 CarLLaVA 官方 6.87 DS 的 checkpoint/提交配置','https://github.com/RenzKa/simlingo/blob/743b243afd6cf5ff51b9fa1f8cac86f22d569684/README.md#L86-L93; https://github.com/RenzKa/simlingo/blob/743b243afd6cf5ff51b9fa1f8cac86f22d569684/README.md#L193'),
'SimLingo-BASE':('同仓库公开 Base 训练目录及共用闭环 agent；未找到能对应旧 Base 官方 6.25 DS 的 checkpoint/提交配置','https://github.com/RenzKa/simlingo/blob/743b243afd6cf5ff51b9fa1f8cac86f22d569684/README.md#L86-L93; https://github.com/RenzKa/simlingo/blob/743b243afd6cf5ff51b9fa1f8cac86f22d569684/README.md#L193'),
'Kyber-E2E':('未检出该提交的官方 agent 仓库；仅论文可核','https://github.com/search?q=Kyber-E2E&type=repositories'),
'DriveFuture':('未检出对应官方方法仓库；仅论文可核','https://github.com/search?q=DriveFuture+driving&type=repositories'),
'Senna':('官方仓库公开 Senna-VLM 代码与权重；未见对应 0.22 L2 的 Senna-E2E 规划实现','https://github.com/hustvl/Senna/blob/31a3a2336e9494c254b612665fe55e95f67e9cae/README.md#L25-L34'),
'OmniSpace':('未检出该论文作者的官方规划代码仓库；同名 GitHub 搜索结果与论文无关','https://github.com/search?q=OmniSpace+driving&type=repositories'),
'LVLDrive':('未检出对应官方方法仓库；仅论文可核','https://github.com/search?q=LVLDrive&type=repositories'),
'NTR':('未检出对应官方方法仓库；仅论文可核','https://github.com/search?q=NTR+driving&type=repositories'),
'Poutine':('未检出对应模型仓库；技术报告公开','https://github.com/search?q=Poutine+driving&type=repositories'),
'DrivoR':('官方 NAVSIM 实现公开；未见 HUGSIM 适配/评测入口','https://github.com/valeoai/DrivoR/blob/fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a/README.md#L140-L185')}
rows=[]
def add(board,method,category,design,effect,page,locator,notes='',extra_source=''):
 if method=='Poutine': url='https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%20Special%20Mention%20-%20Poutine.pdf'
 else: url='https://arxiv.org/pdf/'+papers[method]
 source=f'{url}#page={page} ({locator}, PDF p.{page})'
 if extra_source:
  for page_ref in str(extra_source).split(','):
   source+=f'; {url}#page={page_ref}'
 rows.append(dict(id=f'W5-{len(rows)+1:03d}',board=board,method=method,category=category,design_disclosed=design,reported_score_effect=effect,source=source,disclosure_scope='仅论文披露，无代码可核',code_status=code[method][0],code_status_date='2026-09-24',code_status_source=code[method][1],notes=notes))

add('bench2drive','LinkVLA','language_action_alignment_training','共享离散 action/language codebook，并用轨迹描述的反向辅助目标对齐语言和动作','Table 5：tokenization 后 DS 85.07→89.57；再加 alignment 89.85→91.01',8,'Table 5; §3.1–3.2 pp.4–5','逐级消融，后一增益以已启用 C2F 为基线；不能将多项差值相加成独立效应','3,4,5')
add('bench2drive','LinkVLA','efficient_action_decoding','先预测轨迹终点，再并行细化轨迹，替代逐 token 自回归输出','Table 5：启用 C2F DS 89.57→89.85；SR 73.18→72.27；论文另报推理时间',8,'Table 5; Table 2 p.7','同一模型训练配置是否一致需看实验设置')
add('bench2drive','LinkVLA','output_representation','轨迹离散 token 的 soft label 考虑邻近 token','Table 6：DS 90.85→91.01，SR 72.73→74.55',8,'Table 6')
add('bench2drive','LinkVLA','navigation_input_conditioning','推理导航输入条件比较 GPS target points 与 navigation commands','Table 7：GPS DS 91.01/SR 74.55；命令 DS 91.25/SR 73.18',8,'Table 7','两个指标方向不同，不判哪种更好')

add('bench2drive','SteerVLA','hierarchical_semantic_control','高层 VLM 根据图像、车况与路由输出 meta-action/理由，低层 VLA 按语义指令预测 waypoint','Table 1 给组件消融；正文报告整体 DS 91.00',6,'Fig. 2; Table 1','高层推理延迟论文报 2.51 s，实际部署效应留综合阶段')
add('bench2drive','SteerVLA','language_action_alignment_training','用 VLM 对驾驶轨迹生成密集、事后可得的细粒度语义动作标签，训练高低层接口','Table 1 对比组件；正文称标签对 steerability 有明显作用',5,'§4.2; Fig. 2','训练标签引用轨迹；未见测试时未来真值输入证据')

add('bench2drive','FIVE-VLA','pretrained_backbone','FastViTHD 高分辨率视觉编码器替换 InternViT，降低视觉 token 数','Table 7：DS 85.88→88.49，SR 68.18→73.03；T4 fps 0.50→2.20',11,'Table 7','与 SimLingo 原骨干对照；组件与模型规模同时变化')
add('bench2drive','FIVE-VLA','efficient_action_decoding','efficient driving 模式绕过推理时文本自回归，单次前向输出动作','Table 7：DS/SR 仍 88.49/73.03；fps 2.20→5.35',11,'Table 7','报告的收益为速度，没有独立 DS 增益')
add('bench2drive','FIVE-VLA','recurrent_action_memory','RAM 用前一时刻 action token 作记忆；推理采用累积模式','Table 7：DS 88.49→90.95，SR 73.03→77.27；Table 8 对比推理模式',11,'Tables 7–8')

add('bench2drive','RoG-DAgger','dagger_expert_posttraining','在 student 闭环访问状态查询特权 expert；混合新旧数据做 DAgger 后训练','Table 1：SimLingo 基线到 RoG-DAgger DS +5.3，SR +6.2 个百分点',6,'Table 1','论文还报告 Longest6 与 Fail2Drive；不等同 Bench2Drive 增益。实现也把原 SimLingo 的 InternVL-2 backbone 换成 Qwen3-VL-2B，这一跨行差值不是严格单因素 DAgger 消融','3')
add('bench2drive','RoG-DAgger','dagger_expert_posttraining','用自车/他车短时动力学 rollout 扩充 expert 轨迹和速度候选，形成可预防失误的监督','Table 3 消融轨迹-速度扩展',6,'Fig. 2 p.2; Table 3 p.6','expert 在训练阶段，未表明测试时特权访问')
add('bench2drive','RoG-DAgger','dagger_expert_posttraining','通过 rollout 可挽回性决定 expert 接管时点，并对齐 expert/student 可见范围','Table 3 分别消融 solvability-aware trigger 与 FoV alignment',6,'Table 3','与训练后模型性能关系由表给出；不同触发策略不可直接归因于单独 expert')

add('carla_lb2','CarLLaVA','metric_early_termination','按已行驶距离主动停止路线，且靠近路口时根据转向角避免在路口停下；论文明确以 DS 性质作动机','Table 2c：1300m DS 3.93，2100m 6.87，2400m 6.35',4,'Table 2c; Metrics and Implementation Details','论文 p.5 另报重复提交方差；不能把单次最优当稳定效应')
add('carla_lb2','CarLLaVA','output_representation','预测 path 与 target speed 的半解耦输出，path 用于横向控制','Table 2a：waypoint-only DS 3.21→+Path 4.49',4,'Table 2a','论文将静态布局碰撞 0.68→0.0 也列在同表')
add('carla_lb2','CarLLaVA','dataset_sampling_or_source','从 PDM-lite 特权专家收集训练样本，并按起步、转向、障碍及风险事件分桶抽样','论文说明采样；未见单独对分桶策略的同配置 DS 消融',3,'§3 Dataset; §4 p.4','不把特权训练专家等同于测试时读取特权状态')
add('carla_lb2','CarLLaVA','pretrained_backbone','采用 CLIP 预训练视觉编码器和语言模型预训练','Table 2b：去预训练 DS 0.45；完整 6.87；ResNet-34 2.71',4,'Table 2b','多个骨干差异，不能仅归因于预训练')

add('carla_lb2','SimLingo-BASE','metric_early_termination','沿用按距离阈值提前停止的 LB2 推理设置，论文称以此抵消 DS 非线性扣分','Appendix Table 9c 给不同阈值的官方 DS 对照',17,'Appendix C.1; Table 9c','与 CarLLaVA 同一研究组方法，避免重复计算为独立样本')
add('carla_lb2','SimLingo-BASE','output_representation','path/target-speed 解耦控制输出，并用 PID 转车控','Appendix Table 9a：waypoint-only DS 3.21→加 path 后 4.49',17,'Table 9a; §3.3 p.4','BASE 与完整 SimLingo 分数不可混同','4')
add('carla_lb2','SimLingo-BASE','dataset_sampling_or_source','PDM-lite 专家采集训练数据、按事件类型分桶；Bench2Drive 对照另调 controller','Table 2 p.7：同数据组成+调 controller 的 DS 63.45；与原数据 45.65 不同',7,'Table 2','表内不同数据、专家与控制器组合，不能单独归因')

add('carla_lb2','Kyber-E2E','manual_motion_planner','分模块定位、检测、跟踪、预测、行为规划；采样 11 条 Bezier 轨迹并以手工 cost 排序','Table I 逐模块替换展示 DS/RC/IS 差异',5,'Fig. 1 p.1; Table I p.5','论文成本权重来自 InD 数据 IRL；不等同直接优化 LB2 官方计分公式')
add('carla_lb2','Kyber-E2E','dataset_sampling_or_source','用 nuScenes、InD 与 CARLA 数据分别训练检测和规划组件，解决 LB2 缺专家数据','Table I 显示各模块性能；未见单项数据源同配置消融',4,'§IV Training; Table I p.5','模块替换伴随数据来源变化')
add('carla_lb2','Kyber-E2E','manual_motion_planner','行为规则处理交通标识、让行与阻塞车道；PID 控制器执行规划轨迹','Table I 比较 privileged 与 sensor-based 模块',5,'§III-D p.3; Table I p.5','未审代码；不推断测试期还有论文未披露规则')

add('navsim_v2','DriveFuture','future_latent_conditioning','训练时用 GT future latent 对规划器提供条件，推理时改用预测 future latent，并逐步退火对齐','Table 4：future supervision 消融报告 EPDMS 变化',8,'Fig. 1 p.2; Table 4 p.8','论文明确测试时不用 GT future，不能编码 future_label_conditioning')
add('navsim_v2','DriveFuture','future_latent_conditioning','基于 BEV latent 预测未来，再用 diffusion decoder 生成多模态轨迹候选','Table 4/5 报消融与敏感性',8,'Fig. 2 p.4; Tables 4–5 pp.8–9','未把各组件效应合并')
add('navsim_v2','DriveFuture','learned_candidate_scoring','论文实现段称默认生成 100 个 proposal，并使用 GTRS-Dense scorer 选轨','论文主表 navhard EPDMS 55.5；Table 4 消融明确排除 GTRS-Dense scorer',8,'§4 Implementation; Tables 1 and 4','此论文未交代 scorer 的训练目标；其对 55.5 的独立增益也未由 Table 4 给出')

add('nuscenes','Senna','language_action_alignment_training','Senna-VLM 预测高层 meta-actions，Senna-E2E 将其用于轨迹规划；训练标签源自未来轨迹','Table II：完整 Senna* avg L2 0.22；不同变体需按表比较',6,'Fig. 2 p.2; Table II p.6','训练时未来轨迹生成 meta-action 标签不等于测试时未来标签输入')
add('nuscenes','Senna','dataset_sampling_or_source','DriveX 大规模预训练后再用 nuScenes 微调；混合预训练、驾驶微调、规划微调三阶段','Tables VII/IX 报数据规模和训练流程消融',8,'Tables VII p.7 and IX p.8','不同数据规模与目标的独立贡献需看 W1')
add('nuscenes','Senna','ego_state_fusion','论文 Table II 把使用 ego status 的变体标 *，并报告不使用 ego status 的对照','Table II 直接列有/无 ego status 的 L2 与碰撞率',6,'Table II','0.22 所在 * 行用 ego status；跨论文 nuScenes 口径仍待核')

add('nuscenes','OmniSpace','geometry_supervision','推理端注入相机 pose/Plücker rays，跨视角 epipolar attention 限制几何相容注意力','Table 4 比较模块对 L2/碰撞/交叉率影响',8,'Fig. 1 p.2; Table 4 p.8','保持 2D 图像推理；外部 3D teacher 仅训练时用')
add('nuscenes','OmniSpace','geometry_supervision','用 VGGT 3D teacher、跨时间聚合与蒸馏 loss 训练视觉表征','Table 4–5 报模块与 temporal sampling 消融',8,'§3.3 pp.5–6; Tables 4–5 p.8','teacher 推理时移除，不认作测试时额外 3D 模型')
add('nuscenes','OmniSpace','pretrained_backbone','从 Qwen2.5-VL 等 MLLM 起点微调，原 vision encoder/projector 冻结','Table 1 报不同 backbone 的 nuScenes 开环分数',6,'Table 1','表的不同模型大小/架构不能视为单一变量实验')

add('nuscenes','LVLDrive','multimodal_sensor_fusion','融合 LiDAR、环视图像与文本，Q-Former 用渐进门控引入点云特征','Table 3 对输入模态、门控结构做消融',7,'Fig. 3 p.4; Table 3 p.7','与 camera-only 排行方法的传感器协议不同')
add('nuscenes','LVLDrive','language_action_alignment_training','构造 spatial-aware QA 训练集，用规划、空间感知与 grounding 问答辅助训练','Table 4 报数据组合对 L2/碰撞/交叉率影响',8,'Table 4','训练标签由 nuScenes 标注导出；未证实测试期读未来真值')
add('nuscenes','LVLDrive','evaluation_metric_variant','论文采用 BEV-Planner 式 improved collision rate：某时刻碰撞则其后时刻也记碰撞','Table 1：该口径下平均 collision rate 0.25%；无法从表分离口径本身的增减',7,'§4.2 Metrics; Table 1','跨论文碰撞率需先对齐实现；论文还以 QA 格式输出六个 waypoint')

add('wod_e2e','NTR','masked_scene_token_reconstruction','masked latent 重建只通过紧凑 scene token bottleneck 反向传播；训练期 EMA teacher 给目标','Table 4 比较随机重建、EMA teacher 等变体 RFS',8,'Fig. 2 p.4; Table 4 p.8','推理时重建分支移除')
add('wod_e2e','NTR','masked_scene_token_reconstruction','SAM3 语义 mask 用于决定重建位置，使训练监督聚焦道路参与者、可行驶区域及交通控制元素','Table 4 比较随机重建位置与语义先验 RFS',8,'Table 4; §3.3 pp.5–6; SAM3 p.3','语义模型仅用于训练标注','3,5,6')
add('wod_e2e','NTR','learned_candidate_scoring','沿用 DrivoR 式 trajectory proposal 和 trajectory-conditioned scorer 从候选中择轨','Table 1 主分数；论文无该 scorer 的同配置单独消融',7,'Fig. 2 p.4; Table 1 p.7','论文没有说明评分头以 WOD RFS 或其子项为训练目标，不能归入 metric_proxy_candidate_selection','4')

add('wod_e2e','Poutine','dataset_sampling_or_source','先在 CoVLA 日本道路影像与语言轨迹上预训练，再用 WOD-E2E 训练样本做 VLT 预训练','Table 1：CoVLA-only zero-shot RFS 7.74；完整预训练基座 val RFS 8.12',4,'Table 1','7.74 与 8.12 不同训练语料和阶段，不是单因素因果')
add('wod_e2e','Poutine','language_action_alignment_training','用未来专家轨迹生成关键物体描述、行为解释和 meta-behavior，作为训练期语言监督','Table 1：No Language 变体低于完整 Poutine-Base',4,'Fig. 3 p.3; Table 1 p.4','未来轨迹只用于生成训练标注，未见测试时读取未来真值')
add('wod_e2e','Poutine','metric_reward_finetuning','GRPO 后训练直接将归一化官方 RFS 和格式奖励用于优化策略','论文 p.4：test RFS 从预 RL 的 7.91 到后 RL 的 7.99；Fig. 5 另画验证集保留 63 例的曲线',4,'§2.3 p.3; Figs. 1 and 5','RL 使用 416/479 个 preference-labeled validation scenarios；与正式 test 的独立性待综合')
add('wod_e2e','Poutine','inference_reasoning_mode','Poutine-Base 比较 CoT 开关；最终推理段称 CoT 用 1e-6 temperature，轨迹用 greedy decoding，并移除 intent 条件','Table 1：Poutine-Base no-CoT 8.12，CoT 8.08（val）；不代表最终提交不生成 CoT',4,'Table 1; Implementation Details','最终提交的 CoT 路径与基座对照是否完全同配置待核，不作提速结论')

add('hugsim','DrivoR','metric_proxy_candidate_selection','预测 PDM 各子分数，在生成的固定候选轨迹中选分数最高者','论文 Table 2 报 HUGSIM RC/HDS，但无 HUGSIM 专门 scorer 消融',5,'Fig. 2 p.2; Table 2 p.5','PDMS 代理在 NAVSIM 训练，HUGSIM 相对收益未经单独验证')
add('hugsim','DrivoR','pretrained_backbone','DINOv2 初始化视觉编码器','Table 4a 在 NAVSIM-v1 上报预训练消融；不能直接分解 HUGSIM 分数',7,'Table 4a','论文 Table 2 称使用 NAVSIM-v1 评测的 DrivoR 模型零样本评 HUGSIM；确切 checkpoint 文件和 HUGSIM 入口尚无代码核验','5')
add('hugsim','DrivoR','manual_scorer_reweighting','论文允许推理时重新组合预测子分数以调节保守/进度偏好','Table 10 p.14 给 v2 推理权重；HUGSIM 本身未说明采用哪组权重',14,'§3.4 p.4; Table 10 p.14','列为披露的可调设计，不认定是 HUGSIM 提交所用配置')
add('hugsim','DrivoR','scene_token_compression','用相机专属 register token 压缩视觉 patch token，再供轨迹生成及评分','Table 4b 在 NAVSIM-v1 上比较压缩方式；HUGSIM 无该机制单独消融',7,'Table 4b','论文 Table 2 称 NAVSIM-v1 模型用于 HUGSIM 零样本评测；具体 checkpoint 文件无代码核验','5')

out=root/'out2/w5_paper_only.csv'
with out.open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]))
 w.writeheader();w.writerows(rows)
print(len(rows),'rows',len({r['method'] for r in rows}),'methods')
