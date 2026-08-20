# DexGraspNet2 + Wuji2

这是一个从数据生成、DexGraspNet2 训练与推理，到 LEAP→Wuji2 重定向、双臂
Isaac Lab 闭环执行的完整研究工程。当前正式入口把“抓什么”和“机械臂怎样走”
拆成两个互相独立的维度，同时保留原有训练、推理和单手验证路线。

## 当前主线

### 任务（TASK）

- `semantic-grasp`：输入物品名称；GroundingDINO + SAM 在当前 RGB 中选择目标。
- `color-sort`：输入 `red`/`blue`（也接受“红/蓝”）；在当前 RGB 中用
  GroundingDINO + SAM 匹配颜色，再与 HSV 实例相交，重复抓取该颜色的物体。

### 机械臂路线（MOTION ROUTE）

- `legacy`（Route A）：已有 flexible/keypoint q7 路径与 quintic 机械臂执行。
- `curobo`（Route B）：右臂真实 7DOF cuRobo `MotionPlanner` dense trajectory。

四种组合由同一个入口分发：

| Task | Route A / legacy | Route B / cuRobo |
|---|---:|---:|
| semantic-grasp | 保留 | 当前正式实验重点 |
| color-sort | 保留 | 当前正式实验重点 |

## 快速开始

在终端交互选择任务和路线：

```bash
cd ~/Projects/DexGraspNet2_Wuji2
./run_closed_loop.sh --sim-execute
```

显式运行语义抓取 + Route B：

```bash
./run_closed_loop.sh \
  --task semantic-grasp \
  --motion-route curobo \
  --sim-execute
```

显式运行红色分类 + Route B：

```bash
./run_closed_loop.sh \
  --task color-sort \
  --target-color red \
  --motion-route curobo \
  --color-seed 42 \
  --sim-execute
```

只规划、不执行机械臂：

```bash
./run_closed_loop.sh \
  --task semantic-grasp \
  --motion-route curobo \
  --planning-only
```

非 TTY 调用应显式提供 `--task`、`--motion-route`，color-sort 还必须提供
`--target-color`。Route B 不需要 Route A 的碰撞绕过参数。

## 正式闭环

### 语义抓取

```text
fresh Persistent Isaac capture
  -> authoritative RobotSegmenter
  -> rgb_no_robot
  -> GroundingDINO proposals
  -> robot box hard gate
  -> SAM
  -> robot mask hard gate/residual
  -> valid depth + rigid SourceZone gate
  -> selected target mask
  -> DGN2
  -> LEAP -> Wuji2
  -> shared task endpoints
  -> selected motion route
  -> Persistent Isaac execution/recovery
```

### 颜色分类

```text
fresh capture
  -> same RobotSegmenter
  -> GroundingDINO/SAM for requested color
  -> HSV masks & ~robot_mask before morphology
  -> current SourceZone instances
  -> one selected instance
  -> shared DGN2/LEAP/Wuji2/endpoints/route
  -> place, HOME, fresh capture, repeat
```

运行时红蓝材质 assignment 只用于仿真视觉和审计，不能直接选择目标。

## 冻结合同

- `robot_mask.npy` 的唯一生产者是 `RobotDepthCleaner.remove_robot()`；同一 mask
  同时服务 RGB、HSV 和 depth/point-cloud planning。
- 原始 `rgb.png`、`depth_m.npy` 永远保留；派生输出是 `rgb_no_robot.png` 和
  `filtered_depth.npy`。
- `selected_target_mask` 只代表当前被抓物，不能与 `robot_mask` 混用；它只在
  PREGRASP→COVER 等 intentional-contact 阶段从已去机器人的深度中进一步删除。
- Scene manifest 到 SourceZone 只使用平移+旋转；SourceZone 显示 scale 不参与
  物体位姿变换。资产按 dataset lineage + object identity 解析。
- Route B active joints 只允许 `arm_r_joint_1`…`arm_r_joint_7`；环境碰撞 ON；
  IK、TrajOpt、Graph self collision 全部 OFF。
- 不修改官方 Wuji2 URDF/USD 来掩盖集成错误，不放宽验收阈值来换取 PASS。

## 环境边界

| 环境 | 用途 |
|---|---|
| `graspnet2.0` | 数据、DexGraspNet2 网络、CPU/Trimesh 工具 |
| `groundedsam` | GroundingDINO + SAM |
| `curobo_v2` | cuRobo IK、RobotSegmenter、Route B MotionPlanner |
| `isaaclab22_sim50` | Isaac Sim 5.0 / Isaac Lab 2.2 persistent worker |
| project-local `wuji_retargeting` | LEAP→Wuji2 retarget |

`run_closed_loop.sh` 本身由 `isaaclab22_sim50` 启动 orchestrator，后者按合同调用
其他隔离环境。不要在另一个已打开的 Isaac/Kit 会话中再启动闭环。

## 仓库结构

```text
01_environment/                     环境合同、vendor/submodule 资产
02_training_dataset/                数据生成代码与本地大数据（原位保护）
03_prediction_network/              官方 DexGraspNet2 网络快照
04_training/                        Wuji2 20-DOF 训练与评估
05_inference/                       单视角预测与过滤
06_leap_to_wuji2_final_pipeline/    LEAP→Wuji2 重定向和冻结基线
07_wuji2_network_3p3r_sim/          原生 Wuji2 网络/独立手验证
08_dual_arm_scene_layout/           标定场景、闭环、Route A/B、Isaac 执行
src/wuji2_dgn2/                     跨阶段共享合同
trimesh/                            可再生成的几何查看工具
verified/                           历史冻结业务基线索引
docs/                               架构、数据和执行合同
```

用户维护的场景地址清单为
`02_training_dataset/TRAIN_SCENE_INPUT_PATHS_AND_OBJECTS.txt`。它不属于自动清理或
格式化范围。

## 输出、基线与排障

- 新闭环 session：
  `08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/<session>/`。
- 当前主线说明：
  `08_dual_arm_scene_layout/isaaclab_control/MAINLINE.md`。
- 实验前基线：
  `08_dual_arm_scene_layout/isaaclab_control/core/worklog/PRE_EXPERIMENT_BASELINE.md`。
- 命令和阶段结论：同目录下 `COMMAND_LOG.md`、`SESSION_SUMMARY.md`。
- 清理清单：同目录下 `CLEANUP_20260820.md`。

旧 session 只作为证据，不应复用其中的 capture、mask、candidate 或 trajectory
启动新实验。场景或机械臂状态变化后必须 fresh capture。

## 开发规则

1. 修改前检查 `git status --short` 与 `git diff --check`，保留用户已有改动。
2. cuRobo 只在 `curobo_v2`；Isaac Lab 只在 `isaaclab22_sim50`。
3. 重要命令和结论写入 worklog。
4. 旧实现移出生产路径前先归档；不要用 `git clean -fd` 清理项目。
5. 生成输出、capture、cache、视频和 scratch 不进入源码主线。

更详细的闭环实现与安全边界见
[`08_dual_arm_scene_layout/isaaclab_control/MAINLINE.md`](08_dual_arm_scene_layout/isaaclab_control/MAINLINE.md)。

## 三场景颜色验收

自动监督的三场景批次使用同一生产入口，但把全部 session、终端日志与 JSON 汇总
收进一个自动时间戳目录：

```bash
/home/lin/miniconda3/envs/isaaclab22_sim50/bin/python \
  08_dual_arm_scene_layout/isaaclab_control/closed_loop/tools/run_color_sort_campaign.py
```

默认依次运行 `scene_0000:RED`、`scene_0020:BLUE`、`scene_0065:RED`，Route B、
seed 42、headless sim-execute。输出位于
`isaaclab_control/outputs/color_sort_campaigns/<YYYYMMDD_HHMMSS>/`。根目录中的
`campaign_report.md` 是人工阅读入口，`campaign_summary.json` 是机器读取入口；
每个 case 的终端日志也带启动时间标签。某个场景发生真正进程错误时，批次会保存
错误尾部并停下供诊断；修复后对同一目录加 `--campaign-root ... --resume`，已完成
场景不会重复运行。只重跑一个已完成 case 时再加
`--rerun-case case_02_scene_0020_blue`。
