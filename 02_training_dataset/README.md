# 02 训练数据集

正式数据链：单物体Wuji2 `q_opt`与六方向验证来源 → 稳定多物体场景 → 单视角40000点 → 场景碰撞过滤 → 参考点与graspness → 单视角标签。

场景碰撞过滤统一使用已确认的Wuji2虎口接近方向：在当前`q_grasp`下，从语义掌心中心指向拇指尖与食指尖中点；PREGRASP沿反方向后退100 mm。训练数据和推理执行调用同一份`src/wuji2_dgn2/collision.py`实现。

## 当前正式数据生产

- 配置物体池：60种物体、60个语义类别；当前100场景实际覆盖59种，缺少`id=55`的Pizza。
- 单场景：随机6个不同物体。
- 训练场景：100个。
- 每场景视角：256个。
- 每视角网络点云：40000点。
- 训练视角总数：25600。
- 监督关节：优化后、力调整前的`pre_force_joint_positions_rad`。
- 100个场景全部属于训练集；验证集和测试集必须使用另外生成、场景族不重叠的数据。

正式大数据已复制到`data/scene_datasets/wuji2_train60_100seminal_256view_v1`，配置中的`output_root`是项目根相对路径。不要重复启动完整场景生成器或覆盖这份已验收数据。

## 力调整后关节监督A/B数据集

第二套完整训练集位于：

```text
data/scene_datasets/wuji2_train60_100seminal_256view_force_adjusted_legacy_v1
```

它严格复用上述100个场景、256视角、点云、物体位姿、碰撞过滤、参考点和
graspness，只把20关节监督从`pre_force_joint_positions_rad`切换为
`joint_positions_rad`。后者是Wuji2 1.0力调整后的命令关节目标，不是真机传感器
测得的最终关节状态。

配置文件：

```text
config/wuji2_train60_100seminal_256view_force_adjusted_legacy_v1.json
```

详细的单变量合同和重建命令见
`variants/force_adjusted_legacy_v1/README.md`。两套监督不能在同一训练任务中混用。

## 独立10场景测试集

配置文件：`config/wuji2_test60_10upright_10view_v1.json`。

- 输出目录：`data/scene_datasets/wuji2_test60_10upright_10view_v1`；
- 与100场景训练集使用不同随机种子，不复用训练场景或训练视角；
- 10个场景全部属于`test`，每场景6个物体；
- 每个候选至少2个物体采用经过定义和记录的立放稳定姿态；
- 仍需通过与训练集相同的Isaac Sim稳定性、桌面边界和穿透验收；
- 每场景从官方256视角轨迹近似等间隔抽取10个视角，每视角40000点；
- Stage01、02、02B、03、04与训练集使用同一套代码和参数。

该测试集继承q_opt基线配置，Stage01中`qpos == qpos_pre_force`；它没有被派生为
力调整后关节监督版本。因此它可以直接用于最新模型的单视角推理与Isaac Sim物理
测试，但不能用来报告力调整模型的`loss_joint`或包含该项的总损失。若要做同口径
定量评估，必须另外派生保持场景/点云不变、仅令`qpos == qpos_force_adjusted`的
测试标签版本。

当前结果为10个稳定场景、100个已采集视角、8765条场景安全抓取。加载器按与训练集相同的规则排除没有可映射抓取标签的`scene_0001/view_0000`，因此用于网络定量评估的有效单视角样本为99个；该视角的原始相机数据没有删除。

## 代码执行顺序

1. `generate_stable_pose_candidates.py`：从60物体池构造稳定姿态候选。
2. `generate_scenes_and_views.py`：Isaac Sim 5.0稳定性验收并拍摄256视角。
3. `build_wuji2_scene_grasps.py`：把单物体有效抓取转换到场景。
4. `filter_wuji2_scene_collisions.py`：按统一虎口PREGRASP做场景/桌面碰撞过滤。
5. `filter_wuji2_palm_center_paths.py`：执行明确标记的增强掌心路径过滤。
6. `build_wuji2_reference_graspness.py`：生成完整表面参考点与graspness。
7. `assign_wuji2_single_view_labels.py`：把标签映射到各个单视角点云。
8. `prepare_wuji2_training_dataset.py`：整理网络训练索引。
9. `check_wuji2_dataset.py`：训练前完整性验收。

## 场景生成期间流式筛选

`stream_filter_completed_scenes.py`可以和Isaac Sim场景/相机生成同时运行。它只把
已经出现`scene_manifest.json`的场景视为完成场景，因此不会读取正在写入的当前
场景，也不会修改`scenes/`。默认使用CPU、4个线程和较低进程优先级，避免与
Isaac Sim争抢RTX 4070：

```bash
/home/lin/miniconda3/envs/graspnet2.0/bin/python \
  02_training_dataset/code/stream_filter_completed_scenes.py \
  --device cpu \
  --batch-size 16 \
  --torch-threads 4 \
  --through path
```

只检查它将执行什么而不写文件：

```bash
/home/lin/miniconda3/envs/graspnet2.0/bin/python \
  02_training_dataset/code/stream_filter_completed_scenes.py \
  --once --dry-run --through path
```

场景生成结束后，可以停止CPU版本并改用`--device cuda:0 --batch-size 128`加速
未完成场景。场景生成仍在运行时，程序默认拒绝共享GPU；只有显式加入
`--allow-gpu-sharing`才会放行，不建议在8 GB显存的当前机器上这样做。

筛选是非破坏式的：Stage 02和02B不再执行`arrays[keep]`删除行，而是完整保存
Stage 01中所有单物体有效位姿，并增加：

```text
paper_keep_mask
wuji2_safe_keep_mask
minimum_scene_clearance_m
minimum_table_clearance_m
paper_reject_reason_bits
wuji2_reject_reason_bits
```

Stage 03才按照`grasp_label_generation.training_selection_mask`物化训练子集。默认
使用`wuji2_safe_keep_mask`；复现纯论文筛选时可运行
`build_wuji2_reference_graspness.py --selection-mask paper_keep_mask`。

重复使用的每物体1000个表面点、25 mm张开IK结果和随`q_grasp`变化的虎口方向
缓存在`grasp_label_stages/_cache/`。缓存合同包含输入qpos哈希和全部IK参数，参数
或位姿变化后会自动生成新缓存，不会误用旧结果。

单物体数据、完整物体网格、Wuji2手资产和正式场景数据都在本工程内；路径仍统一由`config/project.json`管理。旧40场景数据不再是默认训练依赖。

## 环境分工

- `generate_scenes_and_views.py`必须在`isaaclab22_sim50`/Isaac Sim 5.0环境中运行；
- 其他标签阶段使用`graspnet2.0`；
- 两套100场景训练标签和25600个相机输入已经完成；不要再次运行完整相机生成脚本。

## 状态检查

```bash
/home/lin/miniconda3/envs/graspnet2.0/bin/python \
  02_training_dataset/status_active_dataset.py
```

只有输出中的`允许训练: 是`才说明场景与全部视角完成；即使相机数据完成，仍需继续生成场景抓取、碰撞过滤、graspness和单视角标签，最后执行`check_wuji2_dataset.py`。

## 标签阶段输出

所有标签中间结果写在数据根目录的`grasp_label_stages/`下：

```text
01_transformed_object_grasps（保留全部单物体有效位姿）
→ 02_scene_table_collision_filtered（保留全部位姿，追加paper_keep_mask）
→ 02b增强掌心路径过滤（保留全部位姿，追加wuji2_safe_keep_mask）
→ 03_reference_points_and_surface_graspness
→ 04_single_view_training_labels
```

每一阶段保存`stage_manifest.json`，下一阶段必须引用上一阶段清单，保证标签来源可追溯。
