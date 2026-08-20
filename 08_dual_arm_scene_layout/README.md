# 08 双臂与桌面手动布置

> 当前闭环入口、task/route 组合和最新合同见 `isaaclab_control/MAINLINE.md`。
> 本文重点保留桌面、SourceZone、PlacementZone、相机和双臂安装的标定方法；
> 旧的单次物理结果只作为历史证据，不代表当前正式实验已完成。

本 README 的前半部分说明桌面、抓取区、放置区、相机和双臂机械臂的手工标定；
机械臂控制、完整抓放和稳定回放以 `isaaclab_control/README.md` 为准。

手工布局与单帧抓拍脚本在已经打开的唯一 Isaac Sim Script Editor 中运行，不需要
Isaac Lab。正式机械臂控制则由 Isaac Lab AppLauncher 单独启动，不能和手工会话并行。

## 1. 当前场景结构

运行创建脚本后，Isaac Sim中的Stage树为：

```text
/World
├── Layout
│   ├── TableAssembly                 整张桌子的可编辑父节点
│   │   ├── Table                     1.60 × 0.40 × 0.04 m
│   │   ├── SourceZone                蓝色，0.50 × 0.30 m
│   │   ├── PlacementZone             绿色，当前标定为0.80 × 0.30 m
│   │   └── TestScene0000             导入后出现；测试集首场景的6个物体
│   └── DualArmMount                  整台机械臂的可编辑父节点
│       └── DualArm                   左原夹爪＋右官方Wuji2的原生USD引用
├── Markers
│   ├── TableCenter
│   ├── SourceZoneCenter
│   ├── PlacementZoneCenter
│   ├── RobotRoot
│   └── DistanceLabels
├── Environment
└── PhysicsScene
```

蓝区和绿区只是半透明视觉标识，没有碰撞；真正承载物体的是灰色桌面。当前标定
JSON 中 Table 中心 `z=0.44 m`、厚度 `0.04 m`，因此物理桌面顶面为
`z=0.46 m`。运行时必须从 live stage/config 读取，不能把 SourceZone 显示 cube 的
顶面或 scale 当作物理支撑。

## 2. 模型位置与保护原则

Isaac Sim最终双臂资产来自：

```text
01_environment/vendor/wuji-description/dual_arm_right_wuji2/usd/dual_arm_right_wuji2.usd
```

它由同一供应商目录中的两个权威源派生：

```text
01_environment/vendor/wuji-description/
├── hand2/                   官方Wuji2 Hand 2，未修改
├── dual_arm/                双臂原始ZIP解压结果，未修改
└── dual_arm_right_wuji2/    派生装配：左夹爪保留、右侧换官方Wuji2 USD
```

创建脚本只引用已经验收的最终USD，绝不现场重转手模型，也不修改任何上游资产。

## 3. 第一次创建场景

1. 打开Isaac Sim 5.0。
2. 点击`File -> New`，保持一个空Stage。
3. 打开`Window -> Script Editor`。
4. 在Script Editor中点击`File -> Open`，打开：

   ```text
   /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/scripts/01_create_manual_layout.py
   ```

5. 点击左下角`Run (Ctrl+Enter)`一次。
6. 等待控制台出现：

   ```text
   [LAYOUT DRAFT CREATED]
   ```

脚本会保存草稿：

```text
08_dual_arm_scene_layout/scenes/manual_layout_draft.usda
```

运行期间不要点击Play。当前只是摆几何位置，时间线必须保持停止。

## 4. 你可以怎样手动调整

### 调整整张桌子

在Stage树中选择：

```text
/World/Layout/TableAssembly
```

然后在右侧Property面板的Transform中修改：

- `Translate X/Y/Z`：整张桌子连同两个区域一起平移；
- `Rotate X/Y/Z`：整张桌子连同两个区域一起旋转。

通常建议先保持桌面原点和朝向不动，把桌面作为世界参考。

### 调整整台机械臂

选择：

```text
/World/Layout/DualArmMount
```

修改它的Translate和Rotate。这样会整体移动机械臂，内部关节关系不变。

不要直接拖动`DualArm`内部的某一个link，否则会破坏机器人装配关系。

### 区域颜色

- 蓝色：物体初始摆放/抓取区；
- 绿色：抓起后放置区；
- 灰色：真实有碰撞的桌面。

只有确实需要改变两个区域在桌面上的划分时，才分别调整`SourceZone`或
`PlacementZone`；一般只移动它们的父节点`TableAssembly`。

## 5. 距离怎么查看

运行后会出现`DGN2 Manual Scene Layout`小窗口，实时显示：

- 机械臂根到桌面中心；
- 机械臂根到抓取区中心；
- 抓取区中心到放置区中心。

视口里也有对应球形标记和连线。数值单位为米，并写入：

```text
/World/Markers/DistanceLabels
```

这些距离是坐标原点间距离，不等于机械臂外壳到桌面的最短表面距离。最终还要
肉眼检查底座无穿透，并在后续运动规划阶段做完整碰撞与可达性检查。

## 6. 手动调整机械臂关节

如果Isaac Sim顶部菜单没有`Window -> Physics`，在Script Editor中打开并运行：

```text
/home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/scripts/03_open_physics_inspector.py
```

它只启用本机已有的`omni.physx.supportui`扩展并打开Physics Inspector，不改场景。
窗口可能停靠在Stage面板下方。打开后点击`Select Articulation`，选择
`/World/Layout/DualArmMount/DualArm/root_joint`这个唯一articulation root，并在Options中设置：

```text
Sliders / Drags -> Joint States Position
Fix Articulation Base -> On
Enable Gravity -> Off
Show Joints Hierarchy -> On
```

拖动关节滑块后点击绿色`Commit Changes`写入当前USD。不要直接旋转某个link的
Transform来伪造关节运动。

### 精确输入关节角度

Physics Inspector的默认界面主要依靠鼠标拖动滑块。需要输入精确角度时，在已经
选中上述articulation以后，再从Script Editor运行：

```text
/home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/scripts/04_numeric_joint_angle_editor.py
```

弹出的`DGN2 Numeric Joint Angle Editor`包含整机35个关节：

- 数值单位统一为度；
- 每个关节同时提供滑块和数字框，两者共享同一个值；
- 拖动滑块或输入数字时，会原子性提交完整35维姿态；
- `Apply`表示以这一行为刷新触发器，但仍会锁定全部35个关节；
- `Re-apply/lock all 35`重新提交并锁定当前完整姿态；
- `Reload current`重新读取当前场景关节状态；
- 每行右侧显示官方关节角度上下限，越界值会被拒绝。

每次编辑都会先完整校验35个输入，再一次性写入全部35个Joint State、把35个
关节速度清零并同步35个Drive Target，最后只用被编辑的关节触发一次Inspector
构型刷新。因此其余关节不会被旧驱动目标或残余速度带动，也不会依次触发35次
物理解算。它不修改官方USD中的刚度、阻尼、最大力矩或关节限位。调整期间时间线
必须保持停止；确认姿态后再运行第7节的导出脚本。

数字编辑器使用自己的滑块和数字框，但需要保持Physics Inspector窗口打开，并让
它停留在`Joint States Position + QuasiStatic`模式。编辑器通过Inspector监听的
`ChangeProperty`命令触发单步构型刷新；如果关闭Inspector，只会看到数字变化，
机器人外观不会随关节更新。编辑器会在窗口创建前一次性为35个关节准备JointState
结构，若因此出现一次`Structural changes`，点击一次`Re-Enable authoring`即可；
之后调整过程只改数值，不再逐个关节改变USD结构。

为了进一步消除重力和惯性晃动，Physics Inspector的Options必须设为：

```text
Sliders / Drags       = Joint States Position
Enable Gravity        = Off
Use QuasiStatic mode  = On
Fix Articulation Base = On
```

## 7. 调整完成后导出

保持当前Stage打开，在Script Editor中打开并运行：

```text
/home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/scripts/02_export_calibrated_layout.py
```

成功后得到：

```text
08_dual_arm_scene_layout/scenes/manual_layout_calibrated.usda
08_dual_arm_scene_layout/config/manual_layout_calibrated.json
```

USD用于以后重新打开场景；JSON明确保存桌面、两个区域和机械臂根的世界变换。

## 8. 下一阶段

桌面与机械臂布局导出后，先创建功能型虚拟深度相机和视锥。在保持标定场景
打开、时间线停止的情况下，从Script Editor运行：

```text
/home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/scripts/05_create_virtual_depth_camera_frustum.py
```

脚本使用顶部`arm_base_link_d435i_2`外壳的世界位置作为虚拟光心，自动把光轴
指向蓝色`SourceZone`中心，根据蓝区四个角反算最小视场角，并添加15%安全边界：

- 绿色线：相机视锥；
- 蓝色线：蓝区真实边界；
- 黄色线：相机光轴；
- 白色球：虚拟相机光心；
- 黄色球：蓝区中心。

脚本会自动关闭视口中的巨大Camera辅助图标，但不会隐藏或删除Camera Prim，也不
影响RGB-D拍摄。若当前会话已经运行过旧版`05`，可以单独运行：

```text
08_dual_arm_scene_layout/scripts/05b_hide_camera_viewport_icon.py
```

窗口显示`Coverage: PASS`才表示四个蓝区角点全部在视野和裁剪范围内。当前相机
是用于打通接口的虚拟针孔相机，不代表真实D435i已经标定。它固定输出1280×720，
并把OpenCV坐标约定、内参K和`T_world_camera`预览记录写入：

```text
08_dual_arm_scene_layout/outputs/virtual_depth_camera_preview.json
```

视锥确认后再继续：

### 8.1 查看真实相机画面

先运行：

```text
08_dual_arm_scene_layout/scripts/06_preview_virtual_depth_camera.py
```

视口会切到`/World/Sensors/TopD435iVirtual/Camera`第一视角，显示蓝色抓取区和绿色
放置区，但隐藏绿色视锥及距离标记。确认两个区域完整可见，而且两条机械臂没有
遮挡。要退出时，从视口相机菜单切回`Perspective`即可。蓝绿区域只用于人工检查；
运行`07`正式抓拍时会自动隐藏，因而不会污染RGB、深度图或点云。

### 8.2 抓拍一帧RGB-D

确认画面后运行：

```text
08_dual_arm_scene_layout/scripts/07_capture_single_rgbd.py
```

控制台出现`[SINGLE RGBD CAPTURE COMPLETE]`后，输出位于：

```text
08_dual_arm_scene_layout/captures/latest/
├── rgb.png
├── depth_m.npy
├── depth_preview.png
├── intrinsics.npy
├── T_world_camera.npy
├── capture_manifest.json
└── scene_snapshot.json
```

`rgb.png`、`depth_m.npy`、`intrinsics.npy`和`T_world_camera.npy`已经严格匹配现有
GroundingDINO＋SAM接口。相机采用OpenCV坐标：`+x`图像右、`+y`图像下、`+z`
相机前。脚本异步抓拍，不推进物理时间，并在完成后主动释放Annotator和Render
Product，避免后台积累采集任务。

### 8.3 向蓝区导入测试集第一个场景

保持当前标定Stage打开、时间线停止，运行：

```text
08_dual_arm_scene_layout/scripts/06b_import_test_scene0000_into_source_zone.py
```

它严格读取测试集`wuji2_test60_10upright_10view_v1`的`scene_0000`，包含：

```text
开罐器、记事本、烟灰缸、狗、锤子、笔
```

该场景没有香蕉，因此脚本不会擅自用香蕉替换真实测试物体。测试场景原桌面和
蓝区均为`0.50 × 0.30 m`，故只做如下刚体映射，不缩放、不重新随机摆放：

```text
T_currentWorld_object = T_currentWorld_SourceZone × T_testTable_object
```

导入节点位于`/World/Layout/TableAssembly/TestScene0000`。它是
`TableAssembly`的子节点，所以桌面、蓝区、绿区和6个物体仍是一个整体；移动
`TableAssembly`会一起移动。当前导入的是固定视觉网格，专用于保持测试集最终
稳定姿态并拍RGB-D，不会因误点Play而重新掉落。审计记录写入：

```text
08_dual_arm_scene_layout/outputs/test_scene0000_import.json
```

导入完成后先重新运行`06_preview_virtual_depth_camera.py`检查画面，再运行
`07_capture_single_rgbd.py`覆盖`captures/latest`。新抓拍清单会额外记录6个物体的
类别、分割编号和世界位姿。之后才接GroundingDINO＋SAM并保留逐像素mask，生成
完整场景40000点与目标成员标记，再进入DexGraspNet2、LEAP到Wuji2迁移和双臂执行。

### 8.4 当前烟灰缸单帧抓取案例

当前`captures/latest`已经完成第一条真实接口链：

```text
TopD435iVirtual单帧RGB-D
→ GroundingDINO("ashtray")
→ SAM逐像素掩码
→ 蓝区完整场景40000点（不是孤立目标点云）
→ 未改权重的官方DexGraspNet2 LEAP网络
→ 1024条候选
→ 烟灰缸种子点成员过滤
→ 官方PREGRASP场景/桌面碰撞过滤
→ 官方wuji-retargeting生成Wuji2 q20
→ 四指指尖Kabsch求Wuji2手根6D
→ 官方SQUEEZE迁移
→ 3P+3R Isaac Sim验证任务
```

可复现的离线脚本依次为：

```text
08_build_target_network_input.py
09_predict_official_leap_target.py
10_filter_target_pregrasp_collision.py
11_prepare_ashtray_retarget_case.py
```

本次审计数值：

- GroundingDINO选中烟灰缸分数：`0.62699`；
- SAM预测IoU：`0.96887`；
- 有效烟灰缸深度像素：`6922`；
- 40000点输入中的目标样本：`7632`；
- 官方候选：`1024`，烟灰缸候选：`258`；
- 通过官方PREGRASP碰撞过滤：`224`；
- 最终候选：`821`，官方分数：`21.71713`；
- 场景/桌面碰撞值：`-50.42/-88.96 mm`，负数表示有净空。

可执行案例固定保存在：

```text
06_leap_to_wuji2_final_pipeline/01_cases/
live_scene0000_ashtray_official_best/
```

先在终端查看四手姿态（LEAP GRASP/SQUEEZE、Wuji2 GRASP/SQUEEZE）：

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2
01_environment/conda/wuji_retargeting/bin/python -c \
'import trimesh; trimesh.load("06_leap_to_wuji2_final_pipeline/01_cases/live_scene0000_ashtray_official_best/05_visualization/four_hand_final.glb").show()'
```

随后在Isaac Sim 5.0的Script Editor中依次打开并点击`Run`：

```text
06_leap_to_wuji2_final_pipeline/01_cases/
live_scene0000_ashtray_official_best/06_isaacsim/01_import.py

06_leap_to_wuji2_final_pipeline/01_cases/
live_scene0000_ashtray_official_best/06_isaacsim/02_execute.py
```

`01_import.py`会创建独立的手部物理验证Stage，不能与当前双臂布局Stage混用；如需
保留当前双臂布局，先保存或以后重新打开`manual_layout_calibrated.usda`。出现
`[01 IMPORT COMPLETE]`后再运行`02_execute.py`。该阶段只验证“迁移后的Wuji2手能否
抓起烟灰缸”，还没有求右机械臂到PREGRASP的IK轨迹；二者必须分开验收，避免把
机械臂轨迹问题误判为网络或手姿态问题。

## 9. 本阶段是否需要Isaac Lab或命令行

- 不需要Isaac Lab；
- 不需要ROS 2；
- 不需要命令行启动第二个Isaac Sim；
- 只需要Isaac Sim GUI的Script Editor运行上述脚本。

## 10. Isaac Lab机械臂平滑控制

场景、相机与手部抓取单独验收后，右臂7关节IK从这里开始：

```text
08_dual_arm_scene_layout/isaaclab_control/
```

第一项测试只让`arm_r_link_tf`沿世界`+Z`平滑移动20 mm并返回，不执行抓取。
它用于先验收35关节装配、自带USD驱动、五次轨迹、SLERP、DLS IK及关节限速，详细运行
命令见[isaaclab_control/README.md](isaaclab_control/README.md)。
