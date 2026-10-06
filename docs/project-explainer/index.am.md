---
title: 从人体动作到机器人随动
subtitle: WT901 × RGB-D × X1 ｜ 用真实图像读懂姿态、标定与融合
template: doc
theme: shadcn
mode: light
lang: zh
style: 80
source: 本地录制、现有代码、离线重建
date: 2026-10-05
---

IMU 提供连续姿态，相机提供可观测的方向参考。
两路标定连接传感器、人体模型与机器人模型。
本项目当前输出到 RViz，不控制实体机器人。

```html
<nav aria-label="阅读路线"><a href="#panel-A">先看效果</a> · <a href="#panel-B">为什么融合</a> · <a href="#panel-C">视觉测量</a> · <a href="#panel-D">姿态传播</a> · <a href="#panel-E">两路标定</a> · <a href="#panel-F">航向纠偏</a> · <a href="#panel-G">机器人映射</a> · <a href="#panel-H">证据与边界</a></nav>
<p><small>点击图片可放大。展开“数学解释”可查看公式。真实观测、计算重建与原理示意分别标注。</small></p>
```

## A 先看结果：同一个动作，三种表示

人抬起双臂，传感器记录各骨段姿态。
人体模型显示这些姿态，机器人模型模仿动作。
这不是把图像中的关节直接贴到机器人上。

```html
<div role="group" aria-label="动作快照选择"><button type="button" data-pose="neutral" aria-pressed="false">中立位 · 6 s</button> <button type="button" data-pose="forward" aria-pressed="true">前平举 · 15 s</button> <button type="button" data-pose="lateral" aria-pressed="false">侧平举 · 31 s</button></div>
<div class="photo-grid">
<figure><img id="demo-rgb" src="assets/forward-rgb.jpg" alt="录制中双臂前平举的真实彩色图"><figcaption>① 真实 RGB 观测<br>记录时间与模型快照按主机时间匹配</figcaption></figure>
<figure><img id="demo-stl" src="assets/forward-stl.png" alt="同一记录的前平举人体STL离线重建"><figcaption>② 人体 STL · 计算重建<br>同期陀螺＋重力姿态，同次穿戴标定</figcaption></figure>
<figure><img id="demo-x1" src="assets/forward-x1.png" alt="由同期人体姿态解算的X1机器人模型"><figcaption>③ X1 URDF · 计算重建<br>当前模仿模式 IK，并非历史 RViz 截屏</figcaption></figure>
</div>
<p><small>来源：20261004-135948-f649c2。该次穿戴标定四部位通过；相机标定仅右臂通过。图片用于讲解数据链，不将该记录包装成全通过案例。本次重建没有手背传感器，手部随前臂继承。</small></p>
```

```flow TB
真实动作 -> IMU姿态: 高频传播
IMU姿态 -> 人体STL: 穿戴映射
人体STL -> X1模型: 几何任务与IK
RGB与深度 -> IMU姿态: 可信时低频纠偏
```

[打开可探索架构图](assets/architecture.html)可查看代码节点与数据接口。
架构图保留上一版已验证的源码快照，不代表未提交改动。

## B 为什么不能只用一种传感器？

IMU 看不到人体，却能持续测量运动。
相机能看到人体，但会受遮挡和检测失败影响。
两者提供互补信息，不是简单平均两个姿态。

| 信息源 | 能提供什么 | 主要限制 | 在系统中的角色 |
| --- | --- | --- | --- |
| 陀螺仪 | 本体系角速度 | 零偏积分形成长期漂移 | 高频传播姿态 |
| 加速度计 | 静止时的重力方向 | 动态加速度污染；航向不可观测 | 修正倾斜，不修正绝对航向 |
| RGB-D | 肩肘腕的三维表面观测 | 遮挡、检测误差、深度噪声 | 低频补偿可观测的航向偏差 |

```html
<figure><img src="assets/yaw-observability.png" alt="两根骨段方向具有相同重力倾斜却有不同水平朝向的数学示意"><figcaption>原理示意，不是设备实测：绕重力轴转动后，重力观测不变。</figcaption></figure>
<details><summary>数学解释：为什么重力无法识别航向？</summary>
<p>令 R 将设备本体系 S 中的向量映射到固定软件参考系 W。k_W 是 W 中的单位向上方向。加计预测方向为 Rᵀk_W。</p>
<math display="block"><mrow><msup><mrow><mo>(</mo><mi>H</mi><mi>R</mi><mo>)</mo></mrow><mi>T</mi></msup><msub><mi>k</mi><mi>W</mi></msub><mo>=</mo><msup><mi>R</mi><mi>T</mi></msup><msup><mi>H</mi><mi>T</mi></msup><msub><mi>k</mi><mi>W</mi></msub><mo>=</mo><msup><mi>R</mi><mi>T</mi></msup><msub><mi>k</mi><mi>W</mi></msub></mrow></math>
<p>对任意绕 k_W 的旋转 H，上式成立。相同的重力读数对应无穷多个航向。SO(3) 投影只修复数值结构，不能消除这种物理不可观测性。</p>
</details>
```

## C 相机究竟测量了什么？

二维视觉模型只预测关键点像素，不预测深度。
程序读取该像素附近的真实配准深度。
再用相机内参反投影，得到三维表面点。

```html
<div class="photo-grid">
<figure><img src="assets/lateral-yolo-l.jpg" alt="本次重新运行YOLO26-l得到的双臂侧平举关键点"><figcaption>① 彩色图 → 2D 关键点<br>本次离线推理：YOLO26-l，输入 960<br>LS／LE／LW：人体左肩／肘／腕</figcaption></figure>
<figure><img src="assets/registered-depth.png" alt="同帧真实配准深度和关键点像素位置"><figcaption>② 对应像素 → 真实深度<br>颜色表示深度，单位毫米<br>此图圆点使用原记录关键点，不是上图新推理结果</figcaption></figure>
<figure><img src="assets/surface-skeleton.png" alt="原记录真实深度反投影得到的三维表面关节点"><figcaption>③ 反投影 → 3D 骨段方向<br>相机系：X 右、Y 下、Z 前<br>计算重建，不是解剖学骨关节中心</figcaption></figure>
</div>
<details><summary>数学解释：从像素与深度构造方向</summary>
<p>在去畸变后的针孔坐标中，像素 (u,v)、深度 z 与内参 K 给出相机坐标 p_C。实现包含畸变处理和配准校验，不能仅假定两幅图同分辨率。</p>
<math display="block"><mrow><msub><mi>p</mi><mi>C</mi></msub><mo>=</mo><mi>z</mi><msup><mi>K</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><msup><mrow><mo>[</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo>,</mo><mn>1</mn><mo>]</mo></mrow><mi>T</mi></msup><mo>,</mo><mspace width="1em"/><msub><mi>d</mi><mi>C</mi></msub><mo>=</mo><mfrac><mrow><msub><mi>p</mi><mi>distal</mi></msub><mo>−</mo><msub><mi>p</mi><mi>proximal</mi></msub></mrow><mrow><mo>‖</mo><msub><mi>p</mi><mi>distal</mi></msub><mo>−</mo><msub><mi>p</mi><mi>proximal</mi></msub><mo>‖</mo></mrow></mfrac></mrow></math>
<p>上臂用肩→肘，前臂用肘→腕。端点置信度、深度支持、骨长与同期性不合格时，系统拒绝纠偏。</p>
</details>
```

深度通常落在皮肤或衣服表面。
因此，这条“三维骨段”也是带噪观测。
二维检测能看到完整手臂，不等于三维深度可靠。

## D 高频端：不用厂商 RPY，也能维护姿态

软件用陀螺角速度逐步更新旋转矩阵。
加计反馈修正倾斜，静止学习估计陀螺零偏。
软件参考系 W 固定，不随每次纠倾斜重新定义。

```flow TB
角速度与设备时间 -> 陀螺传播: 扣除零偏
陀螺传播 -> 预测姿态Rpred
预测姿态Rpred -> 重力方向差: 与加计方向比较
加计 -> 重力方向差: 可靠度加权
重力方向差 -> 纠倾斜姿态R: 小步反馈
纠倾斜姿态R -> 下一次陀螺传播
```

```html
<details><summary>数学解释：传播、重力反馈与数值检查</summary>
<p>Exp(v) ≡ exp([v]×) 是旋转向量的指数映射。ω_S 在设备本体系下表达，因此增量右乘。Δt 优先采用设备时间。</p>
<math display="block"><mrow><msub><mi>R</mi><mi>pred</mi></msub><mo>=</mo><msub><mi>R</mi><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub><mi mathvariant="normal">Exp</mi><mo>(</mo><mo>(</mo><msub><mi>ω</mi><mi>S</mi></msub><mo>−</mo><msub><mi>b</mi><mi>gyro</mi></msub><mo>)</mo><mi>Δt</mi><mo>)</mo></mrow></math>
<math display="block"><mrow><msub><mi>e</mi><mi>S</mi></msub><mo>=</mo><msub><mi>a</mi><mi>S</mi></msub><mo>×</mo><mo>(</mo><msubsup><mi>R</mi><mi>pred</mi><mi>T</mi></msubsup><msub><mi>k</mi><mi>W</mi></msub><mo>)</mo><mo>,</mo><mspace width="1em"/><mi>R</mi><mo>=</mo><msub><mi>R</mi><mi>pred</mi></msub><mi mathvariant="normal">Exp</mi><mo>(</mo><mi>γ</mi><mi>w</mi><mi>Δt</mi><msub><mi>e</mi><mi>S</mi></msub><mo>)</mo></mrow></math>
<p>a_S 是单位化的加计向上方向；γ 为反馈增益，w 为可信度。e_S 不是单位旋转轴，其长度是方向夹角的正弦。大幅动态加速度下，反馈权重降低。</p>
<p>程序检查正交性和行列式。超阈值时用 SVD 投影到 SO(3)。这不代表姿态接近真值，只代表矩阵仍是合法旋转。</p>
<p><a href="../GYRO_GRAVITY_TILT_MATH.md">完整推导：陀螺＋重力纠倾斜</a></p>
</details>
```

设备本体系 S 随设备转动，W 是积分参考系。
选择不同算法模式时，不能混用两路的中立矩阵。
厂商 RPY 模式保留，便于同设备对比。

## E 一次动作，为什么要做两路标定？

穿戴标定回答：设备运动怎样驱动人体模型？
相机标定回答：设备预测方向怎样与视觉比较？
两路共享录制，但各自拟合、各自验收。

```html
<div class="photo-grid">
<figure><img src="assets/neutral-keypoints.jpg" alt="中立位真实记录和关键点"><figcaption>中立位：采集配对基准<br>不是假定所有设备姿态都为单位矩阵</figcaption></figure>
<figure><img src="assets/forward-keypoints.jpg" alt="正对相机前平举时肩肘腕重叠的真实记录"><figcaption>前举：可见性是一个问题<br>相机接近正面时，肩肘腕可能重叠</figcaption></figure>
<figure><img src="assets/lateral-keypoints.jpg" alt="侧平举时肩肘腕分开的真实记录"><figcaption>侧举：补充不同方向<br>相机分支加入斜前举以减少重叠</figcaption></figure>
</div>
```

| 分支 | 固定量 | 这些量表示什么 | 应用后的作用 |
| --- | --- | --- | --- |
| 穿戴标定 | B、配对中立姿态 N | 中立设备轴到人体模型轴的运动换基 | 把相对运动映射到模型 |
| 相机标定 | Q、b_S、k_C | 每设备 W→C 对齐；设备中固定骨轴；相机向上方向 | 在相机系预测骨段并比较航向 |

相机 Q 是每台设备的软件参考系对齐，不是全局唯一外参。
模型系与设备积分系的重力对齐，不等于航向一致。

```html
<details><summary>数学解释：两个优化问题分别在估计什么？</summary>
<p>普通穿戴标定利用中立下垂和前／侧举先验。先约束骨轴，再估计绕模型竖直轴的角 α；手动映射提供弱先验，不是真值。视觉辅助穿戴标定是另一个保留分支。</p>
<math display="block"><mrow><msup><mi>α</mi><mo>*</mo></msup><mo>=</mo><munder><mi mathvariant="normal">arg min</mi><mi>α</mi></munder><mo>{</mo><munder><mo>∑</mo><mi>j</mi></munder><msup><mrow><mo>‖</mo><msub><mi>R</mi><mi>z</mi></msub><mo>(</mo><mi>α</mi><mo>)</mo><msub><mi>a</mi><mi>j</mi></msub><mo>−</mo><msub><mi>d</mi><mrow><mi>M</mi><mo>,</mo><mi>j</mi></mrow></msub><mo>‖</mo></mrow><mn>2</mn></msup><mo>+</mo><mi>λ</mi><msup><mrow><mo>‖</mo><mi>h</mi><mo>(</mo><mi>α</mi><mo>)</mo><mo>−</mo><mi>h</mi><mo>(</mo><msub><mi>α</mi><mi>prior</mi></msub><mo>)</mo><mo>‖</mo></mrow><mn>2</mn></msup><mo>}</mo></mrow></math>
<p>这里 a_j 是预对齐后的动作骨段方向，d_M,j 是模型要求方向，h(α)=[cos α,sin α]ᵀ。拟合产生连续 B，而非仅选择一个轴置换。</p>
<p>相机拟合的预测方向是 Q R(t) b_S。方向残差采用鲁棒最小二乘；Q 被约束为满足 Qk_W=k_C 的单参数航向族，b_S 用两个角参数化，共三个参数。</p>
<math display="block"><mrow><munder><mi mathvariant="normal">min</mi><mrow><mi>β</mi><mo>,</mo><mi>θ</mi><mo>,</mo><mi>φ</mi></mrow></munder><munder><mo>∑</mo><mi>t</mi></munder><munder><mo>∑</mo><mi>i</mi></munder><mi>ρ</mi><mo>(</mo><msup><msub><mrow><mo>[</mo><mi>Q</mi><mo>(</mo><mi>β</mi><mo>)</mo><mi>R</mi><mo>(</mo><mi>t</mi><mo>)</mo><msub><mi>b</mi><mi>S</mi></msub><mo>(</mo><mi>θ</mi><mo>,</mo><mi>φ</mi><mo>)</mo><mo>−</mo><msub><mi>d</mi><mi>C</mi></msub><mo>(</mo><mi>t</mi><mo>)</mo><mo>]</mo></mrow><mi>i</mi></msub><mn>2</mn></msup><mo>)</mo></mrow></math>
<p>上式意指对三维残差的每个分量施加 soft-L1 损失，不是对完整姿态拟合。标定阶段仍假设姿态漂移足够小；固定偏差可能被 Q 吸收，时变漂移不能被固定 Q 完全吸收。</p>
<p><a href="../CONTINUOUS_ARM_CALIBRATION.md">穿戴标定完整推导</a> · <a href="../../OptUSBProj-ubuntu/docs/COMBINED_ARM_CAMERA_CALIBRATION.md">联合标定实现说明</a></p>
</details>
```

## F 长期运行：相机只修正可信的航向差

程序先预测骨段在相机系中的方向。
再比较预测与视觉方向的水平投影。
低通与限速控制补偿，避免一次坏观测造成跳变。

```flow TB
IMU姿态R与固定骨轴 -> 预测dC: Q乘Rb
视觉三维骨段 -> 质量检查
质量检查 -> 水平角差: 同期且可观测
预测dC -> 水平角差
水平角差 -> 平滑累计角δ: 低通与限速
平滑累计角δ -> 相机旋转H
相机旋转H -> 固定系补偿: Q转置乘H乘Q
固定系补偿 -> 融合姿态Rf: 左乘原姿态R
```

```html
<details><summary>数学解释：共轭换基与左乘补偿</summary>
<math display="block"><mrow><msub><mi>d</mi><mi>C</mi></msub><mo>=</mo><mi>Q</mi><mi>R</mi><msub><mi>b</mi><mi>S</mi></msub><mo>,</mo><mspace width="1em"/><msub><mi>H</mi><mi>C</mi></msub><mo>=</mo><mi mathvariant="normal">Exp</mi><mo>(</mo><msub><mi>k</mi><mi>C</mi></msub><mi>δ</mi><mo>)</mo></mrow></math>
<math display="block"><mrow><msub><mi>R</mi><mi>f</mi></msub><mo>=</mo><mo>(</mo><msup><mi>Q</mi><mi>T</mi></msup><msub><mi>H</mi><mi>C</mi></msub><mi>Q</mi><mo>)</mo><mi>R</mi></mrow></math>
<p>QᵀH_CQ 把同一个空间旋转换回 W 系。误差是在固定参考系下作用于姿态，所以左乘 R；不是把补偿当作设备本体系中的运动右乘。</p>
<p>水平单位方向 p、v 给出剩余角差 ε=atan2(k_C·(p×v),p·v)。每次先按当前累计补偿预测，再计算 ε。</p>
<math display="block"><mrow><msub><mi>Δδ</mi><mi>t</mi></msub><mo>=</mo><mi mathvariant="normal">clip</mi><mo>(</mo><mo>(</mo><mn>1</mn><mo>−</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>Δt</mi><mo>/</mo><mi>τ</mi></mrow></msup><mo>)</mo><msub><mi>ε</mi><mi>t</mi></msub><mo>,</mo><mo>−</mo><mi>v</mi><mi>Δt</mi><mo>,</mo><mi>v</mi><mi>Δt</mi><mo>)</mo></mrow></math>
<p>τ 为低通时间常数，v 为最大补偿角速度。实现还有死区、最大创新与累计角限制／实验性不限模式。无可靠观测时冻结补偿；陀螺传播继续。</p>
</details>
```

```callout warn 什么时候应该拒绝纠偏？
骨段接近竖直，水平朝向不可观测。
关键点、深度、骨长或时间匹配不可靠。
设备重力参考与标定参考不一致。
这些时候保持已有补偿，不猜测新的角度。
```

两端点不能观测骨段绕自身长轴的纯扭转。
本版只对双臂上臂／前臂纠偏，不等于手掌纠偏。

## G 从人体姿态到机器人：不是直接拷贝关节角

人体模型先用骨架变换链计算肘腕位置。
程序按长度比例构造机器人的目标位置。
IK 再在机器人自由度与限位内拟合这些任务。

```html
<div class="photo-grid">
<figure><img src="assets/lateral-stl.png" alt="侧平举人体模型的离线计算渲染"><figcaption>人体模型<br>每部位的旋转，经骨架递推得到位置</figcaption></figure>
<figure><img src="assets/lateral-x1.png" alt="同一侧平举姿态解算出的X1机器人模型"><figcaption>机器人模型<br>真实 URDF 几何，当前 IK 离线解算</figcaption></figure>
<figure><img src="assets/rviz-historical.png" alt="用户2026年9月28日录屏中的真实RViz界面"><figcaption>真实历史 RViz 界面<br>来自 9 月 28 日录屏；并非左侧同次记录</figcaption></figure>
</div>
<details><summary>数学解释：运动换基、目标位置与优化任务</summary>
<p>N 是同一姿态流的归零／中立姿态。B 将中立设备系下的运动表示换到模型系。人体骨架随后递推各部位的 4×4 变换。</p>
<math display="block"><mrow><msub><mi>U</mi><mi>M</mi></msub><mo>=</mo><mi>B</mi><mo>(</mo><msup><mi>N</mi><mi>T</mi></msup><msub><mi>R</mi><mi>f</mi></msub><mo>)</mo><msup><mi>B</mi><mi>T</mi></msup><mo>,</mo><mspace width="1em"/><msub><mi>T</mi><mi>child</mi></msub><mo>=</mo><msub><mi>T</mi><mi>parent</mi></msub><msub><mi>T</mi><mi>local</mi></msub></mrow></math>
<p>设 u=p_E−p_S、f=p_W−p_E。人体和机器人使用一致的全局轴。上臂比例 s_u、前臂比例 s_f 由各自中立长度得到；统一比例模式令两者相等。</p>
<math display="block"><mrow><msubsup><mi>p</mi><mi>E</mi><mo>*</mo></msubsup><mo>=</mo><msubsup><mi>p</mi><mrow><mi>E</mi><mo>,</mo><mn>0</mn></mrow><mi>r</mi></msubsup><mo>+</mo><msub><mi>s</mi><mi>u</mi></msub><mo>(</mo><mi>u</mi><mo>−</mo><msub><mi>u</mi><mn>0</mn></msub><mo>)</mo></mrow></math>
<math display="block"><mrow><msubsup><mi>p</mi><mi>W</mi><mo>*</mo></msubsup><mo>=</mo><msubsup><mi>p</mi><mrow><mi>W</mi><mo>,</mo><mn>0</mn></mrow><mi>r</mi></msubsup><mo>+</mo><msub><mi>s</mi><mi>u</mi></msub><mo>(</mo><mi>u</mi><mo>−</mo><msub><mi>u</mi><mn>0</mn></msub><mo>)</mo><mo>+</mo><msub><mi>s</mi><mi>f</mi></msub><mo>(</mo><mi>f</mi><mo>−</mo><msub><mi>f</mi><mn>0</mn></msub><mo>)</mo></mrow></math>
<p>腕目标包含 u，因为上臂一转，肘和腕都移动。IK 联合最小化肘位置、腕位置、可选手腕相对朝向与连续性残差，而非让多个控制器分别抢占关节。</p>
<p>模仿模式优先保留人体动作形态；完整末端位姿模式优先拟合位置与朝向，可能引起更大的关节变化。当前有最小二乘／雅可比两个后端，也有实际 Δt 的速度限幅。</p>
<p><a href="../HUMAN_TO_X1_RETARGETING_MATH.md">重定向数学推导</a> · <a href="../URDF_POE_EQUIVALENCE.md">URDF 变换链与 PoE 的等价关系</a></p>
</details>
```

```callout warn 安全边界
这里的 X1 是机器人模型预览。
本页不打开 UDP 监听，不启动硬件控制。
实体机器人还需要碰撞、速度、力矩与急停验证。
```

## H 哪些是证据，哪些仍有局限？

最新全通过报告来自斜前举联合标定。
四个穿戴结果与四个相机结果均通过各自验收。
“通过”表示满足当前门槛，不表示已取得真值精度。

| 骨段 | 穿戴验收 | 相机验收 | 相机拟合方向中位数／P90 |
| --- | --- | --- | --- |
| 左上臂 | ok 通过 | ok 通过 | 7.04°／7.83° |
| 左前臂 | ok 通过 | ok 通过 | 2.79°／6.45° |
| 右上臂 | ok 通过 | ok 通过 | 5.69°／6.50° |
| 右前臂 | ok 通过 | ok 通过 | 7.47°／9.08° |

报告：`20261004-144730-de9b33-0ab2c0`。
这些数值是训练拟合残差，不是独立测试或 GT 误差。
该记录的原始图像已清理，所以不伪造它的动作图。

```html
<details><summary>展开真实短记录的同数据纠偏对比</summary>
<figure><img src="assets/direction-residuals.png" alt="同一短记录中IMU与RGBD纠偏后对视觉方向残差的实际曲线"><figcaption>真实计算结果：20261004-103131-25da07，约 70 s。橙色为纯 IMU，青色为纠偏后。曲线多数接近重合；这份短记录不能单独证明长期漂移改善。</figcaption></figure>
<p>纵轴是与带噪 RGB-D 方向的夹角，包括标定与验证片段，并非光学动捕真值。用户大于 5 分钟的稳定性观察是另一类证据，尚未在本页重建同条件长期量化基准。</p>
</details>
```

| 已有能力 | 需要保留的边界 |
| --- | --- |
| 双臂上臂／前臂长期视觉航向补偿 | 遮挡和近竖直时不能持续观测 |
| 一次引导动作，两路独立标定 | 标定期漂移、非标准动作可能污染拟合 |
| 同期姿态驱动 STL 与 X1 预览 | 机器人比例、自由度与人体不同 |
| 手掌自然显示参考与视觉原型 | 绑带遮挡显著影响手部检测，尚未形成可靠长期纠偏 |

下一步先提高掌面观测的可靠性，再探索骨关节中心估计。
避免用更复杂的模型掩盖观测不足。

```html
<details><summary>素材来源、复现与分享注意事项</summary>
<ul><li><a href="assets/provenance.json">素材溯源 JSON</a>：源记录、时间、权重身份与文件摘要。</li><li><a href="README.md">页面说明与复现命令</a>：仅离线，不修改运行中 GUI。</li><li><a href="assets/architecture.html">可探索架构图</a>：复用上一版已通过自动检查的 HTML。</li><li>图中包含真实人脸与室内环境。页面仅保存在本地，未上传网络；公开分享前请自行决定是否需要匿名化。</li></ul>
</details>
<!-- Gallery, MathML and image zoom have no equivalent built-in am component.
     This bounded HTML escape extends the renderer, not its page layout/theme. -->
<style>
.photo-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.am-panel-body figure{margin:12px 0;min-width:0}
.am-panel-body figure img{display:block;width:100%;height:auto;border-radius:8px;cursor:zoom-in}
.am-panel-body figcaption{font-size:.88em;line-height:1.6;margin-top:8px;color:var(--ink-2)}
.am-panel-body details{padding:12px;border:1px solid var(--line);border-radius:8px;margin:12px 0}
.am-panel-body summary{cursor:pointer;font-weight:600}
.am-panel-body math{max-width:100%;overflow-x:auto;padding:12px 0;font-size:1.1em}
.am-panel-body nav{line-height:2}
.am-panel-body button{cursor:pointer;padding:7px 12px;border-radius:6px}
.am-panel-body button[aria-pressed="true"]{outline:2px solid #067a85}
#image-zoom{max-width:92vw;max-height:94vh;border-radius:10px;padding:16px;background:var(--paper);color:var(--ink)}
#image-zoom::backdrop{background:rgba(0,0,0,.65)}
#zoom-image{display:block;max-width:85vw;max-height:78vh;object-fit:contain;margin:10px auto}
@media(max-width:720px){.photo-grid{grid-template-columns:1fr}.am-panel-body math{font-size:.94em}}
</style>
<dialog id="image-zoom" aria-label="放大图片"><button type="button" id="close-zoom">关闭 · Esc</button><img id="zoom-image" alt=""><p id="zoom-caption"></p></dialog>
<script>
document.querySelectorAll('.am-panel-body figure img').forEach(img=>{
 img.tabIndex=0;img.setAttribute('role','button');img.setAttribute('aria-label','放大：'+img.alt);
 const open=()=>{document.getElementById('zoom-image').src=img.src;document.getElementById('zoom-image').alt=img.alt;document.getElementById('zoom-caption').textContent=img.closest('figure').querySelector('figcaption').textContent;document.getElementById('image-zoom').showModal();};
 img.addEventListener('click',open);img.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();open();}});
});
document.getElementById('close-zoom').onclick=()=>document.getElementById('image-zoom').close();
document.querySelectorAll('[data-pose]').forEach(button=>button.onclick=()=>{
 const pose=button.dataset.pose;const labels={neutral:'中立位',forward:'前平举',lateral:'侧平举'};
 for(const type of ['rgb','stl','x1']){const image=document.getElementById('demo-'+type);image.src='assets/'+pose+'-'+type+(type==='rgb'?'.jpg':'.png');image.alt=labels[pose]+' · '+type+' · '+(type==='rgb'?'真实记录':'离线计算重建');image.setAttribute('aria-label','放大：'+image.alt);}
 document.querySelectorAll('[data-pose]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
});
</script>
```
