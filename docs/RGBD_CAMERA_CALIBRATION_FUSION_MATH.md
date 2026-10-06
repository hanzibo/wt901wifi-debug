# RGB-D 相机标定与 IMU 长期航向纠偏：数学推导

本文对应当前左右上臂、前臂的相机标定与实时反馈代码，解释固定参数拟合、动态航向纠偏以及它们和穿戴标定的关系。这里“相机标定”指 **IMU—视觉骨段方向对齐**，不是重新标定相机内参、RGB—深度外参或完整机器人手眼变换。

相关文档：[陀螺＋重力纠倾斜](GYRO_GRAVITY_TILT_MATH.md)、[穿戴标定](CONTINUOUS_ARM_CALIBRATION.md)、[实时融合操作说明](../OptUSBProj-ubuntu/docs/RGBD_LIVE_FUSION.md)、[双臂一体标定](../OptUSBProj-ubuntu/docs/COMBINED_ARM_CAMERA_CALIBRATION.md)。

## 1. 符号与参考系

使用右手系、列向量，旋转矩阵属于 $SO(3)$。定义 $\operatorname{Exp}(\boldsymbol\phi)=\exp([\boldsymbol\phi]_\times)$，旋转向量以弧度计。

| 符号 | 含义 |
| --- | --- |
| $S(t)$ | 设备随动本体系 |
| $W$ | 该设备软件积分器的固定初始参考系，不保证 $Z$ 竖直 |
| $C$ | 相机参考系，通常 $X$ 向右、$Y$ 向下、$Z$ 向前 |
| $M$ | GUI 模型中立参考系 |
| $R(t)={}^W R_{S(t)}$ | 未加视觉补偿的陀螺＋加计姿态，已包含重力纠倾斜 |
| $\mathbf k_W,\mathbf k_C$ | 同一竖直向上方向在 $W,C$ 中的单位坐标 |
| $Q={}^C R_W$ | 固定的 IMU 软件参考系到相机系旋转 |
| $\mathbf b_S$ | 骨段近端指向远端的单位方向，在设备本体系中固定 |
| $\mathbf d_C$ | 视觉观测的相机系骨段方向 |
| $\delta(t)$ | 动态累计航向补偿角，不是固定标定参数 |

每台设备有自己的 $W,Q,\mathbf b_S,\delta$；不能默认四个部位共享一个 $Q$。相机向上参考 $\mathbf k_C$ 在一轮中共同使用。

## 2. RGB-D 给出的观测是什么

二维模型提供彩色图上的肩、肘、腕像素。代码先按彩色内参及畸变解算射线，再将射线投影到配准深度像素系，取局部有效深度并恢复三维点。

理想的去畸变射线 $(x_n,y_n,1)$ 与光轴深度 $z$ 给出

$$
\mathbf p_C=z(x_n,y_n,1)^\mathsf T.
$$

$z$ 来自真实深度，而非二维模型的预测深度；毫米统一转换为米。相机参数在这里是输入，算法不重新优化它们。

上臂、前臂方向分别为

$$
\mathbf d_{U,C}=
\frac{\mathbf p_{\rm elbow,C}-\mathbf p_{\rm shoulder,C}}
{\|\mathbf p_{\rm elbow,C}-\mathbf p_{\rm shoulder,C}\|},
$$

$$
\mathbf d_{F,C}=
\frac{\mathbf p_{\rm wrist,C}-\mathbf p_{\rm elbow,C}}
{\|\mathbf p_{\rm wrist,C}-\mathbf p_{\rm elbow,C}\|}.
$$

这是骨段**指向**，不是完整三轴姿态；表面深度也不是解剖学骨关节中心 GT。

## 3. 向上参考的建立与先验

### 3.1 IMU 软件参考系中的向上方向

静止时归一化加计 $\mathbf a_S$ 指向上方，因此

$$
\mathbf k_W=R(t_0)\mathbf a_S(t_0).
$$

当前标定读取实际运行的 `gyro_tilt` 姿态与重力参考，不用重新积分的姿态冒充当时状态。引导中立窗口验证并锁定重力参考：已有参考保持不变；缺失参考可从可信静止观测建立，不重置 $R$、时钟或零偏。后续检查参考系 epoch 与锁定参考的一致性。

### 3.2 相机系中的向上方向

对站直的中立窗口，使用肩中心减髋中心：

$$
\mathbf k_{C,j}=
\operatorname{unit}\left(
\frac{\mathbf p_{\rm leftShoulder,C}+\mathbf p_{\rm rightShoulder,C}}2
-\frac{\mathbf p_{\rm leftHip,C}+\mathbf p_{\rm rightHip,C}}2
\right).
$$

归一化多帧均值得到

$$
\mathbf k_C=\operatorname{unit}\left(\sum_j\mathbf k_{C,j}\right).
$$

这是**直立躯干方向作为重力方向的近似**，不是实测地面法向。代码要求至少五帧完整肩髋观测，中立方向波动 P90 不超过 $8^\circ$。身体前倾可能带来系统误差，即使其窗口内十分稳定。

相机不必正对人体；也不必让相机 $Y$ 与重力严格对齐。但相机与绑带须固定，关键点可见，且直立先验可靠。

## 4. 固定参数 $Q,\mathbf b_S$ 从哪里来

设备本体系中的固定骨轴经姿态和参考系转换，得到预测方向：

$$
\boxed{\hat{\mathbf d}_C(t)=Q R(t)\mathbf b_S.}
$$

标定的任务是让这一预测解释多帧视觉方向。骨轴有两个自由度；在重力对齐约束下 $Q$ 只剩一个航向自由度，共三个参数。

先构造重力对齐旋转 $Q_g$：

$$
Q_g\mathbf k_W=\mathbf k_C.
$$

其计算可用 $\mathbf k_W\times\mathbf k_C$ 给出轴，`atan2` 给出夹角，并处理同向／反向特殊情况。所有满足重力对齐的旋转可写为

$$
\boxed{
Q(\beta)=\operatorname{Exp}(\beta\mathbf k_C)Q_g.
}
$$

因为 $Q Q_g^\mathsf T$ 必须保持 $\mathbf k_C$ 不变，而 $SO(3)$ 中这种旋转恰为绕 $\mathbf k_C$ 的旋转。

骨轴使用球面参数：

$$
\mathbf b_S(\vartheta,\varphi)=
\begin{bmatrix}
\sin\vartheta\cos\varphi\\
\sin\vartheta\sin\varphi\\
\cos\vartheta
\end{bmatrix}.
$$

本方法不要求把中立骨段强制设为竖直，不直接依赖穿戴映射 $B$。前举、斜前举、侧举用于提供不同方向的观测，不要求动作恰为 $90^\circ$。

## 5. 目标函数、初始化与迭代

### 5.1 基本方向残差

令 $\mathbf x=(\beta,\vartheta,\varphi)^\mathsf T$，同期训练数据为 $(R_j,\mathbf d_{C,j})$，残差为

$$
\mathbf r_j(\mathbf x)
=Q(\beta)R_j\mathbf b_S(\vartheta,\varphi)-\mathbf d_{C,j}.
$$

堆叠 $\mathbf r=(\mathbf r_1^\mathsf T,\ldots,\mathbf r_N^\mathsf T)^\mathsf T$，基本最小二乘目标为

$$
\boxed{
\min_{\mathbf x}\frac12\sum_j\|\mathbf r_j(\mathbf x)\|^2.
}
$$

单位方向的距离平方为 $2-2\cos e_j$，它不是夹角平方，也不包含手动映射先验。

### 5.2 当前实际鲁棒目标

代码以 `soft_l1` 损失作用于堆叠后的每个标量残差：

$$
\boxed{
F(\mathbf x)=
c^2\sum_{\ell=1}^{3N}
\left[
\sqrt{1+\left(\frac{r_\ell(\mathbf x)}c\right)^2}-1
\right],
\qquad c=0.08.
}
$$

小残差时近似 $\frac12\sum r_\ell^2$，较大残差影响增长较慢。$c$ 是无量纲方向分量残差尺度，不是角度门限。处理的是各分量而非整帧方向误差，不能说它会识别并剔除所有错误视觉帧，更不能自动消除系统漂移。

### 5.3 八组初值

代码分别尝试

$$
\beta_0\in\{-180^\circ,-135^\circ,-90^\circ,-45^\circ,
0^\circ,45^\circ,90^\circ,135^\circ\}.
$$

每组先构造 $Q_0$，由第一帧反求骨轴：

$$
\mathbf b_0=\operatorname{unit}(R_1^\mathsf T Q_0^\mathsf T\mathbf d_{C,1}).
$$

再提取

$$
\vartheta_0=\arccos(b_{0z}),\qquad
\varphi_0=\operatorname{atan2}(b_{0y},b_{0x}).
$$

此初值可以对齐第一帧，但不保证解释其他帧；后续优化使用全部训练观测。

### 5.4 局部求解与高斯—牛顿的关系

在 $\mathbf x_k$ 附近，

$$
\mathbf r(\mathbf x_k+\Delta\mathbf x)
\approx\mathbf r_k+\mathcal J_k\Delta\mathbf x,
\qquad
\mathcal J_k=\frac{\partial\mathbf r}{\partial\mathbf x}.
$$

普通高斯—牛顿据此求局部线性最小二乘步长。当前调用 `scipy.optimize.least_squares`，未指定 `method`，使用默认 TRF 信赖域方法，**不是 LM**；雅可比使用默认两点有限差分。鲁棒目标在求解器内部处理，不应将普通高斯—牛顿正规方程冒充实际完整算法。

每组设 `max_nfev=200`（求解预算，不是 200 次必定执行的迭代），最后选返回代价最小的组。代码没有另以 `result.success` 作为硬性接受条件，因此返回结果仍须质量验收；它不保证全局最优。

## 6. 标定动作、验收与可辨识性

当前斜前举协议只用首次中立、斜前举、侧举静止窗口训练，窗口首 $0.5s$ 和末 $0.2s$ 不参加拟合；正前举可以服务穿戴标定，不因其遮挡直接否决相机拟合。

基础拟合要求：

| 检查 | 条件 |
| --- | --- |
| 同期有效方向 | 至少 15 帧 |
| 非竖直方向 | 至少 10 帧水平投影长度 $\ge0.35$ |
| 训练方向误差 | 中位数 $<12^\circ$，P90 $<25^\circ$ |
| 最终雅可比最小奇异值 | $>0.03$ |

斜前举协议另外要求每个训练动作至少八帧同期可信观测，斜前举与侧举的平均方向相差至少 $20^\circ$。

重复斜前举、重复侧举、返回中立用于独立验收：每窗至少八帧，未经视觉补偿的 $QR\mathbf b_S$ 与视觉方向误差中位数 $<12^\circ$、P90 $<25^\circ$。这些窗口不参与参数求解。旧录制协议有不同训练窗口逻辑，不应假称所有历史数据都采用现行规则。

雅可比奇异值检查衡量当前参数化下的局部敏感性，不是严格全局唯一性证明；球面参数在极点存在退化。几乎只有同一姿态或缺少非竖直动作时，航向与骨轴可能难以分开。

另保存训练骨长中位数 $L$，用于后续视觉异常筛选。训练与独立残差是两种观测的一致性，不是 GT 准确率。

## 7. 为什么标定期姿态误差会污染参数

假设真实关系为

$$
\mathbf d_C=Q_t R_t\mathbf b_t.
$$

若估计始终带固定参考系偏差 $G$，即 $R=G R_t$，选择

$$
Q=Q_tG^\mathsf T,\qquad \mathbf b_S=\mathbf b_t
$$

即可解释观测。这只是参考系对齐。对于当前重力约束，固定绕重力轴的偏差可这样吸收；任意倾斜偏差并不能在向上参考不相应改变时自由吸收。

若 $R(t)=G(t)R_t(t)$ 且标定中 $G(t)$ 随时间变化，固定的 $Q$ 无法精确吸收。优化会折中调整 $Q$ 和骨轴，可能训练残差较小但其他姿态表现较差。

因此实际假设是：**标定期相对姿态足够准确，参考系偏差近似固定，绑带不滑动**；不是要求绝对航向为零。先做零偏学习、缩短标定过程、独立重复动作验证有帮助，但不保证消除错估骨轴。后续动态纠偏也不会自动重估 $\mathbf b_S$。

## 8. 纯航向误差映射到相机系的证明

先作理想假设：估计 $R$ 与真实姿态 $R_t$ 的倾斜都准确，故

$$
R^\mathsf T\mathbf k_W=R_t^\mathsf T\mathbf k_W.
$$

定义从估计到真值的左旋转 $E_W=R_tR^\mathsf T$，则

$$
E_W\mathbf k_W
=R_tR^\mathsf T\mathbf k_W
=R_tR_t^\mathsf T\mathbf k_W
=\mathbf k_W.
$$

故它只能绕 $\mathbf k_W$ 旋转：

$$
R_t=\operatorname{Exp}(\delta_*\mathbf k_W)R.
$$

若 $Q,\mathbf b_S$ 准确，预测与真实方向满足

$$
\begin{aligned}
\mathbf d_{t,C}
&=QR_t\mathbf b_S\\
&=Q\operatorname{Exp}(\delta_*\mathbf k_W)Q^\mathsf T\,
QR\mathbf b_S.
\end{aligned}
$$

利用

$$
Q[\mathbf k_W]_\times Q^\mathsf T=[Q\mathbf k_W]_\times,
\qquad Q\mathbf k_W=\mathbf k_C,
$$

得到

$$
\boxed{
\mathbf d_{t,C}
=\operatorname{Exp}(\delta_*\mathbf k_C)\hat{\mathbf d}_C.
}
$$

这是共轭／伴随作用：同一个修正运动换系后，轴的坐标改变，转角不变。$\delta_*$ 定义为从估计到真值的修正角。

这是纯航向模型的证明，**不是证明实际相机输出完整姿态真值**。运动加速度、视觉表面深度、骨轴错估和重力方向偏差都可能使这一理想关系仅近似成立。

## 9. 运行时怎样计算剩余航向创新

维护累计角 $\delta$ 与相机系补偿

$$
H_C(\delta)=\operatorname{Exp}(\delta\mathbf k_C).
$$

对当前同期纯 IMU 预测施加已有补偿：

$$
\mathbf p_C=H_C(\delta)QR(t)\mathbf b_S.
$$

观测方向为 $\mathbf d_C$。设投影算子

$$
P_\perp=I-\mathbf k_C\mathbf k_C^\mathsf T.
$$

归一化水平投影：

$$
\mathbf p_h=\operatorname{unit}(P_\perp\mathbf p_C),
\qquad
\mathbf d_h=\operatorname{unit}(P_\perp\mathbf d_C).
$$

剩余创新为

$$
\boxed{
\varepsilon=
\operatorname{atan2}
\left(
\mathbf k_C^\mathsf T(\mathbf p_h\times\mathbf d_h),
\mathbf p_h^\mathsf T\mathbf d_h
\right).
}
$$

它是**已有补偿后剩下的偏差**，不是每帧把原始总偏差再次累加。理想关系下 $\varepsilon=\delta_*-\delta$（按主值角理解）。

当骨段竖直时，水平投影为零，航向不可观测；接近竖直时噪声放大。因此代码对两个投影长度均设下限，而不是强行归一化极小向量。

## 10. 低通、限速与累计范围

### 10.1 一阶低通的来源

将当前观测目标暂看作常数

$$
\delta_{\rm target}=\delta_{\rm old}+\varepsilon.
$$

一阶系统

$$
\dot\delta=\frac{\delta_{\rm target}-\delta}{\tau}
$$

在时长 $\Delta t$ 内的解为

$$
\delta_{\rm new}
=\delta_{\rm target}+(\delta_{\rm old}-\delta_{\rm target})e^{-\Delta t/\tau}.
$$

代入得

$$
\boxed{
\Delta\delta_{\rm lowpass}=(1-e^{-\Delta t/\tau})\varepsilon.
}
$$

这不是随意选的比例；它是固定目标的一阶滤波精确离散形式。实际目标随运动和观测改变，故不能据此宣称实际误差严格指数收敛。

### 10.2 限速

要求 $|\Delta\delta|\le v_{\max}\Delta t$，得到

$$
\boxed{
\begin{aligned}
\Delta\delta
&=\operatorname{clip}\left(
(1-e^{-\Delta t/\tau})\varepsilon,
-v_{\max}\Delta t,v_{\max}\Delta t
\right),\\
\delta_{\rm candidate}&=\delta_{\rm old}+\Delta\delta.
\end{aligned}
}
$$

默认 $\tau=3s$、$v_{\max}=3^\circ/s$。例如 $\Delta t=0.1s$、$\varepsilon=10^\circ$，低通建议约 $0.328^\circ$，限速后实际为 $0.3^\circ$。

这里 $\Delta t$ 是相邻进入纠偏器的视觉观测时间差，**不是陀螺积分的设备采样间隔**。代码首次取 $0.1s$，之后最多取 $0.25s$，避免长空档后一次大步补偿；进入纠偏器但被其门限拒绝的观测也会更新其时间记录。

累计范围默认 $\pm30^\circ$，可配置为 $\pm5^\circ$～$\pm90^\circ$，或选择实验性“不限累计角度”。超限时拒绝本次更新，保留原补偿，不强行截断姿态。不限累计不会取消创新、限速或观测质量门限；大累计值也不等于已测出的真实误差。

## 11. 共轭回 IMU 参考系，为什么左乘

相机系修正转换为 $W$ 中的同一运动：

$$
\boxed{H_W=Q^\mathsf T H_C Q.}
$$

它绕 $\mathbf k_W$ 旋转，因为

$$
H_W=\operatorname{Exp}(\delta Q^\mathsf T\mathbf k_C)
=\operatorname{Exp}(\delta\mathbf k_W).
$$

融合姿态为

$$
\boxed{R_f(t)=H_W(t)R(t).}
$$

左乘是因为修正运动表达在固定参考系 $W$；不是表达在设备本体系。验证：

$$
QR_f\mathbf b_S
=QQ^\mathsf T H_C QR\mathbf b_S
=H_C QR\mathbf b_S.
$$

且在重力对齐精确时

$$
R_f^\mathsf T\mathbf k_W
=R^\mathsf T H_W^\mathsf T\mathbf k_W
=R^\mathsf T\mathbf k_W,
$$

即航向补偿不改变已有重力倾斜。

程序保留独立的 `gyro_tilt` 和 `gyro_rgbd` 流。相机更新输出端修正矩阵，不直接改原积分器 $R$ 或陀螺零偏；新的 IMU 包使用最近可用的补偿，继续高频输出。因此算法是持续补偿漂移，不是使原始流从此不再漂移。

## 12. 同步、冻结和安全检查

按视觉源帧主机时间，在约三秒的纯 IMU 历史中做 SLERP；拒绝大于 $100ms$ 的包间插值空档，边界之外不外推。视觉龄期超过 $500ms$、未来帧、重复／乱序帧不更新。物理曝光与设备内部采样尚非硬件同步；主机时间匹配仍有剩余误差。

实时主要门限：

| 检查 | 拒绝条件 |
| --- | --- |
| 视觉端点 | 任一无效 |
| 二维置信度 | 任一小于 $0.6$ |
| 深度支持 | 任一小于 $0.7$ |
| 骨段长度 | 非有限、短于 $0.08m$，或偏离标定长度超过 $25\%$ |
| 两方向水平投影 | 任一长度小于 $0.35$ |
| 倾斜一致性 | 两方向仰角差大于 $15^\circ$ |
| 单次航向创新 | 绝对值大于 $30^\circ$ |
| 实时死区 | 绝对值小于 $1^\circ$ |
| 累计范围 | 下一补偿将超过所选有限范围 |

实时还要求选择融合算法、相机可用、未人工冻结、参考绑定有效等。基础纠偏器默认死区为零，而实时安装显式设为 $1^\circ$；旧离线对比因此不应假称与实时每个参数完全相同。

冻结通常是停止视觉更新，保留已有补偿，IMU 仍继续传播。停用／失效会撤销继续使用资格；物理移动相机或绑带滑动不能靠软件可靠自动识别，应显式停用并重新标定。

应用标定校验设备、积分 epoch、相机会话、视觉配置，以及 $Q$ 正交性和重力对齐。当前实时重力与标定参考允许差不超过 $5^\circ$，$Q\mathbf k_W$ 与报告 $\mathbf k_C$ 的方向差允许不超过 $0.5^\circ$。这些是生命周期安全检查，不能通过随意放宽掩盖参考系错配。

## 13. 与穿戴标定和模型随动的连接

穿戴标定求的是设备归零本体系到模型中立系的映射 $B$，相机标定求的是 $Q,\mathbf b_S$。它们可用同一次动作采集，但分别拟合、分别验收。

对融合流，设运行中立矩阵为 $N_f$，模型绝对旋转为

$$
\boxed{
R_M(t)=B\bigl(N_f^\mathsf T R_f(t)\bigr)B^\mathsf T.
}
$$

$N_f$ 必须是与运行融合流配对的中立矩阵，不能混用厂商中立位或另一个 epoch 的矩阵。代码还有标准标定参考与运行归零参考的区分；当前归零只更新运行中立位。

可选竖直换基 $A$ 满足 $A\mathbf k_W=\mathbf e_z$。它用于穿戴拟合，不必插入实时相机链路，因为

$$
(AN_f)^\mathsf T(AR_f)=N_f^\mathsf T R_f.
$$

若显式改用竖直系，取 $Q_F=QA^\mathsf T$ 即有 $Q_F(AR)=QR$。

真正的反馈闭环是“融合方向预测—视觉比较—航向更新”；$B$ 与骨架渲染是其输出链路，不反馈到视觉纠偏器。

## 14. 边界、代码索引与核心摘要

本方法不能凭两个端点完整恢复骨段三轴姿态，不能独立校准解剖学轴向扭转，也尚不包含手掌长期视觉纠偏。骨段非竖直时，其绕空间重力轴的变化可以被方向观测到；但绕自身骨轴的转动不改变该方向，这两个概念不要混淆。

校准骨轴不准、人体肩髋方向不代表真重力、表面深度偏置或持续误识别，均可能形成系统误差。独立复测、鲁棒损失、低通和限速降低风险，不构成真值保证。不限累计可能长期吸收视觉错误；SO(3) 投影仅保证数值有效性，不保证方向正确。

| 代码 | 数学职责 |
| --- | --- |
| [rgbd_pose.py](../OptUSBProj-ubuntu/optusbproj_ubuntu/rgbd_pose.py) | 二维＋真实深度生成相机系三维点 |
| [rgbd_fusion.py](../OptUSBProj-ubuntu/optusbproj_ubuntu/rgbd_fusion.py) | 重力对齐、三参数拟合、方向门限、航向更新 |
| [rgbd_fuse_offline.py](../OptUSBProj-ubuntu/rgbd_fuse_offline.py) | 训练窗口、同期姿态、报告与离线对比 |
| [camera_protocol.py](../OptUSBProj-ubuntu/optusbproj_ubuntu/camera_protocol.py) | 相机训练与独立复测动作协议 |
| [combined_calibration.py](../OptUSBProj-ubuntu/optusbproj_ubuntu/combined_calibration.py) | 独立相机验收与一体流程 |
| [live_rgbd_fusion.py](../OptUSBProj-ubuntu/optusbproj_ubuntu/live_rgbd_fusion.py) | 历史插值、重力锁定、绑定、实时共轭补偿 |
| [pose.py](../OptUSBProj-ubuntu/optusbproj_ubuntu/pose.py) | 独立姿态流和模型相对运动输出 |

核心公式：

$$
\boxed{
\begin{aligned}
\hat{\mathbf d}_C(t)&=QR(t)\mathbf b_S,\\
\mathbf p_C(t)&=\operatorname{Exp}(\delta\mathbf k_C)\hat{\mathbf d}_C(t),\\
\varepsilon&=\operatorname{atan2}\!\left(
\mathbf k_C^\mathsf T(\mathbf p_h\times\mathbf d_h),
\mathbf p_h^\mathsf T\mathbf d_h\right),\\
\delta_{\rm new}&=\delta+
\operatorname{clip}\!\left(
(1-e^{-\Delta t/\tau})\varepsilon,
-v_{\max}\Delta t,v_{\max}\Delta t\right),\\
R_f(t)&=
Q^\mathsf T\operatorname{Exp}(\delta_{\rm new}\mathbf k_C)Q\,R(t).
\end{aligned}
}
$$

最后一组公式假定本次观测通过所有门限；拒绝时保留旧 $\delta$。高频传播使用最新已经完成的补偿，不把未来观测用到更早的 IMU 样本上。
