# 陀螺＋加速度计重力纠倾斜：SO(3) 数学推导

本文解释当前 `AdaptiveTiltFilter` 的数学原理，重点是参考系、重力方向、叉积误差的符号、姿态更新和采样时间。它不是完整姿态真值估计，也不是相机航向纠偏算法；代码仍将此六轴方案标为实验性估计器。

实现依据：[重力滤波器](../wt901_viz/so3_filter.py)、[设备时间处理](../wt901_viz/yaw_guard.py)、[人体姿态引擎](../OptUSBProj-ubuntu/optusbproj_ubuntu/pose.py)。相关背景：[IMU 与模型 SO(3) 映射](SO3_IMU_MODEL.md)、[连续穿戴标定](CONTINUOUS_ARM_CALIBRATION.md)。

## 1. 坐标系、符号与单位

使用列向量和右手系。设：

- $S(t)$：随设备运动的本体系。
- $W$：软件积分器建立的固定参考系，初始与首个样本的本体系重合；其 $+Z$ 不保证竖直。
- $F$：可选的、经固定换基得到的 $+Z$ 竖直参考系。
- $R(t)={}^W R_{S(t)}\in SO(3)$：设备三根本体轴在 $W$ 中的表示，满足 $\mathbf x_W=R\mathbf x_S$。
- $\mathbf k_W$：竖直向上在 $W$ 中的单位方向。
- $\mathbf a_S$：归一化加计观测，以设备本体系表示；静止且符号约定正确时指向上方。
- $\boldsymbol\omega_S$：本体系角速度，数学公式以 rad/s 计；设备包原始值以 °/s 计。
- $\Delta t$：相邻有效滤波时间的间隔，以秒计。

定义叉乘矩阵与指数映射：

$$
[\mathbf v]_\times=
\begin{bmatrix}
0&-v_z&v_y\\
v_z&0&-v_x\\
-v_y&v_x&0
\end{bmatrix},
\qquad
[\mathbf v]_\times\mathbf x=\mathbf v\times\mathbf x.
$$

$$
\operatorname{Exp}(\boldsymbol\phi)
:=\exp([\boldsymbol\phi]_\times).
$$

$\boldsymbol\phi$ 是旋转向量：方向给出右手旋转轴，模长给出弧度转角；它不需要是单位向量。后文 $D$ 与 $\Delta R$ 都是旋转矩阵，不是角度。

## 2. 加速度计为什么能提供向上方向

加速度计测量的是比力，而不是直接测量向下重力。理想模型为

$$
\mathbf f_S=R^\mathsf T(\mathbf a_{\rm linear,W}-\mathbf g_W),
\qquad
\mathbf g_W=-g_0\mathbf k_W.
$$

静止时 $\mathbf a_{\rm linear,W}=0$，故

$$
\mathbf f_S=g_0 R^\mathsf T\mathbf k_W.
$$

当前 WT 数据与代码约定静止读数指向上方，将三轴读数归一化：

$$
\boxed{
\mathbf a_S=
\frac{(a_x,a_y,a_z)^\mathsf T}
{\sqrt{a_x^2+a_y^2+a_z^2}}
\approx R^\mathsf T\mathbf k_W.
}
$$

加计分量在代码中以 $g$ 为单位，静止时模长应接近 $1$。归一化消除模长，保留方向。设备转动后，$\mathbf a_S$ 的坐标会改变，空间中的向上方向并没有改变。

运动加速度、振动、传感器偏置会污染该方向；模长接近 $1g$ 只是可靠性指标，不能证明方向完全正确。

## 3. “学习重力参考”到底学什么

### 3.1 初始化不要求 $W$ 的 $Z$ 竖直

代码初始设 $R=I$。这只定义软件参考系，不代表设备水平放置，也不代表重力沿 $W$ 的 $Z$ 轴。

因此不能未经观测就设 $\mathbf k_W=(0,0,1)^\mathsf T$。

在确认静止的时刻 $t_0$，要求

$$
R(t_0)^\mathsf T\mathbf k_W=\mathbf a_S(t_0).
$$

因 $R^{-1}=R^\mathsf T$，直接得到

$$
\boxed{\mathbf k_W=R(t_0)\mathbf a_S(t_0).}
$$

这里“学习”是建立竖直方向的坐标，不需要迭代寻找 $\mathbf k_W$。当前基础滤波器在静止持续时间达标后，使用当次姿态与归一化加计建立 `neutral_up`，并不是在这里对整个窗口做平均；其他引导流程另有静止窗口核验。

初始化时残差自然为零，因为 $\mathbf k_W$ 正是用当前 $R$ 定义的；这一步没有独立纠正初始绝对姿态，也没有确定绝对航向。

### 3.2 后续固定 $\mathbf k_W$，修正 $R$

建立后，基础滤波器保留该参考方向。后续预测是

$$
\hat{\mathbf a}_S(t)=R_{\rm pred}(t)^\mathsf T\mathbf k_W.
$$

将其与当前加计方向比较，调整的是 $R$。

如果每帧都重新令 $\mathbf k_W=R(t)\mathbf a_S(t)$，两者会永远一致，反馈残差为零，重力纠偏就失去意义。

$\mathbf k_W$ 不必等于 $(0,0,1)^\mathsf T$；只要它与 $R$ 使用同一个固定参考系，就能做纠倾斜。

## 4. 陀螺传播为什么右乘

本体系角速度满足姿态运动学：

$$
\boxed{\dot R=R[\boldsymbol\omega_S]_\times.}
$$

假定一个采样间隔内角速度近似恒定，积分为

$$
R_{\rm pred,n}
=R_{n-1}\operatorname{Exp}(\boldsymbol\omega_{S,n}\Delta t_n).
$$

角速度在随动本体系表达，所以增量右乘。若使用固定参考系角速度 $\boldsymbol\omega_W=R\boldsymbol\omega_S$，等价形式为 $\dot R=[\boldsymbol\omega_W]_\times R$。

代码用相邻原始陀螺样本的平均值减去零偏，并把度转成弧度：

$$
\bar{\boldsymbol\omega}_{S,n}
=\frac{\pi}{180}
\left(
\frac{\boldsymbol\omega^{\rm raw}_{n-1}
+\boldsymbol\omega^{\rm raw}_n}{2}
-\hat{\mathbf b}_{\omega,n}
\right).
$$

$$
\boxed{
R_{\rm pred,n}
=R_{n-1}\operatorname{Exp}(\bar{\boldsymbol\omega}_{S,n}\Delta t_n).
}
$$

这是使用平均角速度的离散近似；一般时变三维角速度增量不对易，并不等于任意运动的精确积分。

## 5. 两个重力方向之间的精确几何旋转

为简化记号，令

$$
\mathbf u=\hat{\mathbf a}_S,\qquad
\mathbf v=\mathbf a_S,\qquad
\|\mathbf u\|=\|\mathbf v\|=1.
$$

采用代码的叉积顺序：

$$
\boxed{\mathbf e_S=\mathbf v\times\mathbf u.}
$$

若两方向夹角为 $\theta\in[0,\pi]$，则

$$
\|\mathbf e_S\|=\sin\theta,
\qquad
\theta=\operatorname{atan2}
\left(\|\mathbf e_S\|,\mathbf v^\mathsf T\mathbf u\right).
$$

因此 $\mathbf e_S$ 不是单位轴，也不是完整的精确旋转向量。非平行、非反向时定义

$$
\mathbf n=\frac{\mathbf e_S}{\|\mathbf e_S\|}.
$$

以该轴表示，将预测方向 $\mathbf u$ 转到观测方向 $\mathbf v$ 的最短旋转是

$$
\boxed{
D=\operatorname{Exp}(-\theta\mathbf n),\qquad D\mathbf u=\mathbf v.
}
$$

负号来自“观测 × 预测”的叉积顺序；如果使用 $\mathbf u\times\mathbf v$，则采用正角 $\theta$。不要只说“顺时针”，因为它还依赖观察方向。

## 6. 为什么向量左乘修正，姿态却右乘逆修正

由

$$
\mathbf u=R_{\rm pred}^\mathsf T\mathbf k_W
$$

和 $D\mathbf u=\mathbf v$，有

$$
\mathbf v=D R_{\rm pred}^\mathsf T\mathbf k_W.
$$

为了使修正姿态满足 $R_{\rm corrected}^\mathsf T\mathbf k_W=\mathbf v$，可以选择

$$
R_{\rm corrected}^\mathsf T=D R_{\rm pred}^\mathsf T.
$$

取转置：

$$
\boxed{
R_{\rm corrected}
=R_{\rm pred}D^\mathsf T
=R_{\rm pred}\operatorname{Exp}(\theta\mathbf n).
}
$$

因此必须区分：

- $D$：把预测重力坐标向量转到观测向量的矩阵。
- $\Delta R=D^\mathsf T$：右乘姿态的修正增量。

把 $D$ 直接右乘到 $R_{\rm pred}$ 上会反转纠偏方向。上述选择是最短倾斜修正；仅由 $R_{\rm corrected}^\mathsf T\mathbf k_W=\mathbf v$ 并不能推出姿态解唯一，更不能称其为完整真值。

## 7. 当前代码不是一次性精确对齐，而是小步反馈

代码没有归一化叉积并施加完整 $\theta$，而是

$$
\boxed{
R_n
=R_{\rm pred,n}
\operatorname{Exp}(\gamma w_n\Delta t_n\,\mathbf e_{S,n}).
}
$$

其中 $\gamma$ 为每秒纠偏增益，默认 $0.8\,s^{-1}$；$w_n\in[0,1]$ 为加计可靠性权重。

因 $\mathbf e_S=\mathbf n\sin\theta$，这一帧真正施加的右乘旋转角为

$$
\alpha_{\rm step}=\gamma w_n\Delta t_n\sin\theta.
$$

小误差时 $\sin\theta\approx\theta$，因此相当于沿正确轴纠正一小部分角度。指数映射输入的是旋转向量，不要求单位模长。

一个可选但目前未采用的精确角度小步版本是

$$
R_n=R_{\rm pred,n}\operatorname{Exp}(\eta_n\theta\mathbf n).
$$

$\eta_n=1$ 会一次对齐；较小值只纠正部分误差。当前实现使用叉积反馈，不应将它描述成已经使用上述精确角度版本。

### 7.1 由误差目标推导叉积反馈的符号

固定当前观测 $\mathbf v$，定义方向误差

$$
V(R)=\frac12\|R^\mathsf T\mathbf k_W-\mathbf v\|^2
=1-\mathbf v^\mathsf T\mathbf u.
$$

对姿态施加本体系小扰动

$$
R'=R\operatorname{Exp}(\boldsymbol\eta),
$$

预测重力方向变为

$$
\mathbf u'=\operatorname{Exp}(-\boldsymbol\eta)\mathbf u
\approx\mathbf u-\boldsymbol\eta\times\mathbf u.
$$

因此一阶误差变化为

$$
\begin{aligned}
V(R')-V(R)
&\approx\mathbf v^\mathsf T(\boldsymbol\eta\times\mathbf u)\\
&=\boldsymbol\eta^\mathsf T(\mathbf u\times\mathbf v)\\
&=-\boldsymbol\eta^\mathsf T\mathbf e_S.
\end{aligned}
$$

选择 $\boldsymbol\eta=\gamma w\Delta t\,\mathbf e_S$，得到

$$
V(R')-V(R)\approx-\gamma w\Delta t\|\mathbf e_S\|^2\le0.
$$

这证明当前叉积顺序和正号右乘，在小步条件下是误差下降方向。

### 7.2 连续时间下降性质与边界

仅考虑纠偏、观测固定时，对应连续反馈为

$$
\dot R=R[\gamma w\mathbf e_S]_\times.
$$

于是

$$
\dot{\mathbf u}=-\gamma w\,\mathbf e_S\times\mathbf u,
\qquad
\boxed{\dot V=-\gamma w\|\mathbf e_S\|^2\le0.}
$$

又有 $V=1-\cos\theta$，在 $0<\theta<\pi$ 时

$$
\boxed{\dot\theta=-\gamma w\sin\theta.}
$$

小角度下近似 $\dot\theta=-\gamma w\theta$。权重为常数且非零时，局部时间常数约为 $1/(\gamma w)$；默认满权重时约 $1.25s$。这是理想化的局部反馈性质，不是实际运动下的收敛保证。

恰好反向 $\theta=\pi$ 时，叉积为零，当前反馈不能自行决定纠偏轴；接近反向时也会很慢。恰好同向时无需修正。离散步长太大、运动加速度或观测噪声会破坏上述理想条件，不能用连续时间证明宣称每个实际离散步骤都必然降误差。

## 8. $\Delta t$ 与纠偏增益的物理含义

$$
\Delta t_n=t_n-t_{n-1}.
$$

它不是 GUI 刷新间隔，也不是相机纠偏的更新时间。人体姿态引擎用 `PacketClock.elapsed(packet)` 提供设备包内时钟建立的滤波时间；首次样本 $\Delta t=0$。

原始包时钟由时、分、秒、毫秒构成，并处理跨午夜。当前时间处理有明确的异常回退：重复／倒退或大于 $1s$ 的异常时间跳变，会把有效时间推进 $0.02s$ 并重新锚定。因此并非任何情况下 $\Delta t$ 都等于真实片内采样间隔；这个回退不能重建丢失期间的真实运动。

$\gamma\mathbf e_S$ 给出每秒的反馈旋转速率，乘以 $\Delta t$ 才得到本帧旋转向量。采样越快，每次修正越小、每秒次数越多，在小步和相似观测条件下整体反馈速度近似一致。

## 9. 静止学习、零偏学习和动态权重

当前基础滤波器的静止候选条件是同时满足：

| 条件 | 代码阈值 |
| --- | --- |
| 加速度模长接近 $1g$ | 偏离不超过 $0.08g$ |
| 原始陀螺模长 | 不超过 $0.8^\circ/s$ |
| 相邻归一化加计方向夹角 | 不超过 $1^\circ$ |

连续候选静止累计 $0.3s$ 后，可建立尚不存在的 $\mathbf k_W$。累计达到默认 $0.8s$ 后，首次用静止原始角速度的样本均值估计三轴零偏；后续静止阶段以约 $20s$ 时间常数缓慢更新零偏。条件中断会清空该段静止统计，不会因此重置已经建立的姿态和重力参考。

这两个学习任务不能混为一谈：

- $\mathbf k_W$：固定参考系中的向上方向。
- $\hat{\mathbf b}_\omega$：要从原始角速度中扣除的零偏，单位 °/s。

基础代码的动态权重为

$$
w_a=\operatorname{clip}\left(
\frac{0.18-|\|\mathbf a^{\rm raw}\|-1|}{0.15},0,1
\right),
$$

$$
w_\omega=\operatorname{clip}\left(
\frac{60-\|\boldsymbol\omega^{\rm raw}\|}{50},0,1
\right),
\qquad w=w_aw_\omega.
$$

前式模长以 $g$ 计，后式角速度以 °/s 计。无有效加计或尚无重力参考时，$w=0$。加速度偏离 $1g$、转动过快时降低纠倾斜强度，保留陀螺传播；这些经验权重不是运动加速度的严格分离算法。

## 10. 为什么重力不能纠正航向

任取绕固定向上方向的旋转

$$
H_W(\psi)=\operatorname{Exp}(\psi\mathbf k_W),
\qquad H_W(\psi)\mathbf k_W=\mathbf k_W.
$$

定义另一姿态 $R'=H_W(\psi)R$，则

$$
R'^\mathsf T\mathbf k_W
=R^\mathsf T H_W(\psi)^\mathsf T\mathbf k_W
=R^\mathsf T\mathbf k_W.
$$

即不同航向姿态给出相同重力观测。一个方向只有两个独立约束，无法决定完整的三自由度姿态。

这是观测不可辨识性，不是增益调得不够大。静止零偏学习可以减小航向漂移源，但重力观测仍不能独立恢复航向真值；需要视觉、磁场等外部航向信息。此处“航向”指绕重力轴，不一定等于任意未对齐参考系的欧拉 $Z$ 角。

## 11. 换基矩阵 $A$：数学上成立，但不是基础滤波器每帧重置参考系

构造固定旋转 $A={}^F R_W$，使

$$
A\mathbf k_W=\mathbf e_z,\qquad
\widetilde R=AR.
$$

那么

$$
\widetilde R^\mathsf T\mathbf e_z
=R^\mathsf T A^\mathsf T\mathbf e_z
=R^\mathsf T\mathbf k_W.
$$

所以在 $W$ 中维护一般的 $\mathbf k_W$，与在 $F$ 中把向上方向写为 $\mathbf e_z$ 完全等价。$A$ 是固定换基，不是“重力纠偏后软件参考系逐渐转动”。

若 $\mathbf k_W$ 与 $\mathbf e_z$ 非平行，可用

$$
\mathbf c=\mathbf k_W\times\mathbf e_z,\quad
\theta_A=\operatorname{atan2}(\|\mathbf c\|,\mathbf k_W^\mathsf T\mathbf e_z),
\quad
A=\operatorname{Exp}\left(\theta_A\frac{\mathbf c}{\|\mathbf c\|}\right).
$$

同向时可取 $I$；反向时必须另选垂直轴做 $\pi$ 旋转。满足对齐的 $A$ 不唯一，仍可以附加绕竖直轴的旋转，所以不能由重力确定正前方。

当前链路：

| 链路 | 姿态参考系 |
| --- | --- |
| 基础陀螺＋加计滤波与人体运行流 | 保留 $R={}^W R_S$ 和 $\mathbf k_W$ |
| 普通穿戴标定拟合 | 使用 $\widetilde R=AR$，满足参考系 $Z$ 竖直的先验 |
| 相机标定 | 直接使用 $R$，拟合 $Q={}^C R_W$ 且 $Q\mathbf k_W=\mathbf k_C$ |

若相机链路改写到 $F$，令 $Q_F=QA^\mathsf T$ 即可：

$$
Q_F\widetilde R=QA^\mathsf T AR=QR.
$$

同一固定换基也不改变相对运动：

$$
(AR_0)^\mathsf T(AR_t)=R_0^\mathsf T R_t.
$$

## 12. SO(3) 数值维护不等于物理纠偏

理想情况下，指数映射增量与旋转矩阵乘积始终属于 $SO(3)$；浮点积累可能产生误差。当前实现每次传播／修正后计算

$$
\epsilon_{\rm orth}=\|R^\mathsf T R-I\|_F,\qquad
\epsilon_{\rm det}=|\det R-1|.
$$

并通过 $R=U\Sigma V^\mathsf T$ 求最近的正旋转：

$$
R_{\rm proj}
=U\operatorname{diag}(1,1,\det(UV^\mathsf T))V^\mathsf T,
$$

$$
d_{SO(3)}=\|R-R_{\rm proj}\|_F.
$$

当该距离超过默认 $10^{-8}$ 时替换为投影矩阵；代码每次调用都会进行 $3\times3$ SVD 计算距离，只在超阈值时替换。

投影保证数值旋转有效性，却不能消除航向漂移或错误加计方向：一个姿态可以严格属于 $SO(3)$，但物理方向估错。

## 13. 总结与代码对应

核心链条为

$$
\boxed{
\begin{aligned}
\mathbf k_W&=R(t_0)\mathbf a_S(t_0)\quad\text{（静止初始化一次）},\\
R_{\rm pred,n}&=R_{n-1}\operatorname{Exp}(\bar{\boldsymbol\omega}_{S,n}\Delta t_n),\\
\hat{\mathbf a}_{S,n}&=R_{\rm pred,n}^\mathsf T\mathbf k_W,\\
\mathbf e_{S,n}&=\mathbf a_{S,n}\times\hat{\mathbf a}_{S,n},\\
R_n&=R_{\rm pred,n}\operatorname{Exp}(\gamma w_n\Delta t_n\,\mathbf e_{S,n}).
\end{aligned}
}
$$

| 数学量／步骤 | 代码 |
| --- | --- |
| $R$、$\mathbf k_W$ | `AdaptiveTiltFilter.rotation`、`neutral_up` |
| 归一化加计 $\mathbf a_S$ | `acc_unit` |
| 预测重力 $\hat{\mathbf a}_S$ | `predicted_up_body` |
| 叉积误差 | `tilt_error = cross(acc_unit, predicted_up_body)` |
| 指数映射 | `rotation_step()` |
| 重力修正 | `predicted @ rotation_step(tilt_error * gain * weight * dt)` |
| 设备时间链路 | `PoseEngine.feed()` → `PacketClock.elapsed()` → `AdaptiveTiltFilter.update()` |
| 穿戴拟合的固定换基 | `PoseEngine.calibration_orientation_for()` |

结论：该算法用陀螺高频传播姿态，以固定重力参考约束倾斜，按观测可靠性小步纠偏；它不使用厂商 R/P/Y 来传播该姿态流，不自动获得绝对航向，也不把重力对齐结果称为完整姿态真值。
