# 人体模型的连续轴映射：以右上臂和右前臂为例

这份文档解释当前 Ubuntu 人体模型中的**穿戴标定**：两台 WT901 分别绑在右上臂、右前臂后，程序怎样由指定动作求出各自的连续 $3\times3$ 轴映射矩阵，并由两段姿态求肘关节旋转。文中的公式对应现有 [拟合函数](../OptUSBProj-ubuntu/optusbproj_ubuntu/guided_calibration.py) 和 [模型姿态引擎](../OptUSBProj-ubuntu/optusbproj_ubuntu/pose.py)，并不是尚未实现的理想算法。若想先看单台设备的李群/李代数基础，可读 [从 IMU 到模型的 SO(3) 说明](SO3_IMU_MODEL.md)。

## 一、先分清三件事

1. **设备姿态**：由所选姿态算法提供 $3\times3$ 旋转矩阵，表示设备本体系在该算法参考系中的方向。厂商 R/P/Y 模式按 ZYX 转矩阵；陀螺＋加计模式使用软件积分与重力纠倾斜的矩阵；RGB-D 模式还包含相机航向纠偏。普通穿戴标定跟随部位所选算法，并非一律使用厂商 R/P/Y。
2. **设备到骨段的轴映射**：绑带方向未必与模型轴对齐。右上臂用 $B_U\in SO(3)$，右前臂用 $B_F\in SO(3)$，两者分别标定，不应默认相等。
3. **关节相对旋转**：先分别算上臂、前臂在同一模型系下的方向，再由前臂相对上臂的方向求肘关节运动。前臂设备的某个欧拉角本身不等于肘角。

模型层级是 `躯干 → 右上臂 → 右前臂 → 右手`。右上臂和右前臂的默认局部骨长方向均取模型 $-Z$：自然下垂为 $-Z$，向正前方平举对应 $+X$，向人体右侧平举对应 $-Y$。

### 1.1 不同标定入口使用的姿态来源

| 标定入口 | 实际姿态来源 |
| --- | --- |
| 普通单部位穿戴标定 | 对应部位当前选择的模式 |
| 普通“双臂联合标定” | 每个部位各自当前选择的模式，不要求四个部位同一模式 |
| “双臂一体标定 · 穿戴＋相机” | 参与部位必须选择 `gyro_tilt`，使用未加入 RGB-D 纠偏的陀螺＋加计流 |

`PoseEngine` 初始默认模式是 `vendor_rpy`，但用户可改变。普通采样的 `_attach_orientation_streams()` 根据设备绑定部位读取 `algorithm_modes`，将选中算法的拟合矩阵写入 `orientation_matrix`；同时在 `orientation_by_mode` 保存各算法的原始矩阵。拟合函数 `_hold()` 优先读取 `orientation_matrix`，旧记录没有这个字段时才从 R/P/Y 重建；如果记录明确标注为陀螺模式而缺少拟合矩阵，则直接拒绝，不偷偷退回厂商 R/P/Y。[采样代码](../OptUSBProj-ubuntu/optusbproj_ubuntu/calibration_gui.py)、[静止窗口计算](../OptUSBProj-ubuntu/optusbproj_ubuntu/guided_calibration.py)

一体标定单独使用 `gyro_tilt`，避免用既有相机纠偏反过来拟合这轮相机标定。普通标定虽然可读取已经具备资格的 `gyro_rgbd` 流，但这不表示推荐在相机纠偏持续变化时重新求穿戴映射：后面的推导要求整轮参考系一致，动态纠偏会进入观测，可能影响拟合。[一体标定计算与应用检查](../OptUSBProj-ubuntu/optusbproj_ubuntu/combined_calibration.py)

### 1.2 拟合姿态与运行姿态：区分两个参考系

令 $\bar R_i(t)$ 是姿态引擎维护的**原始运行矩阵**，其参考系记为 $W_i$。陀螺估计器启动时的参考系不保证 $+Z$ 竖直；普通标定通过 `calibration_orientation_for()` 根据已建立的向上重力方向 $\mathbf g_{W_i}$ 构造对齐旋转 $A_i$：

$$
A_i\mathbf g_{W_i}=\mathbf e_z,
\qquad R_i(t)=A_i\bar R_i(t).
$$

这里 $R_i(t)$ 才是送入本篇拟合公式的矩阵，其参考系记为 $F_i$。对陀螺＋加计及 RGB-D 模式，都执行这个重力对齐步骤；厂商模式直接使用其矩阵，可视为 $A_i=I$，竖直参考由厂商算法提供。对齐只确定竖直，不能确定人体正前方，剩余航向仍由前举、侧举估计。若陀螺重力参考 `neutral_up` 尚未建立，标定拒绝计算并提示先静止。[参考系对齐代码](../OptUSBProj-ubuntu/optusbproj_ubuntu/pose.py)

在同一设备参考系／重力基准不变的前提下，$A_i$ 是固定的左乘换基，而不是每帧重设姿态。它不会改变相对运动：

$$
R_i(t_0)^\mathsf T R_i(t)
=\bar R_i(t_0)^\mathsf T A_i^\mathsf T A_i\bar R_i(t)
=\bar R_i(t_0)^\mathsf T\bar R_i(t).
$$

下文第三至第六节的 $R_0,R_f,R_s$ 都是**拟合参考系 $F_i$ 下**的静止平均姿态；第七节显式使用 $\bar R$ 和 $\bar N$ 描述运行。切勿将拟合用的 $R_0=A\bar N$ 与运行原始矩阵 $\bar R(t)$ 直接混用。

## 二、用户做的五个动作分别起什么作用

单部位标定时对右上臂、右前臂分别执行；“双臂联合标定”可以在同一组时间窗口采集多台设备，但**每个部位仍单独拟合自己的矩阵**，不是共用一套 $B$。前臂标定时须尽量伸直肘部，让前臂骨段与整条手臂的目标方向一致。

| 步骤 | 用户姿势 | 在算法中的作用 |
| --- | --- | --- |
| `neutral` | 双臂自然下垂、身体站直 | 建立该设备的标定零位 $R_0$，推算骨段长轴在设备坐标中的方向 |
| `forward` | 目标手臂伸直，向正前方举至肩高 | 用骨段方向 $+X$ 提供第一个水平航向约束，参加拟合 |
| `lateral` | 从中立出发，向人体侧方举至肩高 | 右侧以 $-Y$ 为目标，提供第二个航向约束，参加拟合 |
| `check_forward` | 再做一次前平举 | **独立复测**；不参加矩阵求解，用于检验重复性 |
| `check_neutral` | 放下手臂，回到最初姿势 | **独立复测**；检查设备是否回到标定初始方向 |

每一步先给准备时间，之后才在静止窗口内收样；准备期间走位、抬臂的过渡数据不参与拟合。静止窗口至少需 **30 包**。每包按上述模式取得旋转矩阵，再用 `scipy.spatial.transform.Rotation.mean()` 求该窗口的平均姿态，因此不会直接对可能跨越 $\pm180^\circ$ 的欧拉角做普通算术平均。这里统计的是静止窗口的旋转平均，并非重新积分设备角速度。[采样与静止统计](../OptUSBProj-ubuntu/optusbproj_ubuntu/guided_calibration.py)

下文只写某一个部位的计算，右上臂与右前臂完全重复此过程。令五个静止窗口的平均旋转矩阵分别为 $R_0,R_f,R_s,R_{fc},R_{nc}$。这些矩阵始终取自**同一台**设备。

## 三、从“下垂”反求骨段在设备坐标中的方向

模型已知中立时骨段方向

$$\mathbf d=(0,0,-1)^\mathsf T.$$

设备可能歪着绑，但设备相对骨段固定，因此该骨段在设备本体坐标中的方向是

$$\boxed{\mathbf v=R_0^\mathsf T\mathbf d.}$$

严格地说，上式中的 $\mathbf d$ 是拟合参考系 $F$ 中的向下方向；模型系 $M$ 中的向下方向恰好也写成同样的坐标。这个数值相同依赖两个独立先验：**$F$ 与 $M$ 的正 $Z$ 同向，以及真实骨段在中立窗口竖直向下**。不要求参考系原点重合，也不要求设备本体 $Z$ 竖直。自然下垂时骨段外展或前伸，会违反第二个先验，带来拟合偏差；本篇的纯 IMU 方法不会自动识别并消除这种偏差。视觉辅助穿戴标定使用实际观测方向，是另一条流程。

关键点是：$\mathbf v$ 由**穿戴后的设备中立姿态**求得，未假设设备的 $+X,+Y,+Z$ 中某一根恰好沿前臂或上臂。只要绑带在这轮动作中没有滑动，$\mathbf v$ 可视为同一根固定在设备上的单位向量。

于是前举和侧举窗口实际观测到的骨段空间方向为

$$\mathbf a_f=R_f\mathbf v,\qquad \mathbf a_s=R_s\mathbf v.$$

它们可能与理想的 $+X$ 和 $-Y$ 不重合，原因包括设备融合参考系与模型朝向不同、穿戴偏差、姿势没做到位、绑带滑动、航向漂移等。**标定不能自动区分这些原因**，只能在规定动作假设下拟合并给出误差。

## 四、如何由前举、侧举估计航向修正

右臂的两个目标方向固定为

$$\mathbf b_f=(1,0,0)^\mathsf T,\qquad
\mathbf b_s=(0,-1,0)^\mathsf T.$$

当前实现利用重力确定竖直方向，**只搜索绕竖直 $Z$ 轴的一个航向角** $\alpha$。记

$$H=R_z(\alpha).$$

### 4.1 最初要最小化哪个目标函数

理想情况下 $H\mathbf a_f\approx\mathbf b_f$、$H\mathbf a_s\approx\mathbf b_s$。这里同一个 $H$ 必须同时解释前举和侧举，不能给两次动作各自选择一个修正角。相应的方向误差最小二乘问题是

$$
\boxed{
\min_{\alpha\in\mathbb R} E(\alpha),\qquad
E(\alpha)=\sum_{k\in\{f,s\}}
\left\|R_z(\alpha)\mathbf a_k-\mathbf b_k\right\|^2.
}
$$

角度按 $2\pi$ 周期等价。这里的误差是三维单位方向向量之间的**欧氏距离平方**，不是直接把两个夹角平方相加。对于单个方向，距离平方等于 $2-2\cos e_k$；因此它与夹角单调对应，但多项相加后的优化问题不应称为“夹角平方最小化”。

因为旋转保持长度，且 $\|\mathbf a_k\|=\|\mathbf b_k\|=1$，展开可得

$$
\begin{aligned}
E(\alpha)
&=\sum_k\left(
\|\mathbf a_k\|^2+\|\mathbf b_k\|^2
-2\mathbf b_k^\mathsf T R_z(\alpha)\mathbf a_k\right)\\
&=4-2\sum_k\mathbf b_k^\mathsf T R_z(\alpha)\mathbf a_k.
\end{aligned}
$$

因此，最小化方向误差等价于最大化旋转后的方向与目标的总点积。

### 4.2 水平分量是什么意思，为什么出现 $C,S$

令 $\mathbf a_{k,xy}=(a_{kx},a_{ky})^\mathsf T$，它是骨段方向在水平 $XY$ 平面上的投影。代码只取向量的前两个元素，**不对投影再次归一化**。目标方向 $\mathbf b_f,\mathbf b_s$ 都是水平向量，所以 $b_{kz}=0$，总点积只取决于水平投影。

绕 $Z$ 轴旋转的矩阵为

$$
R_z(\alpha)=
\begin{bmatrix}
\cos\alpha&-\sin\alpha&0\\
\sin\alpha&\cos\alpha&0\\
0&0&1
\end{bmatrix}.
$$

由于 $\mathbf e_z\times\mathbf a_k=(-a_{ky},a_{kx},0)^\mathsf T$，旋转后的水平分量可写为

$$
\bigl(R_z(\alpha)\mathbf a_k\bigr)_{xy}
=\cos\alpha\,\mathbf a_{k,xy}
+\sin\alpha\,(\mathbf e_z\times\mathbf a_k)_{xy}.
$$

再与目标水平向量点乘，并对两次动作求和：

$$
\begin{aligned}
\sum_k\mathbf b_k^\mathsf T R_z(\alpha)\mathbf a_k
&=\cos\alpha\sum_k\mathbf a_{k,xy}\cdot\mathbf b_{k,xy}\\
&\quad+\sin\alpha\sum_k
(\mathbf e_z\times\mathbf a_k)_{xy}\cdot\mathbf b_{k,xy}.
\end{aligned}
$$

因此自然得到两个与 $\alpha$ 无关的系数：

$$C=\sum_{k\in\{f,s\}}\mathbf a_{k,xy}\cdot\mathbf b_{k,xy},\qquad
S=\sum_{k\in\{f,s\}}(\mathbf e_z\times\mathbf a_k)_{xy}\cdot\mathbf b_{k,xy},$$

其中 $\mathbf e_z=(0,0,1)^\mathsf T$。$C$ 是所有动作贡献的**余弦系数之和**，$S$ 是所有动作贡献的**正弦系数之和**。“水平分量汇总”指的就是这种系数汇总，既不是把两个三维观测向量直接相加，也不是把两个修正角取算术平均。

### 4.3 $J(\alpha)$ 是什么，怎样得到最优角

定义待最大化的对齐得分

$$
\boxed{
J(\alpha)=\sum_k\mathbf b_k^\mathsf T R_z(\alpha)\mathbf a_k
=C\cos\alpha+S\sin\alpha.
}
$$

它与原始误差函数的关系是

$$
\boxed{E(\alpha)=4-2J(\alpha).}
$$

所以 $E$ 要最小化，$J$ 要最大化，不应混淆。令 $\rho=\sqrt{C^2+S^2}$，当 $\rho>0$ 时令 $\phi=\operatorname{atan2}(S,C)$，则

$$
C=\rho\cos\phi,\qquad S=\rho\sin\phi,
$$

$$
J(\alpha)=\rho\cos(\alpha-\phi).
$$

余弦在 $\alpha=\phi+2n\pi$ 时达到最大值 $\rho$，因此无先验时的全局最优解为

$$
\boxed{\alpha^*=\operatorname{atan2}(S,C)\pmod{2\pi}.}
$$

若 $\rho=0$，$J$ 与 $\alpha$ 无关，没有可辨识的最佳航向。代码在加入先验前检查 $\rho<0.3$ 并拒绝拟合。较小的 $\rho$ 既可能来自骨段接近竖直、水平投影太短，也可能来自前举与侧举支持的修正方向相互抵消；它不是单独的精度保证。

### 4.4 右臂的具体形式和例子

代入右臂目标 $\mathbf b_f=(1,0,0)^\mathsf T$、$\mathbf b_s=(0,-1,0)^\mathsf T$，可直接得到

$$
\boxed{C=a_{fx}-a_{sy},\qquad S=-a_{fy}-a_{sx}.}
$$

假设前举、侧举观测都相对各自目标绕 $Z$ 偏了 $+\beta$，且骨段水平，那么

$$
\mathbf a_f=(\cos\beta,\sin\beta,0)^\mathsf T,
\qquad
\mathbf a_s=(\sin\beta,-\cos\beta,0)^\mathsf T.
$$

于是 $C=2\cos\beta$、$S=-2\sin\beta$，求得 $\alpha^*=-\beta$。例如共同偏了 $+20^\circ$，就共同修正 $-20^\circ$。

更一般地，设水平投影长度为 $r_k$，观测与目标的水平角分别为 $\theta_{a,k},\theta_{b,k}$，单次动作希望的修正角为 $\delta_k=\theta_{b,k}-\theta_{a,k}$，则

$$
(C,S)=\sum_k r_k(\cos\delta_k,\sin\delta_k).
$$

这说明它等价于对修正角进行以水平投影长度为权重的**圆周方向合成**。当 $r_k=0$ 时该动作没有水平航向信息，$\delta_k$ 本身也无须定义；两次动作不一致时，算法求共同折中角。这样也避免了普通角度平均在 $\pm180^\circ$ 附近的跨界问题。

### 4.5 手动映射的弱先验如何进入目标函数

#### 4.5.1 先验角来自哪里，它不是 GT

GUI 原先的手动轴映射给出一个正交带符号排列矩阵 $B_{\rm manual}$。它只提供弱先验：先取

$$\alpha_{\rm prior}=\operatorname{atan2}
\left((B_{\rm manual}R_0^\mathsf T)_{1,0},
      (B_{\rm manual}R_0^\mathsf T)_{0,0}\right),$$

这里矩阵下标沿用 NumPy 的从零开始索引。$B_{\rm manual}R_0^\mathsf T$ 是手动映射所暗示的“拟合参考系到模型系”的旋转，代码从其第一列的水平投影取出粗略航向；它未必是纯 $Z$ 轴旋转。若该列水平投影接近零，这个角本身就缺少几何意义，代码目前没有针对这一先验退化情况单独设门限。

手动映射可能有较大偏差，因此 $\alpha_{\rm prior}$ **不是真值 GT，不是必须满足的等式约束**。先验的意图只是：主要拟合动作观测，同时适度惩罚偏离手动估计的解。下面的惩罚形式是一个人为选择的正则化设计，不是原始动作误差必然产生的项。

#### 4.5.2 为什么把角度写成 $\mathbf h(\alpha)$

不能直接把角度当作普通实数处理：例如 $179^\circ$ 与 $-179^\circ$ 实际只差 $2^\circ$，直接相减却得到 $358^\circ$。我们将角度嵌入二维单位圆：

$$
\boxed{
\mathbf h(\alpha)=
\begin{bmatrix}\cos\alpha\\\sin\alpha\end{bmatrix},
\qquad
\mathbf h_{\rm prior}=\mathbf h(\alpha_{\rm prior}).
}
$$

它可以看成二维旋转作用于水平单位 $X$ 向量所得的方向：

$$
\mathbf h(\alpha)=
\begin{bmatrix}
\cos\alpha&-\sin\alpha\\
\sin\alpha&\cos\alpha
\end{bmatrix}
\begin{bmatrix}1\\0\end{bmatrix}.
$$

因此 $\mathbf h$ 是角度的**二维方向表示**，不是新的设备姿态、骨段方向或三维映射矩阵。它满足 $\|\mathbf h(\alpha)\|=1$ 且 $\mathbf h(\alpha+2\pi)=\mathbf h(\alpha)$，自动保留角度周期性。

用单位圆上两个点之间的欧氏距离平方，也称弦距离平方，定义先验惩罚：

$$
\boxed{P(\alpha)=\|\mathbf h(\alpha)-\mathbf h_{\rm prior}\|^2.}
$$

展开即可得到文档中余弦形式的来源：

$$
\begin{aligned}
P(\alpha)
&=\|\mathbf h(\alpha)\|^2+\|\mathbf h_{\rm prior}\|^2
-2\mathbf h(\alpha)^\mathsf T\mathbf h_{\rm prior}\\
&=2-2\bigl(\cos\alpha\cos\alpha_{\rm prior}
+\sin\alpha\sin\alpha_{\rm prior}\bigr)\\
&=2\bigl[1-\cos(\alpha-\alpha_{\rm prior})\bigr].
\end{aligned}
$$

两角相同时 $P=0$，相差 $180^\circ$ 时 $P=4$，相差 $360^\circ$ 时又为零。在最短角度差从 $0$ 增至 $\pi$ 的范围内，偏离越大，惩罚越大。对于很小的角度差 $\delta$（以弧度计），$2(1-\cos\delta)\approx\delta^2$；它局部类似角度平方误差，但不会出现跨越 $\pm180^\circ$ 的问题。

#### 4.5.3 怎样嵌入原始误差函数 $E(\alpha)$

原始 $E(\alpha)$ 仍然只表示前举、侧举观测误差。加入先验后，应另命名为 $E_\lambda(\alpha)$：

$$
\boxed{
\begin{aligned}
E_\lambda(\alpha)
&=E(\alpha)+\lambda P(\alpha)\\
&=\sum_{k\in\{f,s\}}
\|R_z(\alpha)\mathbf a_k-\mathbf b_k\|^2
+\lambda\|\mathbf h(\alpha)-\mathbf h_{\rm prior}\|^2.
\end{aligned}
}
$$

第一项要求动作对齐，第二项要求适度靠近手动估计。目标是**一次性求这个联合目标的最优角**，不是先算一个动作角，再单独修改它。

代码默认采用无量纲权重 $\lambda=0.15$，用户可在主界面“穿戴标定参数…”或穿戴窗口调整（范围 0～2）。将 $E(\alpha)=4-2J(\alpha)$ 和刚才的 $P(\alpha)$ 代入：

$$
\begin{aligned}
E_\lambda(\alpha)
&=4-2J(\alpha)+2\lambda
-2\lambda\cos(\alpha-\alpha_{\rm prior})\\
&=4+2\lambda
-2\bigl[J(\alpha)+\lambda\cos(\alpha-\alpha_{\rm prior})\bigr].
\end{aligned}
$$

因子 $2$ 来自单位向量距离平方的展开，不是额外引入的调参权重。

#### 4.5.4 等价的新最大化目标 $J_\lambda(\alpha)$

定义

$$
\begin{aligned}
J_\lambda(\alpha)
&=J(\alpha)+\lambda\cos(\alpha-\alpha_{\rm prior})\\
&=(C+\lambda\cos\alpha_{\rm prior})\cos\alpha
+(S+\lambda\sin\alpha_{\rm prior})\sin\alpha.
\end{aligned}
$$

于是

$$
\boxed{
E_\lambda(\alpha)=4+2\lambda-2J_\lambda(\alpha),
\qquad
\arg\min_\alpha E_\lambda(\alpha)
=\arg\max_\alpha J_\lambda(\alpha).
}
$$

$4+2\lambda$ 与 $\alpha$ 无关，所以去掉常数并将最小化转成最大化，不改变最优解。$J_\lambda$ 不是另一个独立猜出的目标，而是 $E_\lambda$ 的等价表达。

令 $C_\lambda=C+\lambda\cos\alpha_{\rm prior}$、$S_\lambda=S+\lambda\sin\alpha_{\rm prior}$，就仍然有 $J_\lambda=C_\lambda\cos\alpha+S_\lambda\sin\alpha$。再次使用上一节的余弦合成，得到代码实际使用的解：

$$\boxed{\alpha=\operatorname{atan2}
\left(S+\lambda\sin\alpha_{\rm prior},
      C+\lambda\cos\alpha_{\rm prior}\right).}$$

一般而言，$(C_\lambda,S_\lambda)=(0,0)$ 时该目标没有唯一最佳航向。代码先要求 $\rho=\sqrt{C^2+S^2}\ge0.3$；默认 $\lambda=0.15$ 时合成向量长度至少为 $\rho-\lambda\ge0.15$。允许用户调大权重后，额外检查合成向量长度；小于 $10^{-8}$ 时拒绝拟合，提示减小权重或重录，不使用 `atan2(0,0)`。

#### 4.5.5 为什么叫弱先验，而不是“先验占 15%”

几何上，最终角度是两个二维投票向量相加后的方向：

$$
\begin{bmatrix}C_\lambda\\S_\lambda\end{bmatrix}
=\underbrace{\begin{bmatrix}C\\S\end{bmatrix}}_{\text{动作证据，长度 }\rho}
+\underbrace{\lambda\begin{bmatrix}
\cos\alpha_{\rm prior}\\\sin\alpha_{\rm prior}
\end{bmatrix}}_{\text{先验，长度 }\lambda}.
$$

$\lambda=0.15$ 不代表先验固定占 $15\%$，其相对影响取决于 $\lambda/\rho$。两次水平动作完全一致时 $\rho=2$，先验相对很弱；动作证据接近拒绝阈值 $\rho=0.3$ 时，它的影响就明显得多。错误先验仍可能拉偏结果，因此这只是软约束，不保证提高准确率。设 $\lambda=0$ 就恢复纯动作拟合，并跳过手动映射先验计算；重力、规定动作方向和质量验收仍保留。

该配置由单部位（含平举双手）、双臂联合及一体标定的原 IMU 路径共用，保存在本地配置。采集时冻结权重并写入 `steps.json` 或 `camera_guide.json`；结果记录实际 `prior_weight`。离线命令 `--prior-weight` 可显式覆盖，未指定时复用录制参数，旧记录默认 0.15。视觉辅助穿戴的主拟合没有此先验，参数只作用于其原 IMU 对照。躯干 SVD 拟合也支持该参数，但目标函数不同；旧左腕屈腕算法的正则与25°硬约束不在此配置范围内。

因此手动映射不会强迫结果落回 $90^\circ$ 的整数轴排列。例如前举方向在水平面里相对模型前方偏了约 $20^\circ$，拟合的 $H$ 会倾向于施加约 $-20^\circ$ 修正；侧举数据和弱先验共同决定最终值。**代码没有自由优化一个任意三自由度的 $B$**，而是用 $R_0$ 固定中立姿态的信息，再用这两个动作估计一个水平航向修正。

## 五、连续 $3\times3$ 映射矩阵从何而来

估计 $H$ 后，最终输出的矩阵是

$$\boxed{B=H R_0.}$$

$H,R_0\in SO(3)$，故 $B\in SO(3)$。它每个元素是连续实数，可以表达不是整 $90^\circ$ 的安装方向；这就是“连续轴映射”与手动 `+X/-Y/+Z` 下拉选项的差别。结果同时保存 $B$ 和本轮使用的 $R_0$，应用时二者一起进入姿态引擎。

检查这条公式是否合理：由于 $H$ 仅绕竖直轴旋转，$H^\mathsf T\mathbf d=\mathbf d$，因此

$$B^\mathsf T\mathbf d=R_0^\mathsf T H^\mathsf T\mathbf d
=R_0^\mathsf T\mathbf d=\mathbf v.$$

也就是说，映射矩阵会把模型中立位的骨段方向 $-Z$ 对应到设备中那根由首个静止姿势测出的骨段方向。它没有把设备外壳上印的某根轴武断地当作骨骼长轴。

需要强调可辨识性边界：仅观测骨段长轴的指向，**无法独立确定绕骨段长轴的解剖学扭转角**。当前 $B=HR_0$ 对剩余方向作了一个由中立设备姿态与航向约定决定的选择，并不表示前举/侧举动作已经测出了真实的上臂内外旋或前臂旋前旋后。想增强这部分标定，需要加入专门的屈肘、轴向旋转动作和人体关节约束。

## 六、拟合误差与“通过/需重录”

本节对应普通上臂／前臂的 `fit_guided_arm()` 及单部位、双臂联合窗口。当前双手的同动作骨段拟合也复用该函数；躯干转轴标定、旧手腕动作标定、视觉辅助穿戴和相机标定有各自的验收规则，不能直接套用本节阈值。一体标定复用该函数的穿戴部分，但外层提示字段的语义不同，见 6.5 节。

### 6.1 拟合方向误差与独立复测误差

前举、侧举的预测方向为

$$
\hat{\mathbf b}_f=H\mathbf a_f,\qquad
\hat{\mathbf b}_s=H\mathbf a_s.
$$

拟合误差是三维单位方向之间的夹角：

$$
e_f=\arccos(\hat{\mathbf b}_f\cdot\mathbf b_f),\qquad
e_s=\arccos(\hat{\mathbf b}_s\cdot\mathbf b_s).
$$

代码先将点积截断到 $[-1,1]$，避免浮点舍入使反余弦无效，再把弧度转成度供显示和判定。这检验的是预测方向与**规定动作目标**的一致性，不是与外部 GT 的误差，也不是第四节包含先验惩罚的 $E_\lambda$ 数值。

拟合只调水平航向，却用完整三维夹角验收，所以抬臂高度不足、前臂未伸直等竖直偏差也会反映在误差里；当前没有单独拟合“肩高角度”。

独立复测计算

$$
e_{\rm repeat}=\angle(HR_{fc}\mathbf v,\mathbf b_f),\qquad
e_{\rm return}=\operatorname{ang}(R_0^\mathsf T R_{nc}).
$$

再次前举检查骨段方向；返回中立检查**完整姿态**的相对旋转角。因此，即使骨段已向下，仅绕长轴的姿态没有恢复，也可能使返回中立不通过。这两个复测窗口不参加 $H$ 的求解。[误差实现](../OptUSBProj-ubuntu/optusbproj_ubuntu/guided_calibration.py)

### 6.2 静止窗口统计量的定义

对步骤 $k$ 的每包姿态 $R_{k,j}$，先用 `Rotation.mean()` 求平均姿态 $\bar R_k$，再计算每包相对于平均姿态的旋转距离：

$$
\theta_{k,j}=\operatorname{ang}(\bar R_k^\mathsf T R_{k,j}),
\qquad
\operatorname{spread}_k=P_{90}(\{\theta_{k,j}\}_j).
$$

“姿态波动”是上述距离的第 90 百分位，**不是最大波动、标准差或单个欧拉角波动**；最终以度为单位。

陀螺统计先逐包求三轴角速度模长，再取 P90：

$$
\operatorname{gyroP90}_k
=P_{90}\!\left(
\left\{\sqrt{\omega_{x,j}^2+\omega_{y,j}^2+\omega_{z,j}^2}\right\}_j
\right).
$$

这里使用包中的原始陀螺数据，不是在验收函数中重新积分姿态或学习零偏。姿态波动和陀螺检查采用逻辑“或”：任一超限就提示未充分静止。

加速度检查先逐包求模长，再取窗口中位数：

$$
m_{a,k}=\operatorname{median}_j
\sqrt{a_{x,j}^2+a_{y,j}^2+a_{z,j}^2},
\qquad |m_{a,k}-1|>0.2.
$$

其中加速度分量以 $g$ 为单位。它不是对平均加速度向量求模长，也没有在这里单独检查加速度波动。

### 6.3 无法拟合、需重录与通过

以下前置问题直接抛出 `ValueError`，而不是返回一个已求解的合格矩阵：

| 条件 | 实际处理 |
| --- | --- |
| 缺少规定动作窗口 | 无法拟合 |
| 任一静止窗口少于 30 包 | 无法拟合 |
| 陀螺模式记录缺少拟合用的重力对齐矩阵 | 无法拟合；提示先建立重力参考 |
| 水平证据 $\sqrt{C^2+S^2}<0.3$ | 无法拟合；不靠手动先验强行出结果 |

普通单部位窗口显示“无法完成标定”；普通联合窗口按部位保存失败信息。这不同于“矩阵已算出，但质量不合格”。一体标定外层报告会把捕获的计算异常包装成不接受／需重录条目。

求解完成后，再执行硬性质量检查：

| 检查 | 当前判据 |
| --- | --- |
| 姿态波动 / 陀螺模长 P90 | 任一步大于 $5^\circ$，或大于 $20^\circ/s$ |
| UDP 到达间隔 | 任一间隔大于 500 ms，或至少三个间隔大于 150 ms |
| 加速度模长 | 模长的窗口中位数与 $1g$ 相差大于 $0.2g$ |
| 前举／侧举拟合误差 | 任一大于 $25^\circ$ |
| 再次前举／返回中立误差 | 分别大于 $25^\circ$／$15^\circ$ |

这些硬阈值均为严格的 `>`：恰好等于阈值，不因该项判失败。拟合器将超限项写入 `warnings`；存在任一此类警告时返回 `status="需重录"`，否则返回 `status="通过"`。需重录时仍可保存矩阵及诊断数据，但不能应用。

UDP 检查只在窗口内所有记录都有 `received_ns` 时执行，检查相邻收包时间之差；加速度检查只在所有记录都有三轴加速度时执行。旧记录或直接调用拟合函数时，字段缺失会**跳过相应检查**，不意味着相应质量已经验证。当前普通 GUI 正常采样会提供这些字段。

### 6.4 躯干固定检查的范围与例外

普通单部位手臂流程和普通联合流程在绑定躯干设备时，都会尝试求躯干各静止窗口的平均姿态 $T_k$，计算

$$
m_k=\operatorname{ang}(T_0^\mathsf T T_k),
\qquad k\ne\mathrm{neutral}.
$$

若 $\max_k m_k>12^\circ$，外层窗口把部位结果改为“需重录”。这是**各静止窗口平均姿态相对中立平均姿态的变化**，不是整个准备／运动过程的最大瞬时转角，也不检查躯干平移。[参考运动统计](../OptUSBProj-ubuntu/optusbproj_ubuntu/guided_calibration.py)

该检查依赖参考设备数据可计算。如果参考姿态统计抛出 `ValueError`：

- 普通单部位窗口跳过这项检查。
- 普通联合窗口增加“躯干设备数据不足，无法核验躯干是否固定”的建议提醒，但不因此强制判失败。

未绑定躯干时也不能把“通过”解释成已独立验证躯干固定。上述外层处理与目标设备本身数据不足导致的失败不是一回事。[窗口检查](../OptUSBProj-ubuntu/optusbproj_ubuntu/calibration_gui.py)

### 6.5 建议提醒与硬性拒绝，不要混用

普通双臂联合窗口还有更严格的建议：

| 建议条件 | 处理 |
| --- | --- |
| 前臂侧举误差 $\ge15^\circ$ | 添加 `quality_notes`，建议检查肘伸直、前臂旋转与绑带 |
| 躯干窗口姿态变化 $\ge5^\circ$ | 添加 `quality_notes`，建议保持身体朝向 |
| 躯干参考数据不足，无法核验 | 添加 `quality_notes` |

这些建议本身不把“通过”改成“需重录”。例如侧举误差 $18^\circ$ 可以通过 $25^\circ$ 硬阈值，但带有 $15^\circ$ 建议提醒。应用全部合格结果时，若存在提醒，窗口还要求用户确认，默认取消。有提醒不等于算法已证明结果不可用，也不等于建议可以忽略。

“穿戴＋相机一体标定”复用拟合器，但将躯干 $\ge5^\circ$、前臂侧举 $\ge15^\circ$ 等建议也追加到外层报告的 `warnings` 中，**不因此改变 `status`**；穿戴条目的 `accepted` 按 `status=="通过"` 判定。因此不能跨流程简单采用“`warnings` 非空就是失败”的规则，也不能把穿戴合格当作相机条目一定合格。[一体标定报告](../OptUSBProj-ubuntu/optusbproj_ubuntu/combined_calibration.py)

总结：**“通过”只表示该流程实际执行的检查满足当前动作假设和阈值，不保证动作完美、未执行的检查合格，或解剖学转角绝对正确。** 骨段方向拟合不能独立验证绕长轴的解剖学扭转，仍须回放和专项动作复核。

## 七、右上臂、右前臂如何共同驱动肘关节

两台设备分别标定，得到 $B_U,B_F$。运行时使用各算法原始参考系下的实时矩阵 $\bar R_U(t),\bar R_F(t)$，以及同一中立窗口采集的原始平均姿态 $\bar N_U,\bar N_F$，分别计算

$$\begin{aligned}
Q_U(t)&=B_U\bar N_U^\mathsf T\bar R_U(t)B_U^\mathsf T,\\
Q_F(t)&=B_F\bar N_F^\mathsf T\bar R_F(t)B_F^\mathsf T.
\end{aligned}$$

拟合保存的中立矩阵是 $R_{i,0}=A_i\bar N_i$；应用时 `neutral_matrices_by_mode()` 对同一静止窗口的各原始姿态流独立求平均，`apply_calibration()` 以这些 `paired_neutrals` 设置运行中立位。因此，即使拟合用过重力换基，运行也不会错误地计算 $(A_i\bar N_i)^\mathsf T\bar R_i(t)$。这是代码中“拟合矩阵”和“按模式配对的中立矩阵”分别保存的原因。

这两个 $Q$ 表示各骨段在**共同模型坐标系**中的目标方向，而不是直接的关节角。右前臂的父部位是右上臂，因此模型把前臂的绝对方向换算为相对上臂的局部旋转：

$$\boxed{Q_{\rm elbow}=Q_U^\mathsf T Q_F.}$$

例如整条右臂保持伸直一起前举，理想情况下 $Q_U\approx Q_F$，故 $Q_{\rm elbow}\approx I$：肩在动，肘基本不弯。若右上臂保持不动、前臂单独屈肘，则 $Q_U\approx I$ 而 $Q_F\ne I$，相对肘旋转出现。躯干与上臂之间同理，使用 $Q_{\rm torso}^\mathsf TQ_U$ 计算肩部局部旋转。模型通过固定骨长和父子层级确定位置，不对 WT901 的加速度做位置积分。[层级定义](../OptUSBProj-ubuntu/optusbproj_ubuntu/skeleton.py)、[矩阵组合](../OptUSBProj-ubuntu/optusbproj_ubuntu/pose.py)

此处有两点工程前提：

- 两台设备方向须能被解释到**一致的模型航向**。分别做了 GUI 归零，不等于设备融合算法的航向长期同步；6 轴航向漂移、9 轴磁干扰都可能让静止的肘看起来缓慢旋转。
- 初始姿势应尽量与模型中立姿势一致。若真实肘在首个窗口已弯曲，程序仍把那时的上臂、前臂方向当作模型中立基准；它不会自动恢复真实的解剖学零位。

## 八、应用、重新归零和双臂联合标定

应用单部位结果时，程序核对设备 ID 与当前部位分配一致，保存本次 $B$，并用同一窗口的各模式原始中立矩阵设置 `mode_neutral`。另在 `calibration_references` 中保留本轮标准标定的 $B$、各模式标准中立姿态、设备 ID 和参考系 epoch，供自然掌面等功能核验。

之后点击 GUI 归零，会把各已有姿态流的**运行中立位**更新到当前姿态，**不改变已标定的 $B$，也不覆盖保存的标准标定参考**；若手部自然掌面显示已启用，会停止该显示覆盖。普通归零不是向设备发送清零指令，也不等于重做穿戴标定。[应用与归零代码](../OptUSBProj-ubuntu/optusbproj_ubuntu/pose.py)

切换算法时，若当前连续映射缺少目标算法的配对中立矩阵，程序要求重新标定；有配对中立矩阵只表示数学链路可切换，不保证不同估计器的误差完全相同。更换设备、改手动轴映射、设备安装/融合参数改变或绑带相对皮肤滑动后，应重新验证或重做标定。现在已有显式保存／历史恢复流程，但不是无条件自动加载旧矩阵；历史参考系不能直接充当本次积分参考系，仍需通过恢复检查。[历史恢复实现](../OptUSBProj-ubuntu/optusbproj_ubuntu/calibration_reuse.py)

“双臂联合标定”只是把可在线参与的左/右上臂、前臂同步采集五组窗口，随后分别运行同一个 `fit_guided_arm()`。它可以一次预览和应用所有合格部位，但**不会联合求一个能保证肩—肘解剖学一致的全局最优解**；每个部位仍有独立的航向拟合和质量判定。因此若想研究真实肘屈曲角与模型肘角的精度，还需要专门的屈肘实验或视觉/外部参考，而不能仅凭“联合标定通过”。[操作流程与记录位置](../OptUSBProj-ubuntu/docs/WEARING_CALIBRATION.md)

## 九、最简计算摘要

```python
# 数学示意，不是可直接运行的项目 API。
# 每个部位分别执行；bar_R 是选中算法的原始矩阵。
# 陀螺模式用固定 A 做重力对齐；厂商模式 A = I。
R0 = A @ bar_N
R_forward = A @ bar_R_forward
R_side = A @ bar_R_side
bone_in_sensor = R0.T @ np.array([0., 0., -1.])
front_observed = R_forward @ bone_in_sensor
side_observed = R_side @ bone_in_sensor
H = estimate_z_heading(front_observed, side_observed, manual_prior)
B = H @ R0

# 运行时：先在共同模型坐标系下得到两段姿态，再取相对旋转。
Q_upper = B_upper @ (bar_N_upper.T @ bar_R_upper_now) @ B_upper.T
Q_forearm = B_forearm @ (bar_N_forearm.T @ bar_R_forearm_now) @ B_forearm.T
Q_elbow = Q_upper.T @ Q_forearm
```

`estimate_z_heading` 这一行表示上文的 `atan2` 公式，而非项目中真实存在的同名函数。不要把这份骨段指向标定解读为“已完整识别肘关节所有自由度”；它目前是可验证、可回放的穿戴轴映射基线。
