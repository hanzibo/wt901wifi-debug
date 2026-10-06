# 自然掌面参考：手动与视觉估计 C₀ 的数学推导

本文对应当前左手原型，整理标准穿戴映射 $B$、标准参考 $N$、自然参考 $R_c$、重表达映射 $B_c$、自然掌面初始姿态 $C_0$ 的关系，并推导 RGB-D 视觉反推 $C_0$ 的方法。

**范围：这是自然掌面显示参考，不是手部长期航向纠偏，也不是机器人控制的新标定。** 现有 X1 计算显式排除自然显示偏置。人体自然朝向在本文中指自然下垂、掌心向内偏后；标准朝向指下垂、手背向前。

## 1. 坐标与旋转约定

使用列向量、右手坐标系，所有旋转属于

$$
SO(3)=\{R\in\mathbb R^{3\times3}:R^{\mathsf T}R=I,\ \det R=1\}.
$$

${}^A R_B$ 表示 B 系三根单位轴在 A 系中的坐标。因此

$$
v_A={}^A R_Bv_B,
\qquad {}^A R_C={}^A R_B{}^B R_C,
\qquad R^{-1}=R^{\mathsf T}.
$$

本文省略上标时，必须按下表理解，不能只凭矩阵相乘形状推断物理意义。

| 符号 | 定义 |
| --- | --- |
| $W$ | 当前姿态算法的固定参考系；厂商融合参考系或软件估计器参考系 |
| $S(t)$ | 随设备运动的本体系 |
| $M$ | 人体模型固定参考方向：X 前、Y 左、Z 上 |
| $G(t)$ | 随手掌运动的局部系；左手模型 +X 指手背、+Z 沿长轴指向前臂，+Y 由右手系确定 |
| $\mathcal C$ | 相机光学系：X 右、Y 下、Z 前 |
| $R(t)$ | ${}^W R_{S(t)}$，当前设备姿态 |
| $N$ | 标准姿态时的设备姿态 $R(t_s)$，与 $B$ 配套保存 |
| $B$ | 标准参考设备轴到模型方向的映射；可为手动有符号轴排列或连续穿戴标定矩阵 |
| $R_c$ | 自然参考采样时设备姿态 $R(t_c)$，实际以静止窗口均值近似 |
| $C(t)$ | 自然显示下的手掌绝对姿态 ${}^M R_{G(t)}$ |
| $C_0$ | 自然参考时希望显示的手掌姿态 $C(t_c)$ |
| $K$ | 相机方向坐标到模型方向坐标的固定变换 ${}^M R_{\mathcal C}$ 的估计 |

标准手掌局部轴在模型标准状态与模型方向重合，故规定 $C(t_s)=I$。这是模型的坐标约定，不代表真实自然下垂的掌面必然具有单位姿态。

设备本体系与手掌局部系也不是同一个系：绑带和安装方向会让两者存在固定旋转。

## 2. 从标准映射开始

### 2.1 标准设备相对姿态

标准姿态时设备为 $N$，当前为 $R(t)$。以标准时刻的设备轴作为固定表达基底，有

$$
\Delta_s(t)=N^{\mathsf T}R(t).
$$

其列是当前设备轴在标准时刻设备系 $S(t_s)$ 中的表示。它也可被解释为从标准轴到当前轴的主动旋转，表达在 $S(t_s)$ 的坐标基底下。

换成模型坐标表达，采用共轭：

$$
\boxed{C_s(t)=B\Delta_s(t)B^{\mathsf T}
=BN^{\mathsf T}R(t)B^{\mathsf T}.}
$$

因此 $C_s(t_s)=I$。其中 $B$ 改变旋转运动的坐标表达，不是把三个欧拉角分别乘系数。

### 2.2 分解成固定左右因子

定义

$$
A=BN^{\mathsf T},\qquad F_s=B^{\mathsf T},
$$

则

$$
C_s(t)=A R(t)F_s.
$$

在该标定模型中，$A$ 是从姿态算法参考方向到模型参考方向的固定换算；$F_s$ 是标准约定下手掌局部轴到设备局部轴的固定换算。若 $B$ 有误，它们只是模型近似，不能称为测得的真实外参。

## 3. 为什么采集自然参考 R_c 还不够

自然参考姿态为 $R_c$。直接把 $N$ 换成 $R_c$、继续使用旧 $B$，会得到

$$
\widetilde C(t)=B(R_c^{\mathsf T}R(t))B^{\mathsf T}.
$$

但旧 $B$ 配套的是标准时刻的设备轴 $S(t_s)$，新相对矩阵表达在 $S(t_c)$ 下。混用它们，会改变模型中的旋转轴方向。

另外，$R_c$ 本身不告诉我们“掌心向内偏后”在模型里对应哪个角度。仅凭这份 IMU 数据和标准约定，可以算出 $C_s(t_c)$，但不能独立验证它等于真实的自然掌面姿态。原型因此显式引入手动或视觉给出的 $C_0$。

## 4. B_c 的推导：重表达而非重新穿戴标定

### 4.1 参考变化后的轴映射

模型参考方向换算 $A=BN^{\mathsf T}$ 保持不变。自然参考时设备轴在模型参考方向下的映射为

$$
\boxed{B_c=A R_c=BN^{\mathsf T}R_c.}
$$

所以 $B_c$ 与 $B$ 的区别是：

- $B$ 与标准设备参考 $N$ 配套；
- $B_c$ 与自然设备参考 $R_c$ 配套；
- 两者共享同一固定方向换算 $A$。

由于 $B,N,R_c\in SO(3)$，其乘积 $B_c$ 也属于 $SO(3)$。

**$B_c$ 不是自然手掌局部系本身的姿态 $C_0$。** 如果想把自然参考设备轴表达在自然手掌局部系里，矩阵反而是

$$
{}^{G(t_c)}R_{S(t_c)}=C_0^{\mathsf T}B_c
$$

（这是本显示模型中采用的安装关系）。

### 4.2 自然参考到当前的运动

在自然参考设备系中：

$$
\Delta_c(t)=R_c^{\mathsf T}R(t).
$$

换成模型固定方向表示：

$$
\boxed{D(t)=B_c\Delta_c(t)B_c^{\mathsf T}.}
$$

代入 $B_c$ 并消去相邻逆矩阵：

$$
\begin{aligned}
D(t)
&=BN^{\mathsf T}R_c(R_c^{\mathsf T}R(t))R_c^{\mathsf T}NB^{\mathsf T}\\
&=BN^{\mathsf T}R(t)R_c^{\mathsf T}NB^{\mathsf T}\\
&=C_s(t)C_s(t_c)^{\mathsf T}.
\end{aligned}
$$

这证明：重表达后，运动等价于标准显示中从参考掌面到当前掌面的**空间旋转增量**，表达在固定模型方向下。

### 4.3 作用到自然初始姿态

$D(t)$ 在模型固定方向下表达，故左乘初始姿态：

$$
\boxed{C(t)=D(t)C_0
=B_c(R_c^{\mathsf T}R(t))B_c^{\mathsf T}C_0.}
$$

在 $t=t_c$ 时，$\Delta_c=I$、$D=I$，因此

$$
\boxed{C(t_c)=C_0.}
$$

反过来，若希望在手掌局部系下右乘运动，应先改变表达基底：

$$
C(t)=C_0\big(C_0^{\mathsf T}D(t)C_0\big).
$$

不能直接改成 $C_0D(t)$；一般旋转不交换。

### 4.4 与标准链路的相容性

若选择 $C_0=C_s(t_c)$，则

$$
C(t)=C_s(t)C_s(t_c)^{\mathsf T}C_s(t_c)=C_s(t).
$$

所以仅更换参考并使用一致的 $C_0$ 不会改变旧链路。如果手动或视觉得到的 $C_0$ 与 $C_s(t_c)$ 不同，显示差异来自显式初始姿态修订，不是共轭公式“自动发现”了新偏差。

## 5. 手动 C₀：绕长轴偏角

自然下垂时假定手掌长轴基本沿模型 +Z，仅估计一个掌面朝向偏角 $\alpha$。定义

$$
e_z=(0,0,1)^{\mathsf T},\qquad
C_{0,m}=\exp([e_z]_\times\alpha)=R_z(\alpha).
$$

其中

$$
C_{0,m}=\begin{bmatrix}
\cos\alpha&-\sin\alpha&0\\
\sin\alpha&\cos\alpha&0\\
0&0&1
\end{bmatrix}.
$$

角度正负按模型 +Z 的右手定则，左右手不能不加分析就使用同一正负角。GUI 输入是度，指数公式中的 $\alpha$ 按弧度计算。

手动自然显示为

$$
C_m(t)=D(t)C_{0,m}.
$$

操作顺序：先固定穿戴并建立标准 $(B,N)$；再自然下垂采 $R_c$；调节 $\alpha$ 并观察预览；确认后应用。**采 $R_c$ 是采运动参考，调 $\alpha$ 才是在给模型指定自然掌面朝向。**

该一参数模型不能恢复明显屈腕、侧偏或长轴倾斜。它也不是由 IMU 独立测出的解剖真值。

## 6. 参考系任意性与普通归零

### 6.1 固定姿态算法参考系变化不改变公式

若只是用另一套固定参考方向 $W'$ 表示同一轨迹，令 $H={} ^{W'}R_W$ 为常量，则

$$
R'(t)=HR(t),\quad N'=HN,\quad R_c'=HR_c.
$$

有

$$
(R_c')^{\mathsf T}R'(t)=R_c^{\mathsf T}R(t),
$$

$$
B_c'=B(N')^{\mathsf T}R_c'=BN^{\mathsf T}R_c=B_c.
$$

所以姿态可以来自厂商 RPY 或软件陀螺＋重力估计器；数学并不依赖某一套绝对 yaw 零点。前提是 $N,R_c,R(t)$ 属于**同一算法轨迹和参考会话**。

重启积分器、切换算法而混用旧 $N$、或发生随时间变化的漂移，都不属于上述常量 $H$ 的情形，不能据此认为旧参考仍有效。

### 6.2 理论上的再次重设采样参考

如果改用 $R_a$ 作为参考，并要求显示轨迹保持原样，则必须一起更新

$$
B_a=B_cR_c^{\mathsf T}R_a,
\qquad C_a=D(t_a)C_0.
$$

新的显示

$$
B_a(R_a^{\mathsf T}R(t))B_a^{\mathsf T}C_a=C(t)
$$

可由上式直接代入验证。只替换参考姿态、保留旧映射和旧初始姿态通常不等价。

### 6.3 当前代码的普通归零行为

当前普通“归零”不是执行上述轨迹保持重设，而是切回标准化显示，并停用自然显示。它不覆盖独立保存的标准 $(B,N)$，也不把旧自然参考解释成“最后一次归零的参考”。

“恢复已保存自然显示”恢复原来的 $(B_c,R_c,C_0)$。设备分配、安装映射、算法、积分会话或配套标准参考变化时，必须检查绑定或重新采样。

## 7. 视觉方案先建立相机到模型的方向变换 K

第一版不使用之前每个 IMU 的相机纠偏矩阵作为手掌外参，而是从自然中立窗口的肩髋信息估计身体参考方向。

相机系下的左右肩、左右髋坐标为 $p_{LS},p_{RS},p_{LH},p_{RH}$。每帧先计算

$$
z_i=\operatorname{unit}\left(\frac{p_{LS}+p_{RS}-p_{LH}-p_{RH}}2\right),
\qquad y_i=\operatorname{unit}(p_{LS}-p_{RS}).
$$

多帧平均后正交化：

$$
\begin{aligned}
z_C&=\operatorname{unit}(\textstyle\sum_i z_i),\\
\widetilde y_C&=\tfrac1n\textstyle\sum_i y_i,\\
y_C&=\operatorname{unit}\big(\widetilde y_C-z_Cz_C^{\mathsf T}\widetilde y_C\big),\\
x_C&=\operatorname{unit}(y_C\times z_C),\\
y_C&=\operatorname{unit}(z_C\times x_C).
\end{aligned}
$$

这三列是模型参考方向在相机系中的表示，故

$$
{}^{\mathcal C}R_M=[x_C\ y_C\ z_C],
\qquad \boxed{K={}^M R_{\mathcal C}=[x_C\ y_C\ z_C]^{\mathsf T}.}
$$

相机不必正对人体，但需要可靠肩髋信息和左右身份。该方法依赖站直及解剖肩线；它不是测得的物理重力真值，也不等于完整相机位姿标定。这里只处理方向，因此无需平移或人体到机器人比例。

$K$ 在本轮冻结。采样时身体朝向需与建立标准 $(B,N)$ 时的模型方向约定一致，相机也不能移动；否则两个“模型方向”的定义可能不一致，误差会混入 $C_0$。

## 8. 从 RGB-D 得到相机系下的掌面三轴

### 8.1 二维关键点与真实深度

视觉模型只提供手部21点二维像素。第一版取腕点和四个指根：

$$
\mathcal I=(0,5,9,13,17).
$$

彩色像素先去畸变得到归一化射线，再映射到已配准深度像素，读取稳健深度 $z_i$。如果相机深度表示光学 Z 距离，则

$$
p_i=z_i\begin{bmatrix}x_i^{\rm ray}\\y_i^{\rm ray}\\1\end{bmatrix}.
$$

毫米转换成米；不使用 MediaPipe 预测的 Z 或 world landmarks 来替代相机深度。彩色和配准深度的内参/畸变参数分别处理；相机系点云与关键点必须在同一光学参考系。

### 8.2 语义轴与平面法向量

定义向近端的长轴候选和向食指侧的横向候选：

$$
\ell=p_0-\frac{p_5+p_9+p_{13}+p_{17}}4,
\qquad r=p_5-p_{17}.
$$

左手手背方向候选为

$$
n_s=\frac{\ell\times r}{\|\ell\times r\|}.
$$

此符号依赖左手及当前模型轴约定，不能直接复制到右手。

在掌部内缩多边形中提取真实深度点云 $q_k$。对均值 $\mu$ 和协方差

$$
\Sigma=\frac1m\sum_k(q_k-\mu)(q_k-\mu)^{\mathsf T}
$$

做特征分解。最小特征值对应的单位特征向量是平面法向量候选 $n$。通过迭代剔除离平面过远的点减少混合深度，再用语义方向固定正负号：若 $n^{\mathsf T}n_s<0$，令 $n\leftarrow-n$。

注意：PCA 平面法向量具有正负二义性；平面内主方向在特征值接近时也不稳定。当前方法**只用 PCA 求法向量，长轴来自腕点和指根的语义关系**，没有用 PCA 独自确定全部三轴。

构造正交右手轴：

$$
\begin{aligned}
z_G^C&=\operatorname{unit}(\ell-nn^{\mathsf T}\ell),\\
y_G^C&=\operatorname{unit}(z_G^C\times n),\\
x_G^C&=\operatorname{unit}(y_G^C\times z_G^C),\\
V_j&={} ^{\mathcal C}R_{G(t_j)}=[x_G^C\ y_G^C\ z_G^C].
\end{aligned}
$$

通过单位化与叉乘，非退化时三轴正交且 $\det V_j=1$。随后转换到模型系：

$$
\boxed{C_j^{\rm vis}=K V_j.}
$$

首版要求掌心可见，以避免把手背 IMU 外壳当作掌面。观察到的是表面几何和语义轴，不是解剖骨骼坐标系真值。

## 9. 视觉反推 C₀

### 9.1 单帧逆解

在可见姿态 $t_j$，用同期纯 IMU 姿态求

$$
D_j=B_c(R_c^{\mathsf T}R_j)B_c^{\mathsf T}.
$$

自然显示关系要求

$$
C_j^{\rm vis}\approx D_jC_0.
$$

若理想无误差，由 $D_j^{-1}=D_j^{\mathsf T}$ 得

$$
\boxed{A_j=D_j^{\mathsf T}C_j^{\rm vis}=C_0.}
$$

这就是“举起手让相机看清掌面，再用 IMU 反推自然下垂时朝向”的数学依据。无需在自然下垂时看清完整手掌。

这里 $A_j$ 是每帧反推的初始姿态候选，**不是**第2节的固定方向换算 $A=BN^{\mathsf T}$。

已知完整 $D_j$ 和完整视觉三轴时，理论上一帧即可求完整 $C_0$。多帧和不同可见姿态用于减小噪声、检查一致性，而不是重新求解 $B$ 或证明视觉绝对准确。

### 9.2 多帧 SO(3) 最小二乘与 SVD 解

采用旋转矩阵弦距离目标，给定非负固定权重 $w_j$：

$$
\min_{C_0\in SO(3)}J(C_0),\qquad
J(C_0)=\sum_jw_j\|C_j^{\rm vis}-D_jC_0\|_F^2.
$$

正交左乘保持 Frobenius 范数，故

$$
J(C_0)=\sum_jw_j\|A_j-C_0\|_F^2.
$$

对旋转矩阵展开：

$$
\|A_j-C_0\|_F^2=6-2\operatorname{tr}(C_0^{\mathsf T}A_j).
$$

令 $S=\sum_jw_jA_j$，最小化等价于最大化 $\operatorname{tr}(C_0^{\mathsf T}S)$。对 $S$ 做奇异值分解

$$
S=U\Sigma V^{\mathsf T},\qquad
\Sigma=\operatorname{diag}(\sigma_1,\sigma_2,\sigma_3),\quad
\sigma_1\ge\sigma_2\ge\sigma_3\ge0.
$$

令 $Q=U^{\mathsf T}C_0V$，则 $Q$ 正交且 $\det Q=\det(UV^{\mathsf T})$。迹目标变为 $\sum_i\sigma_iQ_{ii}$。正交 Procrustes 的最大值由

$$
Q_*=\operatorname{diag}(1,1,\det(UV^{\mathsf T}))
$$

达到：若需反射修正，把负号放在最小奇异值方向，损失最小。于是

$$
\boxed{C_{0,v}=U\operatorname{diag}(1,1,\det(UV^{\mathsf T}))V^{\mathsf T}.}
$$

该结果正交且行列式为 +1，避免直接平均矩阵后离开 $SO(3)$。秩退化等情况不保证唯一解。

当前代码先等权 SVD 求初值，再按旋转距离剔除离群，最后等权重算。**固定内点集合时有上述闭式最优解；离群选择过程不保证整体鲁棒目标的全局最优。** 弦距离平均也不等同于一般的内禀测地距离最小二乘。

## 10. 对比、验证与可观测性

### 10.1 旋转距离

对 $R_1,R_2\in SO(3)$，定义

$$
d(R_1,R_2)=\|\operatorname{Log}(R_1^{\mathsf T}R_2)^\vee\|
=\arccos\left(\operatorname{clip}\left(\frac{\operatorname{tr}(R_1^{\mathsf T}R_2)-1}{2},-1,1\right)\right).
$$

范围为 $[0,\pi]$，报告显示度。实现使用 SciPy 旋转向量幅值；该式用于解释几何距离。

独立复测误差为

$$
e_j=d(D_jC_{0,v},C_j^{\rm vis}).
$$

复测帧不参与拟合，报告中位数和 P90。它检验同一套 IMU、身体对齐、视觉掌面是否相容，**不是带独立 GT 的姿态误差**。

### 10.2 手动与视觉对照

报告完整旋转差

$$
e_{mv}=d(C_{0,m},C_{0,v})
$$

以及三轴差

$$
\beta_i=\arccos\big((C_{0,m}e_i)^{\mathsf T}(C_{0,v}e_i)\big),\quad i=x,y,z.
$$

如果希望把视觉完整 $C_{0,v}$ 简化成最近的绕 Z 旋转，最小化 $\|C_{0,v}-R_z(\alpha)\|_F^2$ 等价于最大化

$$
(C_{11}+C_{22})\cos\alpha+(C_{21}-C_{12})\sin\alpha+C_{33}.
$$

所以

$$
\boxed{\alpha_* =\operatorname{atan2}(C_{21}-C_{12},C_{11}+C_{22}).}
$$

这里元素索引为数学的1起始，$C=C_{0,v}$。若 atan2 的两个参数同时为零，则偏角不唯一。还应报告 $d(R_z(\alpha_*),C_{0,v})$，说明只保留长轴偏角丢掉多少倾斜信息。

### 10.3 为什么 C₀ 不能治漂移

$C_0$ 是一个常量。如果 IMU 姿态为 $\widehat R(t)=H(t)R_{\rm true}(t)$，其中 $H(t)$ 随时间变化，则

$$
\widehat\Delta_c(t)=R_{{\rm true},c}^{\mathsf T}H(t_c)^{\mathsf T}H(t)R_{\rm true}(t).
$$

误差因子 $H(t_c)^{\mathsf T}H(t)$ 仍在运动链中，固定的 $C_0$ 无法对所有时刻消除它。视觉初始姿态估计和长期低频纠偏是两个独立问题。

即使某一可见姿态下通过 $C_0$ 拟合成功，错误的 $B$、错误的 $K$ 或漂移也可能被暂时吸收到拟合结果里；换姿态后就暴露不一致。这也是独立复测不可省略的原因。

## 11. 当前实现门限和实践限制

门限是原型验收设置，不是数学定理或相机性能保证：

- 自然参考至少12个新 IMU 抽样，姿态波动 P90 不超过3°；肩髋参考至少5帧，上方向波动 P90 不超过8°、肩方向不超过12°。
- 拟合与复测各至少10个同期可信掌面；静止窗口首0.5秒、末0.2秒排除；IMU 插值间隔超过100 ms拒绝，身体ROI关联帧差超过150 ms拒绝。
- 指根横向像素宽至少16 px；关键点5×5深度窗口至少18个有效点，深度250–4000 mm，P90−P10不超过30 mm。
- 平面至少40点、内点比例至少70%、RMS不超过6 mm；特征值按 $\lambda_1\le\lambda_2\le\lambda_3$，要求 $\lambda_2\ge10^{-5}\ \mathrm{m}^2$、$\lambda_1/\lambda_2\le0.2$；语义/平面法向夹角不超过25°。
- 反推初值离群剔除门限为 $\max(8^\circ,2.5\times\text{中位距离})$，保留至少10帧及70%。训练中位误差小于12°；复测中位小于12°、P90小于25°；自然长轴偏离模型Z小于20°。
- 返回自然参考至少12个抽样，中位旋转差不超过12°；该差不能区分人为复位误差与 IMU 漂移。

用户已做穿戴/不穿戴对照：横跨掌面的黑色松紧带会明显降低当前检测成功率，并让指根点不稳定。此观察说明输入外观及遮挡是重要工程限制，尚不是跨模型量化结论。二维指根不稳会同时影响 $\ell,n_s$、点云采样区域和最终三轴；平面拟合或平滑不能自动补出被遮挡的可靠语义点。

不能摘下设备采样后重新戴回去而声称原安装关系不变。更换固定方式后，需重建相关标准/自然参考。手部后端 Demo 只比较二维检测，不代表完成本节的三维验收。

## 12. 如何进入人体骨架渲染

前臂绝对姿态为 $P(t)$，手掌绝对姿态为 $C(t)$，则手相对于前臂的旋转为

$$
L_G(t)=P(t)^{\mathsf T}C(t).
$$

以既有腕关节偏移 $o_G$ 构成局部齐次变换：

$$
T_{FG}(t)=\begin{bmatrix}L_G(t)&o_G\\0&1\end{bmatrix},
\qquad T_{MG}(t)=T_{MF}(t)T_{FG}(t).
$$

故绝对旋转块为 $P(t)L_G(t)=C(t)$。自然参考只替换手掌旋转，不改变骨长、STL网格或腕关节偏移；手掌点随既有父级骨架运动。

这解释了为何手掌局部“相对前臂角度”与绝对掌面朝向不同，也解释了为何不能直接把手掌设备欧拉角当成局部腕关节角。

## 13. 代码及数据索引

以下路径相对于仓库根目录 `/home/hzb/Work/WT901`：

| 数学/状态 | 当前实现 |
| --- | --- |
| 标准 $(B,N)$ 保存，$B_c=BN^{\mathsf T}R_c$ | `OptUSBProj-ubuntu/optusbproj_ubuntu/pose.py`：`apply_calibration`、`capture_manual_standard_hand`、`make_natural_hand_reference` |
| $C(t)=B_cR_c^{\mathsf T}R(t)B_c^{\mathsf T}C_0$ | `OptUSBProj-ubuntu/optusbproj_ubuntu/natural_hand.py`：`rotation` |
| 手动参考采样、绕Z偏角、预览/应用/恢复 | `OptUSBProj-ubuntu/optusbproj_ubuntu/natural_hand_gui.py` |
| 肩髋构建 $K$ | `OptUSBProj-ubuntu/optusbproj_ubuntu/visual_wearing.py`：`body_frame` |
| 真实深度、平面和掌面三轴、SVD反推 | `OptUSBProj-ubuntu/optusbproj_ubuntu/palm_geometry.py`：`lift_palm`、`frame_from_points`、`estimate_c0` |
| 同期 IMU、引导窗口、独立复测、显式应用 | `OptUSBProj-ubuntu/optusbproj_ubuntu/palm_visual_gui.py` |
| 手部二维检测、失败诊断图 | `OptUSBProj-ubuntu/optusbproj_ubuntu/palm_worker.py` |
| 手相对前臂与骨架变换 | `OptUSBProj-ubuntu/optusbproj_ubuntu/pose.py`：`transforms`、`rotation_for` |
| X1排除自然显示偏置 | `OptUSBProj-ubuntu/optusbproj_ubuntu/x1_live.py`、`x1_retarget.py`：`transforms(include_natural=False)` |

自然参考字段：`mapping_basis`=$B$、`standard_neutral`=$N$、`anchor`=$R_c$、`basis`=$B_c$、`neutral_pose`=$C_0$。字段 `basis` 在自然参考中不是原穿戴 $B$，这是读记录时最容易混淆的一点。

视觉报告位于 `OptUSBProj-ubuntu/calibrations/palm_visual/日期-编号/report.json`；其中 `K_model_from_camera`=$K$，训练/复测记录含 $D_j,C_j^{\rm vis}$，并分别保存 `C0_manual` 与 `C0_visual`。失败报告不一定含已拟合的 $C_0$。报告及少量诊断图不是完整RGB-D原始录像。

相关操作说明：[手动参考](../OptUSBProj-ubuntu/docs/NATURAL_PALM_REFERENCE.md)、[视觉参考](../OptUSBProj-ubuntu/docs/VISUAL_PALM_C0.md)、[二维后端Demo](../OptUSBProj-ubuntu/docs/HAND_BACKEND_DEMO.md)。

## 14. 最终公式汇总

$$
\boxed{
\begin{aligned}
A&=BN^{\mathsf T},\\
B_c&=AR_c,\\
D(t)&=B_cR_c^{\mathsf T}R(t)B_c^{\mathsf T},\\
C(t)&=D(t)C_0,\\
C_{0,m}&=R_z(\alpha),\\
C_j^{\rm vis}&=KV_j,\\
A_j&=D_j^{\mathsf T}C_j^{\rm vis},\\
C_{0,v}&=U\operatorname{diag}(1,1,\det(UV^{\mathsf T}))V^{\mathsf T},
\quad \sum_{j\in\mathrm{inliers}}A_j=U\Sigma V^{\mathsf T}.
\end{aligned}}
$$

先有有效标准 $(B,N)$，再采自然 $R_c$ 并重表达 $B_c$；手动或视觉给出 $C_0$；最终在模型固定方向下用 $D(t)$ 左乘它。**重表达、初始姿态修订、长期纠偏三件事必须分开理解。**
