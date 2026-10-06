# URDF 变换链与空间、体系 PoE 正运动学的等价证明

本文使用列向量、右手坐标系及旋量顺序 $[\omega;v]$。以旋转关节链为主，移动关节可同样处理。公式统一使用 Markdown 数学定界符，以便 VS Code 预览。

主要结论：**URDF 链式 FK 与 PoE 不是两种不同的机器人运动，而是同一运动学映射的等价表达。固定几何变换没有被取消，而是编码进零位构型和螺旋轴。**

## 1. 参考系和 URDF 链式表达

设基系为 $L_0=W$，第 $i$ 个关节的父、子连杆分别为 $L_{i-1},L_i$，关节零角参考系为 $J_i$。

$$
O_i={}^{L_{i-1}}T_{J_i},\qquad
E_i(q_i)={}^{J_i}T_{L_i}(q_i).
$$

$O_i$ 固定，来自 URDF origin；$q_i=0$ 时 $J_i$ 与 $L_i$ 重合。对旋转关节，单位轴 $a_i$ 在 $J_i$ 中表达：

$$
\mathcal A_i=\begin{bmatrix}a_i\\0\end{bmatrix},\qquad
[\mathcal A_i]^\wedge=
\begin{bmatrix}[a_i]_\times&0\\0&0\end{bmatrix},
$$

$$
E_i(q_i)=\exp([\mathcal A_i]^\wedge q_i)
=\begin{bmatrix}\exp([a_i]_\times q_i)&0\\0&1\end{bmatrix}.
$$

局部线速度分量为零，因为旋转轴通过关节系原点。移动关节可取 $\mathcal A_i=[0;a_i]$。

因此

$$
\boxed{T_k(q)={}^{W}T_{L_k}(q)=O_1E_1\cdots O_kE_k}.
$$

右乘来自参考系的匹配。不能仅凭某个矩阵位于右边，就将整个公式称为体系 PoE；这里每个 $E_i$ 原本分别在不同关节参考系中定义。

固定关节可以保留在链条中、令其运动为单位矩阵，也可先合并到相邻固定变换。以下为简洁，假设已选定串联路径并处理固定段。

## 2. SE(3) 共轭与伴随作用

### 2.1 旋量与帽映射

$$
\mathcal V=\begin{bmatrix}\omega\\v\end{bmatrix},\qquad
[\mathcal V]^\wedge=
\begin{bmatrix}[\omega]_\times&v\\0&0\end{bmatrix}\in\mathfrak{se}(3).
$$

设 $G=[R,p;0,1]\in SE(3)$。直接矩阵相乘给出

$$
G[\mathcal V]^\wedge G^{-1}
=\begin{bmatrix}
R[\omega]_\times R^\mathsf T&-R[\omega]_\times R^\mathsf T p+Rv\\0&0
\end{bmatrix}.
$$

由 $R[\omega]_\times R^T=[R\omega]_\times$，平移列为 $p\times R\omega+Rv$。于是

$$
\boxed{G[\mathcal V]^\wedge G^{-1}
=[\operatorname{Ad}_G\mathcal V]^\wedge},
$$

$$
\boxed{\operatorname{Ad}_G=
\begin{bmatrix}R&0\\[p]_\times R&R\end{bmatrix}}.
$$

这不仅转换轴方向，还转换轴线位置带来的线速度部分。只有旋转、无平移时，退化为两个分量各自乘 $R$；SO(3) 的伴随作用是其旋转部分。

### 2.2 指数与共轭交换

对任意可逆 $G$ 与矩阵 $X$，由指数幂级数：

$$
G e^X G^{-1}
=\sum_{m=0}^{\infty}\frac{GX^mG^{-1}}{m!}
=\sum_{m=0}^{\infty}\frac{(GXG^{-1})^m}{m!}
=e^{GXG^{-1}}.
$$

所以

$$
\boxed{G\exp([\mathcal V]^\wedge q)G^{-1}
=\exp([\operatorname{Ad}_G\mathcal V]^\wedge q)}.
$$

这就是从局部运动指数转换到其他参考基底的核心等价关系。

## 3. 从 URDF 链推导空间 PoE

### 3.1 定义零位累计变换

$$
G_i=O_1\cdots O_i,\qquad G_0=I,\qquad M_k=G_k.
$$

$G_i$ 是关节 $J_i$ 在机器人全零位时相对于基系的变换。定义固定空间螺旋轴

$$
\boxed{\mathcal S_i=\operatorname{Ad}_{G_i}\mathcal A_i}.
$$

这里的“固定”是零位定义固定，不是说运动时关节轴线在基系中不变。

### 3.2 插入单位矩阵并逐项抵消

由 $G_{i-1}^{-1}G_i=O_i$，有

$$
\begin{aligned}
&(G_1E_1G_1^{-1})(G_2E_2G_2^{-1})\cdots(G_kE_kG_k^{-1})G_k\\
&=O_1E_1(G_1^{-1}G_2)E_2\cdots(G_{k-1}^{-1}G_k)E_k\\
&=O_1E_1O_2E_2\cdots O_kE_k.
\end{aligned}
$$

利用共轭指数恒等式，得到

$$
\boxed{T_k(q)=
\exp([\mathcal S_1]^\wedge q_1)\cdots
\exp([\mathcal S_k]^\wedge q_k)M_k}.
$$

这是严格恒等式，不是小角度近似。它对任意有效 $q$ 成立，前提是两边使用相同的关节轴、固定几何和零位定义。

### 3.3 螺旋轴中的几何意义

若 $G_i=[R_i^0,p_i^0;0,1]$，旋转关节有

$$
\mathcal S_i=
\begin{bmatrix}R_i^0a_i\\p_i^0\times R_i^0a_i\end{bmatrix}
=\begin{bmatrix}\omega_i\\-\omega_i\times p_i^0\end{bmatrix}.
$$

原来各 $O_i$ 的方向和位置已经编码进 $\mathcal S_i$；所有固定零位几何还共同决定 $M_k$。因此，不能去掉全部 $O_i$ 后只用原始局部 $a_i$ 连乘。

## 4. 从空间 PoE 推导体系 PoE

指定末端 $L_k$ 后，定义

$$
\boxed{\mathcal B_i^{(k)}=
\operatorname{Ad}_{M_k^{-1}}\mathcal S_i},\qquad i\le k.
$$

它们是同一组关节螺旋轴在**零位末端 $L_k$ 系**中的表达；不是逐项重新表达在运动后的末端系。

利用

$$
\exp([\mathcal S_i]^\wedge q_i)
=M_k\exp([\mathcal B_i^{(k)}]^\wedge q_i)M_k^{-1},
$$

将各因子代回空间 PoE，相邻 $M_k^{-1}M_k$ 抵消，得到

$$
\boxed{T_k(q)=M_k
\exp([\mathcal B_1^{(k)}]^\wedge q_1)\cdots
\exp([\mathcal B_k^{(k)}]^\wedge q_k)}.
$$

关节顺序仍为 $1,\ldots,k$，没有反转。完整等价关系为

$$
\boxed{
\prod_{i=1}^k O_i\exp([\mathcal A_i]^\wedge q_i)
=\left(\prod_{i=1}^k\exp([\mathcal S_i]^\wedge q_i)\right)M_k
=M_k\left(\prod_{i=1}^k\exp([\mathcal B_i^{(k)}]^\wedge q_i)\right)}.
$$

乘积按索引递增从左到右排列，不允许因为“都是旋转”而交换因子。

### 4.1 为什么 L4 和 L6 的体系旋量不同？

空间旋量 $\mathcal S_i$ 对同一基系与关节链可复用，但体系旋量取决于所选末端：

$$
\mathcal B_i^{(4)}=\operatorname{Ad}_{M_4^{-1}}\mathcal S_i,
\qquad
\mathcal B_i^{(6)}=\operatorname{Ad}_{M_6^{-1}}\mathcal S_i.
$$

因此不能将 L6 的体系轴不加转换地当作 L4 的体系轴。求多个 link 时，空间 PoE 的截断前缀通常更方便。

## 5. L4 肘点：为什么完整位姿有四项，点位置只有前三项？

完整位姿为

$$
T_4=M_4\prod_{i=1}^4\exp([\mathcal B_i^{(4)}]^\wedge q_i).
$$

由于 $M_4=G_4$，

$$
\mathcal B_4^{(4)}
=\operatorname{Ad}_{G_4^{-1}}\operatorname{Ad}_{G_4}\mathcal A_4
=\mathcal A_4=\begin{bmatrix}a_4\\0\end{bmatrix}.
$$

令齐次原点 $\bar o=[0,0,0,1]^T$，则

$$
\exp([\mathcal B_4^{(4)}]^\wedge q_4)\bar o=\bar o.
$$

所以

$$
\boxed{\begin{bmatrix}p_E(q)\\1\end{bmatrix}
=M_4\left(\prod_{i=1}^{3}\exp([\mathcal B_i^{(4)}]^\wedge q_i)\right)\bar o}.
$$

这只是说明第 4 关节不改变 L4 原点。它仍改变 L4 的方向，以及 L4 上非轴线点和下游点的位置。L6 原点同理不受第 6 关节自身影响。

## 6. 固定轴定义与运动中的雅可比轴

定义空间前缀

$$
P_{i-1}(q)=\prod_{j=1}^{i-1}\exp([\mathcal S_j]^\wedge q_j).
$$

运动时第 $i$ 关节贡献的空间速度轴为

$$
\boxed{J_{s,i}(q)=\operatorname{Ad}_{P_{i-1}(q)}\mathcal S_i}.
$$

理由是：该关节的零位轴受其上游运动搬运；对 PoE 求导并计算 $\dot T T^{-1}$ 得到

$$
[\mathcal V_s]^\wedge=\dot T T^{-1},\qquad
\mathcal V_s=J_s(q)\dot q.
$$

体系速度为

$$
[\mathcal V_b]^\wedge=T^{-1}\dot T
=T^{-1}[\mathcal V_s]^\wedge T,
$$

因此

$$
\boxed{J_b(q)=\operatorname{Ad}_{T(q)^{-1}}J_s(q)}.
$$

区别：$\mathcal B_i^{(k)}$ 是零位构造的固定轴；$J_{b,i}(q)$ 是当前配置的速度雅可比列。二者不能一般性地混同。

## 7. 一个带偏置的单关节例子

设零位关节位置为 $r=[d,0,0]^T$，轴为 $a=[0,0,1]^T$，且

$$
O=M=\begin{bmatrix}I&r\\0&1\end{bmatrix}.
$$

局部旋量 $\mathcal A=[a;0]$，空间旋量

$$
\mathcal S=\begin{bmatrix}0\\0\\1\\0\\-d\\0\end{bmatrix}.
$$

记 $R=R_z(q)$，绕空间轴线旋转的指数为

$$
\exp([\mathcal S]^\wedge q)
=\begin{bmatrix}R&(I-R)r\\0&1\end{bmatrix}.
$$

于是

$$
\exp([\mathcal S]^\wedge q)M
=\begin{bmatrix}R&r\\0&1\end{bmatrix}
=O\exp([\mathcal A]^\wedge q).
$$

局部写法的运动指数没有平移，空间写法的指数通常有平移；但最终位姿完全一样。平移差异是参考基底转换导致的，不是增加了一个移动关节。

## 8. 与 DH 的关系、适用范围和常见误解

- DH 使用特定的连杆系安排与四参数构造相邻变换；URDF 使用一般的固定 origin 加局部轴。二者都可以描述串联链，但参数并非逐字段一一对应。
- 将 URDF 改写成 PoE 不需要先转成 DH。
- 右乘不自动意味着标准体系 PoE：关键是是否所有固定螺旋轴在同一个零位末端基底中表达。
- 空间 PoE 的轴是零位固定参数，但实际空间关节轴随上游运动变化，由雅可比的伴随变换体现。
- 指数积没有取消平移几何；旋量线分量与零位构型共同保存几何。
- 若实际工具系相对 L6 有固定变换 $X$，则 $T_{TCP}=T_{L6}X$、$M_{TCP}=M_6X$；空间轴不变，体系轴应重新用 $M_{TCP}$ 转换。
- 证明是正运动学恒等，不保证逆运动学目标可达、唯一或数值迭代全局收敛。

## 9. 与当前代码对应

| 数学对象/公式 | 当前实现 |
| --- | --- |
| $O_i,a_i$ | `ArmKinematics.origins`、`axes` |
| URDF 链式 FK | `ArmKinematics.frames()` |
| $G_i,M_k$ | `PoEArmKinematics.homes` |
| 空间轴 $\mathcal S_i$ | `PoEArmKinematics.S` 的列 |
| L6/R56 零位 $M$ | `PoEArmKinematics.M` |
| L6/R56 体系轴 $\mathcal B_i$ | `PoEArmKinematics.B` 的列 |
| 空间/体系 FK | `fk_space()`、`fk_body()` |
| 各 link 的截断空间 PoE | `poe_frames()` |
| 雅可比 | `space_jacobian()`、`body_jacobian()` |

源码：[URDF 链式实现](../OptUSBProj-ubuntu/optusbproj_ubuntu/x1_retarget.py)、[PoE 与李群运算实现](../OptUSBProj-ubuntu/optusbproj_ubuntu/x1_kinematics.py)。

更多背景：[URDF 可获取元数据](URDF_GEOMETRY_METADATA.md)、[人体姿态到机器人关节角](HUMAN_TO_X1_RETARGETING_MATH.md)。
