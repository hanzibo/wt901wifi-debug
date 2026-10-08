---
title: 陀螺＋重力：纠正倾斜，留下航向
subtitle: 第一讲 · 从一个向量到姿态约束 · 可播放的三维数学演示
template: doc
theme: shadcn
mode: dark
lang: zh
style: off
---

这不是把设备朝向直接变成真值。陀螺先传播姿态，加计再约束向上方向；绕向上方向的转动仍无法由重力确定。

## A 先认识三个量

| 符号 | 含义 | 如何得到 |
| --- | --- | --- |
| $W$ | 软件建立的固定参考系 | 初始与设备本体系重合；$Z$ 不一定竖直 |
| $S(t)$、$R(t)$ | 随设备运动的本体系；其三根轴在 $W$ 中的表示 | $R$ 把本体坐标变为 $W$ 坐标：$\mathbf x_W=R\mathbf x_S$ |
| $\mathbf a_S(t)$ | 竖直向上在本体系中的单位坐标 | 静止或低动态时，把加计比力归一化 |
| $\mathbf k_W$ | 向上方向在 $W$ 中的固定坐标 | 在静止参考时刻，$\mathbf k_W=R_0\mathbf a_0$ |

这里用“向上”避免混淆：地球重力指向下，但静止加计比力指向上。动画的真值仅是教学设定，不是设备额外读出的数据。

## B 播放一次纠倾斜过程

```html
<style>
.tg-math-block{display:block;text-align:center;overflow-x:auto;padding:14px 6px;margin:12px 0}
.tg-math-inline{display:inline-block;max-width:100%;vertical-align:middle}
.tg-math-block math{font-size:1.22em}.tg-math-inline math{font-size:1.08em}
#tgEquation{overflow-x:auto}#tgEquation math{font-size:1.18em}
@media print{.tg-math-block{break-inside:avoid}}
</style>
<section id="tiltLesson" style="color:#dbeafe;background:#0c1628;border:1px solid #334155;border-radius:16px;padding:18px;font:16px system-ui;">
 <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">
  <button id="tgPlay" type="button">▶ 播放</button><button id="tgPrev" type="button">上一步</button><button id="tgNext" type="button">下一步</button><button id="tgReset" type="button">重置</button>
  <label>进度 <input id="tgTime" aria-label="动画进度" type="range" min="0" max="5.999" step="0.001" value="0"></label>
 </div>
 <div id="tgSteps" style="display:flex;gap:6px;flex-wrap:wrap;margin:14px 0"></div>
 <h3 id="tgTitle" style="color:#fff;margin:10px 0"></h3><p id="tgExplain" style="min-height:3em;line-height:1.7"></p>
 <div style="display:flex;gap:20px;flex-wrap:wrap;margin:12px 0">
  <label>教学模拟：纯倾斜误差 <input id="tgTilt" aria-label="纯倾斜误差" type="range" min="0" max="65" value="28"><output id="tgTiltOut"></output></label>
  <label>人为注入绕 k_W 的误差 <input id="tgYaw" aria-label="航向误差" type="range" min="-120" max="120" value="42"><output id="tgYawOut"></output></label>
 </div>
 <p style="font-size:14px;color:#a9bad2">滑杆只为构造带误差的教学估计，不是实际滤波器需要注入的量。纯倾斜轴 b 垂直于本体中的向上方向 v；因此倾斜滑杆就是初始重力夹角。完整纠倾斜后，只留下航向滑杆指定的误差。</p>
 <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:12px 0">
  <button id="tgViewIso" type="button">立体视角</button><button id="tgViewFront" type="button">正视（沿 −Y 看）</button><button id="tgViewSide" type="button">侧视（沿 −X 看）</button><button id="tgViewTop" type="button">俯视（沿 −Z 看）</button><button id="tgViewReset" type="button">复位视角</button>
  <label><input id="tgLinkViews" type="checkbox" checked> 联动两个视角</label><label><input id="tgPerspective" type="checkbox" checked> 透视投影</label>
 </div>
 <p style="font-size:14px;color:#a9bad2">在任一视图内拖动旋转视角，滚轮缩放；触屏支持单指旋转、双指缩放。双击视图复位。视角操作不改变设备姿态或动画进度。标准视角相对于各图的 W / S 坐标轴。</p>
 <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:12px">
  <div><strong>固定参考系 W：看设备姿态</strong><canvas id="tgWorld" tabindex="0" aria-label="原参考系三维视图，拖动旋转，滚轮缩放，方向键旋转，加减键缩放" style="width:100%;height:420px;display:block;touch-action:none;cursor:grab;border:1px solid #334155;border-radius:10px"></canvas></div>
  <div><strong>本体坐标 S：比较两个向量的坐标</strong><canvas id="tgBody" tabindex="0" aria-label="本体系三维视图，拖动旋转，滚轮缩放，方向键旋转，加减键缩放" style="width:100%;height:420px;display:block;touch-action:none;cursor:grab;border:1px solid #334155;border-radius:10px"></canvas></div>
 </div>
 <p style="font-size:14px;color:#a9bad2">轴色：X 红、Y 绿、Z 蓝。白色：向上；橙色：预测/当前估计；青色：教学真值。左图是 W 中的姿态；右图是统一本体轴坐标中的向量比较，不是第二台设备。</p>
 <div id="tgEquation" style="background:#16243a;padding:15px;border-radius:10px;font-size:18px;line-height:1.8;overflow-wrap:anywhere"></div>
 <div id="tgValues" style="white-space:pre-wrap;font:14px/1.7 ui-monospace,monospace;padding:12px 0"></div>
 <details><summary>查看当前矩阵与数学检查</summary><pre id="tgMatrices" style="white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px"></pre></details>
 <p style="font-size:14px;color:#a9bad2">示例数值由 SO(3) 指数映射计算，不是实测精度。W 的 Z 故意不与向上方向重合。公式由 LaTeX 预排版为原生 MathML，无需联网加载字体或公式库，断网也能播放。</p>
</section>
<script>
(()=>{
 const $=id=>document.getElementById(id), root=$('tiltLesson');
 const I=[[1,0,0],[0,1,0],[0,0,1]], rad=Math.PI/180;
 const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0), cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
 const norm=a=>Math.hypot(...a), scale=(a,s)=>a.map(x=>x*s), unit=a=>scale(a,1/norm(a));
 const T=A=>A[0].map((_,j)=>A.map(r=>r[j])), mv=(A,v)=>A.map(r=>dot(r,v));
 const mm=(A,B)=>A.map(r=>T(B).map(c=>dot(r,c)));
 const exp=v=>{let t=norm(v);if(t<1e-12)return I.map(r=>r.slice());let n=scale(v,1/t),c=Math.cos(t),s=Math.sin(t),K=[[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]];return I.map((r,i)=>r.map((x,j)=>c*x+(1-c)*n[i]*n[j]+s*K[i][j]));};
 const angle=(u,v)=>Math.atan2(norm(cross(u,v)),dot(u,v))/rad;
 const sub=(a,b)=>a.map((x,i)=>x-b[i]);
 // The build step pre-renders these TeX formulas; no runtime library is needed.
 const equationTex=[
  String.raw`\mathbf x_W=R_0\mathbf x_S\quad\Longrightarrow\quad\mathbf k_W=R_0\mathbf a_0`,
  String.raw`R_{\mathrm{pred},n}=R_{n-1}\operatorname{Exp}(\bar{\boldsymbol\omega}_S\Delta t)`,
  String.raw`\begin{aligned}\mathbf u&=R_{\mathrm{pred}}^{\mathsf T}\mathbf k_W,\quad\mathbf v=\mathbf a_S\\\mathbf e&=\mathbf v\times\mathbf u,\quad\mathbf n=\frac{\mathbf e}{\lVert\mathbf e\rVert}\\\theta&=\operatorname{atan2}(\lVert\mathbf e\rVert,\mathbf v^{\mathsf T}\mathbf u)\end{aligned}`,
  String.raw`\begin{aligned}D&=\operatorname{Exp}(-\theta\mathbf n),\quad D\mathbf u=\mathbf v\\\Delta R&=D^{\mathsf T}=\operatorname{Exp}(\theta\mathbf n)\\R_c&=R_{\mathrm{pred}}\Delta R\end{aligned}`,
  String.raw`R_c^{\mathsf T}\mathbf k_W=R_{\mathrm{true}}^{\mathsf T}\mathbf k_W=\mathbf a_S`,
  String.raw`\begin{aligned}H(\mathbf k_W,\psi)&=\operatorname{Exp}(\psi\mathbf k_W)\\(HR_c)^{\mathsf T}\mathbf k_W&=R_c^{\mathsf T}\mathbf k_W\end{aligned}`
 ];
 const equationMathML = /* @precompute-math */ [];
 const k=unit([.35,-.25,.90]),R0=exp([.32,-.20,.18]),Rt=mm(exp(scale(k,.30)),mm(R0,exp([.65,.25,-.12]))),a0=mv(T(R0),k),v=mv(T(Rt),k);
 // Project a seed axis onto the plane normal to the true body up direction.
 // This keeps the synthetic tilt separate from the synthetic heading error.
 function pureTiltAxis(up){const seed=Math.abs(up[0])<.9?[1,0,0]:[0,1,0];return unit(sub(seed,scale(up,dot(seed,up))));}
 function signedHeading(E){return Math.atan2(dot([E[2][1]-E[1][2],E[0][2]-E[2][0],E[1][0]-E[0][1]],k)/2,(E[0][0]+E[1][1]+E[2][2]-1)/2)/rad;}
 const titles=['① t₀：静止建立重力参考','② t₁：陀螺传播到新姿态','③ 预测重力，与加计比较','④ 用本体系增量右乘修正','⑤ 倾斜对齐，不等于真值','⑥ 绕重力转动仍不可见'];
 const desc=[
 '软件参考系 W 保持固定。设备在 t₀ 已有姿态 R₀。静止加计给出 a₀，把其坐标乘 R₀，就得到固定的向上方向 k_W。',
 '设备真实转动到 t₁。陀螺积分得到 R_pred，可能带有倾斜与航向误差。青色真值只供讲解，实际程序并不知道它。',
 '同一个 k_W，经 R_pred 的转置转换成预测本体坐标 u。加计独立提供观测 v。它们不重合，说明估计的倾斜有误。',
 '先求 D，让 D u = v。姿态却必须右乘 ΔR = Dᵀ。动画逐渐施加这一最短修正，观察右图两个向量趋于重合。',
 '修正后 R_cᵀ k_W = a_S，方向约束已经满足。但橙色姿态与青色真值仍不同：重力对齐没有找回完整姿态。',
 '在 W 中，继续绕 k_W 转动整个姿态，右图预测向上坐标仍不变。试着调节航向滑杆：重力残差保持为零，姿态误差却改变。'
 ];
 let time=0,playing=false,last=0;
 const buttons=titles.map((t,i)=>{let b=document.createElement('button');b.textContent=t.slice(0,2);b.title=t;b.type='button';b.onclick=()=>{playing=false;time=i;update();};$('tgSteps').appendChild(b);return b;});
 root.querySelectorAll('button').forEach(b=>{b.style.cssText='background:#22334e;color:#e0f2fe;border:1px solid #557092;border-radius:7px;padding:8px 12px;cursor:pointer';});
 function calc(){
  const tilt=+$('tgTilt').value*rad,yaw=+$('tgYaw').value*rad;
  const b=pureTiltAxis(v),Rp=mm(exp(scale(k,yaw)),mm(Rt,exp(scale(b,tilt)))),u=mv(T(Rp),k),e=cross(v,u),theta=Math.atan2(norm(e),dot(v,u)),n=norm(e)>1e-12?unit(e):b;
  const delta=exp(scale(n,theta)),Rc=mm(Rp,delta);
  let R,q=0;if(time<1)R=R0;else if(time<2){let f=time-1;let truth=mm(exp(scale(k,.30*f)),mm(R0,exp(scale([.65,.25,-.12],f))));let bf=pureTiltAxis(mv(T(truth),k));R=mm(exp(scale(k,yaw*f)),mm(truth,exp(scale(bf,tilt*f))));}else if(time<3)R=Rp;else if(time<4){q=time-3;R=mm(Rp,exp(scale(n,theta*q)));}else if(time<5)R=Rc;else R=mm(exp(scale(k,Math.sin((time-5)*Math.PI*2)*.5)),Rc);
  const current=mv(T(R),k),E=mm(Rc,T(Rt));
  return {Rp,u,e,theta,n,b,delta,Rc,R,current,E,q};
 }
 function setup(canvas){let dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);let c=canvas.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,w,h);return {c,w,h};}
 // All geometry remains in 3D. Only the orbit camera and projection change.
 const defaultCamera=()=>({az:-Math.atan2(.8,.6),el:Math.asin(.6),zoom:1});
 const cameras={tgWorld:defaultCamera(),tgBody:defaultCamera()};let activeView='tgWorld';
 function cameraBasis(camera){let ca=Math.cos(camera.az),sa=Math.sin(camera.az),ce=Math.cos(camera.el),se=Math.sin(camera.el);return {right:[-sa,ca,0],up:[-se*ca,-se*sa,ce],near:[ce*ca,ce*sa,se]};}
 function projection(x,camera,w,h){let b=cameraBasis(camera),z=dot(b.near,x),factor=$('tgPerspective').checked?4.5/(4.5-z):1,s=Math.min(w,h)*.22*camera.zoom;return [w*.48+s*factor*dot(b.right,x),h*.58-s*factor*dot(b.up,x),z];}
 function changeCamera(id,action){activeView=id;action(cameras[id]);if($('tgLinkViews').checked){let other=id==='tgWorld'?'tgBody':'tgWorld';Object.assign(cameras[other],cameras[id]);}update();}
 function setView(az,el){changeCamera(activeView,c=>Object.assign(c,{az,el,zoom:1}));}
 function scene(canvas,body,d){let {c,w,h}=setup(canvas),camera=cameras[canvas.id],geometry=[],labels=[];
  const p=x=>projection(x,camera,w,h);
  function line(a,b,color,width=2,dash=[]){geometry.push({kind:'line',a:p(a),b:p(b),color,width,dash,depth:(p(a)[2]+p(b)[2])/2});}
  function arrow(a,b,color,label,width=3){line(a,b,color,width);let A=p(a),B=p(b),t=Math.atan2(B[1]-A[1],B[0]-A[0]);labels.push(()=>{c.beginPath();c.fillStyle=color;c.moveTo(B[0],B[1]);if(Math.hypot(B[0]-A[0],B[1]-A[1])>3){c.lineTo(B[0]-10*Math.cos(t-.4),B[1]-10*Math.sin(t-.4));c.lineTo(B[0]-10*Math.cos(t+.4),B[1]-10*Math.sin(t+.4));}else c.arc(B[0],B[1],3,0,2*Math.PI);c.closePath();c.fill();c.font='13px system-ui';let textWidth=c.measureText(label).width;c.fillText(label,Math.max(5,Math.min(w-textWidth-5,B[0]+7)),Math.max(66,Math.min(h-32,B[1]-7)));});}
  function device(R,color,offset=[0,0,0],axes=false){let corners=[];for(let z of [-.10,.10])for(let y of [-.28,.28])for(let x of [-.42,.42])corners.push(mv(R,[x,y,z]).map((v,i)=>v+offset[i]));for(let indices of [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]){let points=indices.map(i=>p(corners[i]));geometry.push({kind:'face',points,color:axes?'rgba(251,191,36,.16)':'rgba(34,211,238,.07)',depth:points.reduce((s,a)=>s+a[2],0)/4});}for(let i=0;i<8;i++)for(let mask of [1,2,4])if(i<(i^mask))line(corners[i],corners[i^mask],color,2);if(axes)T(R).forEach((a,i)=>arrow(offset,mv(I,scale(a,.85)).map((v,j)=>v+offset[j]),['#fb7185','#4ade80','#60a5fa'][i],['X_S','Y_S','Z_S'][i],2));}
  // A faint reference XY grid supplies depth cues; it is not a gravity plane.
  for(let a=-1;a<=1.001;a+=.25){line([a,-1,0],[a,1,0],'#23344c',1);line([-1,a,0],[1,a,0],'#23344c',1);}
  let colors=['#fb7185','#4ade80','#60a5fa'];I.forEach((a,i)=>arrow([0,0,0],scale(a,1.22),colors[i],['X','Y','Z'][i]+(body?'_S':'_W'),1.6));
  if(!body){arrow([0,0,0],scale(k,1.65),'#fff','k_W',3);device(d.R,'#fbbf24',[0,0,0],true);if(time>=1)device(Rt,'#22d3ee');if(time<1)arrow([0,0,0],scale(mv(R0,a0),1.65),'#fff','R₀ a₀ = k_W',3);c.fillStyle='#fbbf24';c.fillText(time<1?'橙色：t₀ 姿态':'橙色：当前估计姿态',12,26);if(time>=1){c.fillStyle='#22d3ee';c.fillText('青色：教学真值（同一原点）',12,48);}if(time>=4){let b=unit(cross(k,[1,0,0])),a=cross(k,b);for(let j=0;j<70;j++){let f=j/70*2*Math.PI,g=(j+1)/70*2*Math.PI;line(b.map((x,i)=>.8*(Math.cos(f)*x+Math.sin(f)*a[i])),b.map((x,i)=>.8*(Math.cos(g)*x+Math.sin(g)*a[i])),'#667891',1,[3,3]);}c.fillStyle='#a9bad2';c.fillText('虚线圆：绕 k_W 的不可观测转动',12,h-12);}}
  else{let obs=time<1?a0:v,pred=time<1?a0:d.current;arrow([0,0,0],scale(obs,1.60),'#22d3ee',time<1?'a₀（加计）':'v = a_S（加计）',5);arrow([0,0,0],scale(pred,1.22),'#fbbf24',time<1?'R₀ᵀ k_W':'u_current = Rᵀ k_W',3);if(time>=2&&time<4&&norm(d.e)>1e-10)arrow([0,0,0],scale(d.n,.9),'#c4b5fd','n ∥ v × u',2);c.fillStyle='#a9bad2';c.fillText('两向量表达在同一组本体轴坐标中',12,26);c.fillText('向量长度错开，仅为避免重合遮挡',12,h-12);}
  geometry.sort((a,b)=>a.depth-b.depth);for(let g of geometry){c.beginPath();if(g.kind==='face'){c.fillStyle=g.color;c.moveTo(g.points[0][0],g.points[0][1]);g.points.slice(1).forEach(p=>c.lineTo(p[0],p[1]));c.closePath();c.fill();}else{c.strokeStyle=g.color;c.lineWidth=g.width;c.setLineDash(g.dash);c.moveTo(g.a[0],g.a[1]);c.lineTo(g.b[0],g.b[1]);c.stroke();c.setLineDash([]);}}
  labels.forEach(draw=>draw());
 }
 function fmt(v){return '['+v.map(x=>x.toFixed(3)).join(', ')+']';}
 function matrix(R){return R.map(fmt).join('\n');}
 function update(){let d=calc(),i=Math.min(5,Math.floor(time));$('tgTime').value=time;$('tgPlay').textContent=playing?'Ⅱ 暂停':'▶ 播放';buttons.forEach((b,j)=>b.style.background=j===i?'#0e7490':'#22334e');$('tgTitle').textContent=titles[i];$('tgExplain').textContent=desc[i];$('tgTiltOut').textContent=$('tgTilt').value+'°';$('tgYawOut').textContent=$('tgYaw').value+'°';
  const notes=['之后固定重力参考，不逐帧重新学习。','本体系角速度产生右乘增量。','同向时无需纠偏，不计算零向量的归一化。','向量修正与姿态右乘增量互为转置。','注意：真值姿态这里也必须取转置。','重力观测无法区分不同的航向。'];
  $('tgEquation').innerHTML=equationMathML[i];$('tgEquation').insertAdjacentHTML('beforeend','<p style="font-size:14px;margin:12px 0 0">'+notes[i]+'</p>');let obs=time<1?a0:v;
  const currentError=mm(d.R,T(Rt));
  $('tgValues').textContent='k_W = '+fmt(k)+' （固定）\n观测 a_S = '+fmt(obs)+'\n预测 Rᵀ k_W = '+fmt(time<1?a0:d.current)+'\n当前重力方向误差 = '+angle(time<1?a0:d.current,obs).toFixed(2)+'°'+(time>=4?'；当前姿态与真值之差 = '+(Math.acos(Math.max(-1,Math.min(1,(currentError[0][0]+currentError[1][1]+currentError[2][2]-1)/2)))/rad).toFixed(2)+'°':'');
  const residual=norm(sub(mv(T(d.Rc),k),v)),axisResidual=norm(sub(mv(d.E,k),k));
  $('tgMatrices').textContent='教学注入轴 b = '+fmt(d.b)+'，b·v = '+dot(d.b,v).toExponential(3)+'\n初始重力夹角 = '+(d.theta/rad).toFixed(2)+'°（应等于纯倾斜滑杆）\n完整纠偏后绕 k_W 的有符号误差 = '+signedHeading(d.E).toFixed(2)+'°（应等于航向滑杆）\nR₀:\n'+matrix(R0)+'\nR_pred:\n'+matrix(d.Rp)+'\nΔR（完整对齐）:\n'+matrix(d.delta)+'\nR_c:\n'+matrix(d.Rc)+'\nR_true:\n'+matrix(Rt)+'\n‖R_cᵀ k_W − a_S‖ = '+residual.toExponential(3)+'\n‖(R_c R_trueᵀ) k_W − k_W‖ = '+axisResidual.toExponential(3);
  scene($('tgWorld'),false,d);scene($('tgBody'),true,d);
 }
 $('tgPlay').onclick=()=>{if(time>=5.999)time=0;playing=!playing;last=performance.now();update();};$('tgPrev').onclick=()=>{playing=false;time=Math.max(0,Math.floor(time)-1);update();};$('tgNext').onclick=()=>{playing=false;time=Math.min(5,Math.floor(time)+1);update();};$('tgReset').onclick=()=>{playing=false;time=0;$('tgTilt').value=28;$('tgYaw').value=42;update();};$('tgTime').oninput=()=>{playing=false;time=+$('tgTime').value;update();};['tgTilt','tgYaw'].forEach(id=>$(id).oninput=update);
 $('tgViewIso').onclick=()=>setView(defaultCamera().az,defaultCamera().el);$('tgViewFront').onclick=()=>setView(Math.PI/2,0);$('tgViewSide').onclick=()=>setView(0,0);$('tgViewTop').onclick=()=>setView(-Math.PI/2,Math.PI/2);$('tgViewReset').onclick=()=>{Object.assign(cameras.tgWorld,defaultCamera());Object.assign(cameras.tgBody,defaultCamera());update();};
 $('tgPerspective').onchange=update;$('tgLinkViews').onchange=()=>{if($('tgLinkViews').checked)changeCamera(activeView,()=>{});};
 for(let id of ['tgWorld','tgBody']){
  let canvas=$(id),pointers=new Map();
  const position=e=>[e.clientX,e.clientY];
  const distance=()=>{let a=[...pointers.values()];return a.length>=2?Math.hypot(a[0][0]-a[1][0],a[0][1]-a[1][1]):0;};
  canvas.addEventListener('pointerdown',e=>{if(e.pointerType==='mouse'&&e.button!==0)return;activeView=id;canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,position(e));canvas.style.cursor='grabbing';});
  canvas.addEventListener('pointermove',e=>{if(!pointers.has(e.pointerId))return;let previous=pointers.get(e.pointerId),before=distance();pointers.set(e.pointerId,position(e));if(pointers.size===1)changeCamera(id,c=>{c.az-=(e.clientX-previous[0])*.009;c.el=Math.max(-Math.PI/2+.005,Math.min(Math.PI/2-.005,c.el+(e.clientY-previous[1])*.009));});else{let after=distance();if(before>2&&after>2)changeCamera(id,c=>{c.zoom=Math.max(.55,Math.min(1.75,c.zoom*after/before));});}});
  const release=e=>{pointers.delete(e.pointerId);canvas.style.cursor=pointers.size?'grabbing':'grab';};
  ['pointerup','pointercancel','lostpointercapture'].forEach(event=>canvas.addEventListener(event,release));
  canvas.addEventListener('wheel',e=>{e.preventDefault();changeCamera(id,c=>{c.zoom=Math.max(.55,Math.min(1.75,c.zoom*Math.exp(-Math.sign(e.deltaY)*.10)));});},{passive:false});
  canvas.addEventListener('dblclick',()=>changeCamera(id,c=>Object.assign(c,defaultCamera())));
  canvas.addEventListener('keydown',e=>{let directions={ArrowLeft:[-.12,0],ArrowRight:[.12,0],ArrowUp:[0,.12],ArrowDown:[0,-.12]};if(directions[e.key]){e.preventDefault();changeCamera(id,c=>{c.az+=directions[e.key][0];c.el=Math.max(-Math.PI/2+.005,Math.min(Math.PI/2-.005,c.el+directions[e.key][1]));});}else if(['+','=','-'].includes(e.key)){e.preventDefault();changeCamera(id,c=>{c.zoom=Math.max(.55,Math.min(1.75,c.zoom*(e.key==='-'?.9:1.1)));});}else if(e.key==='Home'){e.preventDefault();changeCamera(id,c=>Object.assign(c,defaultCamera()));}});
 }
 function tick(now){if(playing){time=Math.min(5.999,time+(now-last)/4500);if(time>=5.999)playing=false;update();}last=now;requestAnimationFrame(tick);}window.addEventListener('resize',update);update();requestAnimationFrame(tick);
 // Public read-only diagnostics for reproducible browser tests.
 window.tiltLessonCheck=()=>{let d=calc();return {alignmentResidual:norm(sub(mv(T(d.Rc),k),v)),headingAxisResidual:norm(sub(mv(d.E,k),k)),tiltAxisDotUp:dot(d.b,v),recoveredHeadingDegrees:signedHeading(d.E),thetaDegrees:d.theta/rad,stage:Math.floor(time),cameras:JSON.parse(JSON.stringify(cameras)),projectedWorldAxes:I.map(x=>projection(x,cameras.tgWorld,$('tgWorld').clientWidth,$('tgWorld').clientHeight))};};
})();
</script>
```

## C 为什么修正要右乘逆矩阵？

使用列向量，定义预测与观测方向：

$$
\mathbf u=R_{\mathrm{pred}}^{\mathsf T}\mathbf k_W,
\qquad \mathbf v=\mathbf a_S.
$$

在本体坐标中寻找向量修正 $D$，再逐步代入并取转置：

$$
\begin{aligned}
D\mathbf u&=\mathbf v,\\
\mathbf v&=D R_{\mathrm{pred}}^{\mathsf T}\mathbf k_W,\\
R_c^{\mathsf T}&=D R_{\mathrm{pred}}^{\mathsf T},\\
R_c&=R_{\mathrm{pred}}D^{\mathsf T}.
\end{aligned}
$$

所以，作用于向量的 $D$ 与右乘姿态的 $\Delta R$ 不是同一个矩阵。右乘 $\Delta R$ 表示在当前本体系轴坐标中的姿态增量。

动画使用观测方向叉乘预测方向：

$$
\mathbf e=\mathbf v\times\mathbf u,\qquad
\mathbf n=\frac{\mathbf e}{\lVert\mathbf e\rVert},\qquad
\theta=\operatorname{atan2}\!\left(\lVert\mathbf e\rVert,\mathbf v^{\mathsf T}\mathbf u\right).
$$

$$
\boxed{D=\operatorname{Exp}(-\theta\mathbf n),\qquad
\Delta R=D^{\mathsf T}=\operatorname{Exp}(\theta\mathbf n).}
$$

同向时取单位矩阵，不归一化零叉积；反向时最短轴不唯一。本演示限制误差范围，避开反向退化。这里 $\operatorname{Exp}(\boldsymbol\phi)=\exp([\boldsymbol\phi]_\times)$ 是旋转向量的指数映射。

### 两个误差滑杆怎样独立？

设教学真值的本体向上方向为 $\mathbf v=R_{\mathrm{true}}^{\mathsf T}\mathbf k_W$。将候选轴投影到 $\mathbf v$ 的垂直平面，得到单位注入轴 $\mathbf b$：

$$
\boxed{\mathbf b=
\frac{\mathbf e_x-(\mathbf e_x^{\mathsf T}\mathbf v)\mathbf v}
{\left\lVert\mathbf e_x-(\mathbf e_x^{\mathsf T}\mathbf v)\mathbf v\right\rVert},
\qquad \mathbf b^{\mathsf T}\mathbf v=0.}
$$

如果候选轴接近 $\mathbf v$，就换用 $\mathbf e_y$。这只是教学构造，真实滤波器不需要知道真值或 $\mathbf b$。

构造带误差的教学估计：

$$
R_{\mathrm{pred}}=H(\mathbf k_W,\psi)\,R_{\mathrm{true}}\operatorname{Exp}(\theta\mathbf b).
$$

因为 $H\mathbf k_W=\mathbf k_W$，有 $\mathbf u=\operatorname{Exp}(-\theta\mathbf b)\mathbf v$。又因为 $\mathbf b^{\mathsf T}\mathbf v=0$，在本页 0–65° 范围内，两方向的夹角恰好为 $\theta$。

完整最短修正及其结果为：

$$
\Delta R=\operatorname{Exp}(-\theta\mathbf b),\qquad
\boxed{R_c=H(\mathbf k_W,\psi)\,R_{\mathrm{true}}.}
$$

注入倾斜被消除，$\psi$ 原样保留。$\mathbf b$ 是注入轴，$\mathbf n$ 是纠偏轴；非零倾斜时 $\mathbf n=-\mathbf b$，两者不要混淆。

## D 数学上的对齐，和代码的小步反馈

| 教学用精确对齐 | 当前代码 |
| --- | --- |
| $R_c=R_{\mathrm{pred}}\operatorname{Exp}(\theta\mathbf n)$ | $R_n=R_{\mathrm{pred}}\operatorname{Exp}(\gamma w\Delta t\,\mathbf e)$ |
| 一次消除理想方向误差 | 每帧只修正一小步 |
| $\theta$ 是完整夹角 | 单步角为 $\gamma w\Delta t\sin\theta$ |

默认 $\gamma=0.8\,\mathrm{s}^{-1}$，$w$ 是加计可靠性权重。$\Delta t$ 来自设备包内时间链路，不是画面刷新间隔。动画中缓慢施加完整 $\theta$，是教学插值，不代表代码实际迭代过程。

当加计受明显平移加速度影响时，方向不再等于真实向上方向。代码降低权重，但不能保证完全分离运动加速度。本讲先在理想、低动态条件下解释几何关系。

## E 为什么航向仍然未知？

假设修正与真值已经满足同一重力约束：

$$
R_c^{\mathsf T}\mathbf k_W=R_{\mathrm{true}}^{\mathsf T}\mathbf k_W.
$$

定义相对误差 $E=R_c R_{\mathrm{true}}^{\mathsf T}$。两边左乘 $R_c$，可得：

$$
\mathbf k_W=R_c R_{\mathrm{true}}^{\mathsf T}\mathbf k_W=E\mathbf k_W.
$$

$E\in SO(3)$ 又保持 $\mathbf k_W$ 不变，因此它是绕 $\mathbf k_W$ 的旋转：

$$
\boxed{E=\operatorname{Exp}(\psi\mathbf k_W),\qquad
R_c=H(\mathbf k_W,\psi)R_{\mathrm{true}}.}
$$

```callout warn 这是一条有条件的结论
“仅剩绕重力转动”要求理想重力观测，且倾斜已完全对齐。实际噪声、动态加速度和有限反馈增益会留下倾斜残差。$\mathbf k_W$ 在 $t_0$ 建立的是软件坐标中的参考方向，不能据此宣称初始绝对姿态已知。
```

一个向量给出两个独立约束，无法确定三自由度姿态。即使重力方向误差为零，航向误差也可以很大。这正是下一讲引入相机方向观测的原因；本页尚未演示相机标定。

## F 文档和代码依据

- 数学文档：`docs/GYRO_GRAVITY_TILT_MATH.md`，第 3–7、10 节。
- 滤波实现：`wt901_viz/so3_filter.py`，`AdaptiveTiltFilter.update()`。
- 时间和人体姿态入口：`OptUSBProj-ubuntu/optusbproj_ubuntu/pose.py`。

本页只解释第一步。后续可分别扩展：重力不可观测性的完整证明、相机换基共轭、相机标定与航向补偿。
