from pathlib import Path
import sys,json,hashlib
P=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(P/'.deps'))
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'font.size':9,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
F=P/'report_work/figures05';F.mkdir(exist_ok=True);A=P/'report_work/thesis05/analysis';manifest={}
blue='#0072B2';orange='#D55E00';gray='#666666';green='#009E73'
def save(fig,name,df=None,sources=[]):
 for ext in ['png','svg','pdf']:fig.savefig(F/(name+'.'+ext),dpi=320)
 plt.close(fig)
 if df is not None:df.to_csv(F/(name+'.csv'),index=False,encoding='utf-8-sig')
 manifest[name]={'sources':[{'path':str(p.relative_to(P)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],'kind':'method diagram' if df is None else 'data visualization','smoothing':'none'}
def box(ax,x,y,w,h,t,edge=blue):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.007',fc='#f3f6f8',ec=edge,lw=.9));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8.6)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.1,'color':'#526a7a'})
fig,ax=plt.subplots(figsize=(6.5,3.5),layout='constrained');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
items=[('创新1 精确核验\n能确定哪些目标','E1 证据条件与消融\n无支撑交付和误拒','5.1 精确与部分回答\n保留完整证书路径'),('创新2 增量选路\n缺口如何估计','E2 同信息强基线\n均值与最大值分评','5.2 基础与增强选路\n保留各指标最优值'),('创新3 精度与补证\n下一条证据取什么','E3 同点区间与同预算\n宽度 覆盖 补证误差','5.3 精度服务与补证\n保留时间覆盖规则'),('创新4 复用与核验\n更新后如何追踪','E4 缓存消融与源行回查\n时延 错配 不可变记录','5.4 安全复用闭环\n保留版本与等级约束')]
for i,row in enumerate(items):
 y=.78-i*.235
 for x,t in zip([.01,.355,.70],row):box(ax,x,y,.285,.17,t)
 arrow(ax,(.30,y+.085),(.348,y+.085));arrow(ax,(.645,y+.085),(.693,y+.085))
save(fig,'roadmap')
fig,ax=plt.subplots(figsize=(6.5,3.1),layout='constrained');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
coords=[(.02,.72,'固定统计合同\n站点 时刻 变量 统计量'),(.36,.72,'I1 精确路径核验\n确定目标直接交付'),(.70,.72,'I2 本站优先选路\n不足目标形成估计'),(.70,.20,'I3 精度门控\n满足偏好则交付'),(.36,.20,'I3 时间覆盖补证\n预算内取回关键时刻'),(.02,.20,'I4 保存与追加核验\n原推论 源行 新证书')]
for x,y,t in coords:box(ax,x,y,.27,.21,t)
for a,b in [((.295,.825),(.35,.825)),((.635,.825),(.69,.825)),((.835,.71),(.835,.42)),((.695,.305),(.64,.305)),((.495,.42),(.495,.71))]:arrow(ax,a,b)
ax.plot([.39,.155,.155],[.715,.59,.42],color='#526a7a',lw=1.1);arrow(ax,(.155,.48),(.155,.415));ax.text(.02,.53,'已确定目标',fontsize=8,color=gray)
ax.plot([.835,.835,.155],[.195,.10,.10],color='#526a7a',lw=1.1);arrow(ax,(.155,.10),(.155,.195));ax.text(.57,.055,'满足精度',fontsize=8,color=gray)
ax.text(.665,.36,'需补证',fontsize=7,ha='center',color=gray);ax.text(.51,.56,'新证据',fontsize=8,color=gray)
save(fig,'workflow05')
q=pd.read_csv(P/'v3/results/data_audit.csv',dtype={'station':str});mat=q.pivot(index='station',columns='year',values='coverage');fig,ax=plt.subplots(figsize=(6.5,2.4),layout='constrained');im=ax.imshow(mat,vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.set_xticks(range(4),mat.columns);ax.set_yticks(range(3),mat.index);ax.set(xlabel='年份',ylabel='站点编号')
for i in range(3):
 for j in range(4):v=mat.iloc[i,j];ax.text(j,i,f'{v:.1%}',ha='center',va='center',color='white' if v>.7 else 'black',fontsize=9)
fig.colorbar(im,ax=ax,label='有效小时比例',shrink=.8);save(fig,'data_quality',q,[P/'v3/results/data_audit.csv'])
e=pd.read_csv(P/'v3/results/evidence_metrics.csv');e=e[e.split=='test'].set_index('method');order=['A_all_or_none','A_raw_only','A_scope_off','H_naive_visible','M_exact'];names=['整题全答或全拒','仅原始值','去对象核验','直接算可见值','完整证书路径'];fig,ax=plt.subplots(1,3,figsize=(6.5,2.75),layout='constrained')
for a,col,title in zip(ax,['UAR','FAR','coverage'],['无支撑交付率 ↓','误拒率 ↓','目标覆盖率 ↑']):
 values=e.loc[order,col].to_numpy()*100;a.barh(range(5),values,color=[gray]*4+[blue]);a.set_yticks(range(5),names if col=='UAR' else ['']*5);a.invert_yaxis();a.set_xlim(0,105);a.set_title(title,fontsize=9);a.set_xlabel('%');a.grid(axis='x',alpha=.16);a.set_axisbelow(True)
 for i,v in enumerate(values):a.text(v+2,i,f'{v:.1f}',va='center',fontsize=7.5)
save(fig,'e1_results',e.loc[order].reset_index(),[P/'v3/results/evidence_metrics.csv'])
m=pd.read_csv(P/'v4/results/metrics.csv');g=m[(m.statistic=='mean')&m.scenario.isin(['mask1','mask2','mask4'])&m.method.isin(['P_persistence','P_local_interp','P_delta_ridge','M_router'])];fig,ax=plt.subplots(figsize=(6.5,2.65),layout='constrained')
for j,(method,label,color,hatch) in enumerate([('P_persistence','持续性 P0',gray,''),('P_local_interp','本站插值 L','#9bc5dc','//'),('P_delta_ridge','强制残差 D',green,'..'),('M_router','基础选路 B',blue,'xx')]):
 f=g[g.method==method].set_index('scenario').loc[['mask1','mask2','mask4']];ax.bar(np.arange(3)+(j-1.5)*.19,f.RMSE,width=.18,label=label,color=color,hatch=hatch,edgecolor='white',lw=.4)
ax.set_xticks(range(3),['缺1条','缺2条','全缺口']);ax.set(ylabel='均值 RMSE（℃）',ylim=(0,.56));ax.legend(fontsize=7,ncol=2);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);save(fig,'e2_baselines',g,[P/'v4/results/metrics.csv'])
b=pd.read_csv(A/'paired_intervals.csv');b=b[b.block_days==7];fig,ax=plt.subplots(1,2,figsize=(6.5,2.85),layout='constrained')
for a,(method,ref,title,color) in zip(ax,[('v4','local','(a) 基础B − 本站L',blue),('v5_no_guard','v4','(b) 增强B+ − 基础B',orange)]):
 f=b[(b.method==method)&(b.reference==ref)].sort_values(['cohort','statistic'],ascending=[True,False]);labs=[str(r.cohort)+' '+('均值' if r.statistic=='mean' else '最大值') for r in f.itertuples()];yy=np.arange(len(f));a.errorbar(f.delta_RMSE,yy,xerr=[f.delta_RMSE-f.lower,f.upper-f.delta_RMSE],fmt='o',color=color,capsize=3,ms=4);a.axvline(0,color=gray,ls='--',lw=1);a.set_yticks(yy,labs);a.invert_yaxis();a.set_title(title,fontsize=9);a.set_xlabel('配对 RMSE 差（℃）');a.grid(axis='x',alpha=.16)
save(fig,'e2_paired',b,[A/'paired_intervals.csv'])
pt=pd.read_csv(A/'point_metrics.csv');pt=pt[pt.cohort==2021] if pt.cohort.dtype!='object' else pt[pt.cohort=='2021'];scenes=['mask4','block72','bias2','relation_shift'];fig,ax=plt.subplots(1,2,figsize=(6.5,2.6),layout='constrained');parts=[]
for a,stat,title in zip(ax,['mean','max'],['(a) 均值','(b) 最大值']):
 for j,(method,label,color,marker) in enumerate([('local','本站L',gray,'s'),('v4','基础B',blue,'o'),('v5_no_guard','增强B+',orange,'^'),('v5','增强含回退G',green,'D')]):
  f=pt[(pt.statistic==stat)&(pt.method==method)].set_index('scenario').loc[scenes].reset_index();parts.append(f);a.plot(range(4),f.RMSE,label=label,color=color,marker=marker,ls=[':', '-', '--','-.'][j],ms=4)
 a.set_xticks(range(4),['正常','72h缺口','供体+2℃','关系反转'],rotation=18);a.set(title=title,ylabel='RMSE（℃）',ylim=(0,None));a.grid(axis='y',alpha=.16)
ax[0].legend(fontsize=6.8);save(fig,'e2_stress',pd.concat(parts),[A/'point_metrics.csv'])
c=pd.read_csv(P/'report_work/thesis02/station_quarter.csv',dtype={'station':str});fig,ax=plt.subplots(figsize=(6.5,2.2),layout='constrained');mat=c.pivot(index='station',columns='quarter',values='picp');cm=plt.get_cmap('YlGnBu').copy();cm.set_bad('#dddddd');im=ax.imshow(mat,vmin=0,vmax=1,cmap=cm,aspect='auto');ax.set_xticks(range(4),['Q1','Q2','Q3','Q4']);ax.set_yticks(range(3),mat.index)
for i in range(3):
 for j in range(4):v=mat.iloc[i,j];ax.text(j,i,'NA' if pd.isna(v) else f'{v:.1%}',ha='center',va='center',color='white' if v>.75 else 'black',fontsize=9)
fig.colorbar(im,ax=ax,label='区间覆盖率 PICP',shrink=.8);save(fig,'conditional_coverage',c,[P/'report_work/thesis02/station_quarter.csv'])
i=pd.read_csv(A/'interval_metrics.csv');g=i[(i.scenario=='mask4')&(i.statistic=='mean')];fig,ax=plt.subplots(1,2,figsize=(6.5,2.6),layout='constrained')
for j,(policy,label,color,hatch,mark) in enumerate([('fixed','固定区间 F',blue,'//','o'),('conditional','条件区间 Q',orange,'..','s')]):
 f=g[g.interval_policy==policy].sort_values('cohort');ax[0].bar(np.arange(2)+(j-.5)*.27,f.MPIW,width=.25,label=label,color=color,hatch=hatch,edgecolor='white');ax[1].plot(range(2),f.PICP*100,label=label,color=color,marker=mark,ls=['-','--'][j],ms=5)
ax[0].set(ylabel='平均区间宽度（℃）',ylim=(0,1.5));ax[1].set(ylabel='实测覆盖率（%）',ylim=(0,105));ax[1].axhline(90,color=gray,ls=':',lw=1);ax[1].text(.4,82,'名义90%',fontsize=8,color=gray)
for a in ax:a.set_xticks(range(2),['2021 n=922','2026 n=80']);a.grid(axis='y',alpha=.16);a.set_axisbelow(True)
ax[0].legend(fontsize=8);save(fig,'interval_matched',g,[A/'interval_metrics.csv'])
fig,ax=plt.subplots(1,2,figsize=(6.5,2.55),layout='constrained');parts=[]
for a,(folder,title,cohort) in zip(ax,[('v6','(a) 留出年份 n=922','2021'),('v5','(b) 近期快照 n=80','2026')]):
 f=pd.read_csv(P/folder/'results/acquisition_metrics.csv');f['cohort']=cohort;parts.append(f)
 for policy,label,color,mark,ls in [('bridge','时间覆盖 T',orange,'o','-'),('hashed','哈希顺序 H',blue,'s','--'),('earliest','时间顺序 E',gray,'^',':')]:
  g=f[f.policy==policy];a.plot(g.budget,g.RMSE,label=label,color=color,marker=mark,ls=ls,lw=1.3,ms=4)
 a.set(title=title,xlabel='补回记录数（条）',ylabel='均值 RMSE（℃）',xticks=range(5),ylim=(0,None));a.grid(alpha=.16)
ax[0].legend(fontsize=7.5);save(fig,'acquisition05',pd.concat(parts),[P/'v6/results/acquisition_metrics.csv',P/'v5/results/acquisition_metrics.csv'])
parts=[]
for cohort,folder in [('2021','v6'),('2026','v5')]:
 f=pd.read_csv(P/folder/'results/acquisition_bootstrap.csv');f=f[f.budget==2].copy();f['cohort']=cohort;parts.append(f)
g=pd.concat(parts);fig,ax=plt.subplots(figsize=(6.5,2.1),layout='constrained');yy=range(4);labs=[r.cohort+' T−'+('H' if r.reference=='hashed' else 'E') for r in g.itertuples()];ax.errorbar(g.delta_RMSE,yy,xerr=[g.delta_RMSE-g.lower,g.upper-g.delta_RMSE],fmt='o',color=orange,capsize=3);ax.axvline(0,color=gray,ls='--');ax.set_yticks(yy,labs);ax.invert_yaxis();ax.set_xlabel('两条预算的配对 RMSE 差（℃）');ax.grid(axis='x',alpha=.16);save(fig,'acquisition_paired',g,[P/'v6/results/acquisition_bootstrap.csv',P/'v5/results/acquisition_bootstrap.csv'])
c=pd.read_csv(P/'v3/results/cache_metrics.csv');g=c.groupby('policy')[['P50_ms','P95_ms','unsafe_rate']].median();order=['C0_resident_no_cache','C1_LRU','C3_hot_path','A_no_hotness','C2_naive_similar'];labs=['无缓存 K0','结果LRU Kr','热度路径 Kh','路径LRU Kp','相似答案 Kn'];fig,ax=plt.subplots(1,2,figsize=(6.5,2.85),layout='constrained');y=np.arange(5)
ax[0].barh(y-.16,g.loc[order,'P50_ms'],height=.29,label='P50',color=blue);ax[0].barh(y+.16,g.loc[order,'P95_ms'],height=.29,label='P95',color=orange);ax[0].set_yticks(y,labs);ax[0].invert_yaxis();ax[0].set_xlabel('延迟（ms）');ax[0].legend(fontsize=8,loc='lower right',bbox_to_anchor=(1,1.0),ncol=2,frameon=False);ax[1].barh(y,g.loc[order,'unsafe_rate']*100,color=[gray,gray,gray,blue,orange]);ax[1].set_yticks(y,labs);ax[1].invert_yaxis();ax[1].set_xlabel('错误复用请求比例（%）');ax[1].set_xlim(0,75)
for a in ax:a.grid(axis='x',alpha=.16);a.set_axisbelow(True)
save(fig,'cache05',g.reset_index(),[P/'v3/results/cache_metrics.csv'])
(F/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print('Exported',len(manifest),'figures')


from pathlib import Path
import sys,json,hashlib,shutil
P=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(P/'.deps'))
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'font.size':9,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
F=P/'report_work/figures06';F.mkdir(exist_ok=True)
for p in (P/'report_work/figures05').iterdir():
 if p.is_file():shutil.copy2(p,F/p.name)
manifest=json.loads((F/'manifest.json').read_text(encoding='utf-8'));blue='#0072B2';orange='#D55E00';gray='#666666';green='#009E73'
def save(fig,name,df=None):
 for ext in ['png','svg','pdf']:fig.savefig(F/(name+'.'+ext),dpi=320,facecolor='white')
 plt.close(fig)
 if df is not None:df.to_csv(F/(name+'.csv'),index=False,encoding='utf-8-sig')
 manifest[name]={'kind':'conceptual diagram' if df is None else 'data visualization','sources':[] if df is None else [{'path':'v7/results/workflow_metrics.csv','sha256':hashlib.sha256((P/'v7/results/workflow_metrics.csv').read_bytes()).hexdigest()}],'smoothing':'none'}
fig,ax=plt.subplots(figsize=(6.5,3.5),layout='constrained');ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
rows=[('充分性与引用研究','缺原始值是否一定拒答','I1 合法证书与逐目标交付\nE1 检验误拒与误交付'),('按需计算与缺测重建','相关是否真的带来增量','I2 本站优先与增益选路\nE2 保留强基线和负结果'),('缺口澄清与主动获取','补哪条 何时停止','I3 时间覆盖与阶段校准\nE3选时刻 E5验证停止'),('语义缓存与来源追踪','高频复用如何保持可信','I4 合同版本与不可变记录\nE4消融 E5组合回放')]
for i,row in enumerate(rows):
 y=.79-i*.235
 for j,(x,w) in enumerate([(.01,.28),(.35,.28),(.69,.30)]):
  ax.add_patch(FancyBboxPatch((x,y),w,.16,boxstyle='round,pad=.006',fc='#f3f6f8',ec=blue,lw=.9));ax.text(x+w/2,y+.08,row[j],ha='center',va='center',fontsize=8)
 for a,b in [(.295,.345),(.635,.685)]:ax.annotate('',xy=(b,y+.08),xytext=(a,y+.08),arrowprops={'arrowstyle':'->','color':gray})
save(fig,'thinking06')
z=pd.read_csv(P/'v7/results/workflow_metrics.csv');g=z[z.tau==.25];fig,ax=plt.subplots(figsize=(6.5,2.8),layout='constrained')
configs=[('B_only','B不补证',gray,'s',(8,9)),('E_adapt2','E自适应2',gray,'v',(8,0)),('H_adapt2','H自适应2',blue,'^',(-84,0)),('T_adapt2','T自适应2/4',orange,'o',(-10,-22)),('T_fixed2','T固定2',green,'D',(8,-3)),('all4','全补4条',blue,'*',(-66,-12))]
for p,label,color,mark,offset in configs:
 r=g[g.policy==p].iloc[0];ax.scatter(r.avg_records,r.qualified_rate*100,color=color,marker=mark,s=45,zorder=3);ax.annotate(label,(r.avg_records,r.qualified_rate*100),xytext=offset,textcoords='offset points',fontsize=8)
ax.set(xlabel='平均补回记录数（条）',ylabel='精度成功交付率（%）',xlim=(-.18,4.3),ylim=(-5,108));ax.grid(alpha=.18);save(fig,'workflow_tradeoff06',g)
fig,ax=plt.subplots(1,2,figsize=(6.5,2.7),layout='constrained')
for p,label,color,marker,ls in [('T_adapt2','T自适应2',orange,'o','-'),('T_fixed2','T固定2',green,'s','--'),('T_adapt4','T自适应4',blue,'^',':')]:
 g=z[z.policy==p].sort_values('tau');ax[0].plot(g.tau,g.qualified_rate*100,label=label,color=color,marker=marker,ls=ls,ms=4);ax[1].plot(g.tau,g.avg_records,label=label,color=color,marker=marker,ls=ls,ms=4)
for a in ax:a.set_xticks([.1,.25,.5]);a.set_xlabel('要求半宽 τ（℃）');a.grid(alpha=.18)
ax[0].set(ylabel='精度成功交付率（%）',ylim=(0,105));ax[1].set(ylabel='平均补回记录数（条）',ylim=(0,4.2));ax[0].legend(fontsize=7,loc='lower right');save(fig,'workflow_sensitivity06',z)
(F/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print('Figures:',len(manifest))
