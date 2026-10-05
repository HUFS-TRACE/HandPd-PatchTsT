# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for cand in ["Malgun Gothic","MalgunGothic","Gulim","Batang"]:
    try:
        fm.findfont(fm.FontProperties(family=cand), fallback_to_default=False)
        plt.rcParams["font.family"]=cand; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"]=False

conds=["R0\n전체 61명","R4\n2016년 제외 40명","R5\n45세 이상 41명"]
gap=[14.7,11.3,3.3]; rule=[.738,.775,.659]; model=[.813,.849,.805]; major=[.574,.650,.585]
adv=[m-r for m,r in zip(model,rule)]
x=range(3)
fig,(a1,a2)=plt.subplots(1,2,figsize=(10.5,4.0),gridspec_kw={"width_ratios":[1.35,1]})

a1.plot(x,model,"o-",lw=2.6,ms=9,color="#1b4965",label="PatchTST 모델",zorder=3)
a1.plot(x,rule,"s--",lw=2.4,ms=8,color="#c1121f",label="연령 규칙(임계 재최적화)",zorder=3)
a1.plot(x,major,"^:",lw=1.6,ms=7,color="#8d99ae",label="다수결 기준선",zorder=2)
for i,(m,r) in enumerate(zip(model,rule)):
    a1.annotate(f"{m:.3f}",(i,m),textcoords="offset points",xytext=(0,11),ha="center",fontsize=9,color="#1b4965",fontweight="bold")
    a1.annotate(f"{r:.3f}",(i,r),textcoords="offset points",xytext=(0,-17),ha="center",fontsize=9,color="#c1121f")
a1.set_xticks(list(x)); a1.set_xticklabels(conds,fontsize=9)
a1.set_ylim(.53,.95); a1.set_ylabel("피험자 정확도",fontsize=10)
a1.set_title("(a) 연령을 맞추면 규칙은 무너지고 모델은 버틴다",fontsize=11,pad=26)
a1.legend(fontsize=8.5,loc="upper center",bbox_to_anchor=(.5,-.13),ncol=3,frameon=False)
a1.grid(axis="y",alpha=.25)
for i,g in enumerate(gap):
    a1.annotate(f"연령차 +{g}세",(i,.918),ha="center",fontsize=9,color="#444",
                bbox=dict(boxstyle="round,pad=.3",fc="#f2f2f2",ec="none"))

b=a2.bar([0,1,2],adv,color=["#a9bcc9","#a9bcc9","#1b4965"],width=.58)
for i,v in enumerate(adv):
    a2.annotate(f"+{v:.3f}",(i,v),textcoords="offset points",xytext=(0,5),ha="center",fontsize=10,fontweight="bold")
a2.set_xticks([0,1,2]); a2.set_xticklabels(["R0\n+14.7세","R4\n+11.3세","R5\n+3.3세"],fontsize=9)
a2.set_ylim(0,.205); a2.set_ylabel("모델 정확도 - 연령 규칙",fontsize=10)
a2.set_title("(b) 교란을 끊을수록 우위가 커진다",fontsize=11,pad=10)
a2.grid(axis="y",alpha=.25)
a2.annotate("연령을 대리 학습했다면\n반대 방향이어야 한다",
            xy=(1.70,.140), xytext=(0.50,.180),
            ha="center", va="center", fontsize=9.5, color="#1b4965",
            arrowprops=dict(arrowstyle="->", color="#1b4965", lw=1.3,
                            connectionstyle="arc3,rad=-.25"))
fig.tight_layout()
fig.savefig("results/fig_three_gates.png",dpi=200,bbox_inches="tight",facecolor="white")
print("saved")
