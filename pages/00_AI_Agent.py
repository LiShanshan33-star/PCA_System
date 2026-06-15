import streamlit as st
st.set_page_config(page_title="全流程分析智能体", page_icon="⚡", layout="wide")
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from datetime import datetime
import hashlib, time, json
from io import StringIO

from utils.agent_engine import load_and_profile, detect_structure, execute_analysis, ai_interpret, generate_agent_pdf
from utils.theme import apply_theme, Colors, kpi_card, render_copilot_sidebar
apply_theme()

from matplotlib import font_manager
for name in ["Microsoft YaHei","SimHei","Arial Unicode MS"]:
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"]=[name]; plt.rcParams["axes.unicode_minus"]=False; break

# Cache layer
@st.cache_data(ttl=3600, show_spinner=False)
def cached_profile(file_bytes, file_name):
    from io import BytesIO; buf=BytesIO(file_bytes); buf.name=file_name; return load_and_profile(buf)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_structure(df_json, profile_json):
    return detect_structure(pd.read_json(StringIO(df_json)), profile_json)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_analysis(df_json, profile_json, structure_json, lsl, usl, target):
    return execute_analysis(pd.read_json(StringIO(df_json)), profile_json, json.loads(structure_json), lsl, usl, target)

def _init_state():
    for k,v in {"ag_df":None,"ag_profile":None,"ag_structure":None,"ag_result":None,"ag_ai_text":None,"ag_pdf_bytes":None,"ag_lsl":None,"ag_usl":None,"ag_target":None,"ag_file_hash":None,"ag_raw_bytes":None,"ag_file_name":None}.items():
        if k not in st.session_state: st.session_state[k]=v

def agent_page():
    _init_state(); render_copilot_sidebar()
    st.markdown('<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;"><div style="width:40px;height:40px;background:linear-gradient(135deg,#2563eb,#7c3aed);border-radius:8px;display:flex;align-items:center;justify-content:center;color:white;font-size:1.2rem;">AI</div><div><div style="font-weight:700;font-size:1.2rem;color:var(--pca-text-primary);">全流程分析智能体</div><div style="font-size:0.8rem;color:var(--pca-text-secondary);">上传 -> 自动识别 -> 分析 -> AI解读 -> 导出</div></div></div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        uf=st.file_uploader("拖拽或点击上传数据文件",type=["csv","xlsx","xls"],label_visibility="collapsed")
    
    with st.expander("规格限设置（可选）"):
        c1,c2,c3=st.columns(3)
        l=c1.number_input("LSL",value=None,step=0.01,format="%.4f")
        u=c2.number_input("USL",value=None,step=0.01,format="%.4f")
        t=c3.number_input("Target",value=None,step=0.01,format="%.4f")
        st.session_state["ag_lsl"]=l if l!=0 else None
        st.session_state["ag_usl"]=u if u!=0 else None
        st.session_state["ag_target"]=t if t!=0 else None
    
    if uf is None and st.session_state["ag_df"] is None:
        st.info("请上传数据文件开始分析")
        return
    
    if uf is not None:
        raw=uf.getvalue(); nh=hashlib.md5(raw).hexdigest()
        if nh!=st.session_state.get("ag_file_hash"):
            st.session_state["ag_raw_bytes"]=raw; st.session_state["ag_file_name"]=uf.name; st.session_state["ag_file_hash"]=nh
            for k in ["ag_df","ag_profile","ag_structure","ag_result","ag_ai_text","ag_pdf_bytes"]: st.session_state[k]=None
    
    if st.session_state["ag_df"] is None:
        with st.spinner("加载数据..."):
            df,profile=cached_profile(st.session_state["ag_raw_bytes"],st.session_state["ag_file_name"])
            st.session_state["ag_df"]=df; st.session_state["ag_profile"]=profile
    
    df=st.session_state["ag_df"]; profile=st.session_state["ag_profile"]
    if df is None: return
    
    st.markdown('<div class="pca-divider"></div>',unsafe_allow_html=True)
    ch,cr=st.columns([4,1])
    with ch: st.markdown("##### 数据画像")
    with cr:
        if st.button("重新上传",key="ag_rst",use_container_width=True):
            for k in list(st.session_state.keys()):
                if k.startswith("ag_"): del st.session_state[k]
            st.cache_data.clear(); st.rerun()
    
    k1,k2,k3,k4=st.columns(4)
    k1.markdown(kpi_card("行数",f"{profile['rows']:,}",""),unsafe_allow_html=True)
    k2.markdown(kpi_card("列数",f"{profile['cols']}",""),unsafe_allow_html=True)
    k3.markdown(kpi_card("数值列",f"{len(profile['num_cols'])}",""),unsafe_allow_html=True)
    m=max(profile["missing_pct"].values()) if profile["missing_pct"] else 0
    k4.markdown(kpi_card("缺失率",f"{m:.1f}%",""),unsafe_allow_html=True)
    with st.expander("数据预览"): st.dataframe(df.head(10),use_container_width=True)
    
    if st.session_state["ag_structure"] is None:
        df_json=df.to_json(); structure=cached_structure(df_json,profile); st.session_state["ag_structure"]=structure
    
    structure=st.session_state["ag_structure"]
    if "error" in structure: st.error(structure["error"]); return
    
    st.markdown('<div class="pca-divider"></div>',unsafe_allow_html=True)
    st.markdown("##### 结构识别结果")
    fl={"wide":"宽格式（"+str(structure.get("subgroup_size","?"))+"列测量值）","long_count":"计数型数据","long_defect":"缺陷计数数据","single_column":"单列连续数据"}
    fmt=structure.get("format","unknown")
    st.markdown("**数据结构**："+fl.get(fmt,fmt))
    if "normality_p" in structure:
        p=structure["normality_p"]; st.markdown("**正态性**："+("P="+f"{p:.4f} (正态)" if p>0.05 else "P="+f"{p:.4f} (非正态)"))
    st.markdown("**推荐方法**：")
    for m in structure.get("methods",[]):
        b=" 推荐" if m["id"]==structure.get("recommended") else ""; st.markdown("- **"+m["name"]+"**"+b)
    
    if st.session_state["ag_result"] is None:
        df_json=df.to_json(); sj=json.dumps(structure,default=str)
        t0=time.time(); result=cached_analysis(df_json,profile,sj,st.session_state.get("ag_lsl"),st.session_state.get("ag_usl"),st.session_state.get("ag_target"))
        st.session_state["ag_result"]=result; st.caption("耗时 "+f"{time.time()-t0:.1f}s（缓存后秒开）"); st.rerun()
    
    result=st.session_state["ag_result"]
    if result is None: return
    
    cap=result.get("capability",{}); method=result.get("method","")
    st.markdown('<div class="pca-divider"></div>',unsafe_allow_html=True)
    st.markdown("##### 分析结果")
    
    if method=="binomial":
        s=cap.get("sigma",0); c1,c2,c3,c4,c5=st.columns(5)
        c1.markdown(kpi_card("不合格率",f"{cap.get('p_bar',0):.4%}",""),unsafe_allow_html=True)
        c2.markdown(kpi_card("Sigma",f"{s:.2f}",""),unsafe_allow_html=True)
        c3.markdown(kpi_card("PPM",f"{cap.get('ppm',0):,.0f}",""),unsafe_allow_html=True)
        c4.markdown(kpi_card("Zbench",f"{cap.get('z_bench',0):.2f}",""),unsafe_allow_html=True)
        c5.markdown(kpi_card("DPMO",f"{cap.get('dpmo',0):,.0f}",""),unsafe_allow_html=True)
    elif method=="u_chart":
        c1,c2,c3=st.columns(3)
        c1.markdown(kpi_card("单位缺陷",f"{cap.get('u_bar',0):.4f}",""),unsafe_allow_html=True)
        c2.markdown(kpi_card("总缺陷",f"{cap.get('total_defects',0)}",""),unsafe_allow_html=True)
        c3.markdown(kpi_card("批次数",f"{cap.get('n',0)}",""),unsafe_allow_html=True)
    elif result.get("xbar_r"):
        xr=result["xbar_r"]; c1,c2,c3,c4,c5=st.columns(5)
        c1.markdown(kpi_card("Xbarbar",f"{xr['xbarbar']:.4f}",""),unsafe_allow_html=True)
        c2.markdown(kpi_card("Rbar",f"{xr['rbar']:.4f}",""),unsafe_allow_html=True)
        c3.markdown(kpi_card("Xbar UCL",f"{xr['xbar_ucl']:.4f}",""),unsafe_allow_html=True)
        c4.markdown(kpi_card("Xbar LCL",f"{xr['xbar_lcl']:.4f}",""),unsafe_allow_html=True)
        c5.markdown(kpi_card("Cpk" if "cp" in cap else "子组",f"{cap.get('cpk',0):.2f}" if "cp" in cap else str(xr.get('subgroup_size','')),""),unsafe_allow_html=True)
    else:
        c1,c2,c3,c4,c5=st.columns(5)
        c1.markdown(kpi_card("Cp",f"{cap.get('cp',0):.2f}",""),unsafe_allow_html=True)
        c2.markdown(kpi_card("Cpk",f"{cap.get('cpk',0):.2f}",""),unsafe_allow_html=True)
        c3.markdown(kpi_card("Pp",f"{cap.get('pp',0):.2f}",""),unsafe_allow_html=True)
        c4.markdown(kpi_card("Ppk",f"{cap.get('ppk',0):.2f}",""),unsafe_allow_html=True)
        c5.markdown(kpi_card("PPM",f"{cap.get('ppm',0):,.0f}",""),unsafe_allow_html=True)
    
    figs=result.get("figures",{})
    if figs:
        if "main" in figs: st.pyplot(figs["main"]); del figs["main"]
        if figs:
            nm={"capability":"能力图","qq":"Q-Q","boxplot":"箱线","p_chart":"P图","cumulative":"趋势","distribution":"分布","fit":"拟合"}
            tabs=st.tabs([nm.get(k,k) for k in figs])
            for i,k in enumerate(figs): 
                with tabs[i]: st.pyplot(figs[k])
    
    # AI - synchronous with spinner
    st.markdown('<div class="pca-divider"></div>',unsafe_allow_html=True)
    st.markdown("##### AI 智能解读")
    
    if st.session_state.get("ag_ai_text") is None:
        with st.spinner("AI 正在分析..."):
            try:
                ai_text = ai_interpret(structure, result)
                st.session_state["ag_ai_text"] = ai_text
            except Exception as e:
                st.session_state["ag_ai_text"] = "AI分析暂时不可用: " + str(e)
    
    at = st.session_state.get("ag_ai_text", "")
    if at:
        st.markdown(at)
    
    # PDF
    st.markdown('<div class="pca-divider"></div>',unsafe_allow_html=True)
    if st.button("导出 PDF 报告",key="ag_pdf",use_container_width=True):
        with st.spinner("生成PDF..."):
            try:
                pb=generate_agent_pdf(method,result,st.session_state.get("ag_ai_text",""),st.session_state.get("ag_lsl"),st.session_state.get("ag_usl"),st.session_state.get("ag_target"))
                st.session_state["ag_pdf_bytes"]=pb
            except Exception as e: st.error("PDF:"+str(e))
    pb=st.session_state.get("ag_pdf_bytes")
    if pb: st.download_button("下载 PDF",data=pb,file_name=f"SPC_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",mime="application/pdf",key="ag_dl",use_container_width=True)
    st.caption("分析完成 "+datetime.now().strftime("%Y-%m-%d %H:%M"))

agent_page()
