import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import plotly.graph_objects as go

# 设置随机种子
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

# ==========================
# 数据预处理和风险值计算模块
# ==========================
@st.cache_data(show_spinner=False)
def process_risk_data():
    # 不端原因严重性权重
    misconduct_weights = {
        '伪造、篡改图片': 6,
        '篡改图片': 3,
        '篡改数据': 3,
        '篡改数据、图片': 6,
        '编造研究过程': 4,
        '编造研究过程、不当署名': 7,
        '篡改数据、不当署名': 6,
        '伪造通讯作者邮箱': 2,
        '实验流程不规范': 2,
        '数据审核不严': 2,
        '署名不当、实验流程不规范': 5,
        '篡改数据、代写代投、伪造通讯作者邮箱、不当署名': 13,
        '篡改数据、伪造通讯作者邮箱、不当署名': 8,
        '第三方代写、伪造通讯作者邮箱': 7,
        '第三方代写代投、伪造数据': 8,
        '一稿多投': 2,
        '第三方代写代投、伪造数据、一稿多投': 10,
        '篡改数据、剽窃': 8,
        '伪造图片': 3,
        '伪造图片、不当署名': 6,
        '委托实验、不当署名': 6,
        '伪造数据': 3,
        '伪造数据、篡改图片': 6,
        '伪造数据、不当署名、伪造通讯作者邮箱等': 8,
        '伪造数据、一图多用、伪造图片、代投问题': 14,
        '伪造数据、署名不当': 6,
        '抄袭剽窃他人项目申请书内容': 6,
        '伪造通讯作者邮箱、篡改数据和图片': 8,
        '篡改数据、不当署名': 6,
        '抄袭他人基金项目申请书': 6,
        '结题报告中存在虚假信息': 5,
        '抄袭剽窃': 5,
        '造假、抄袭': 5,
        '第三方代写代投': 5,
        '署名不当': 3,
        '第三方代写代投、署名不当': 8,
        '抄袭剽窃、伪造数据': 8,
        '买卖图片数据': 3,
        '买卖数据': 3,
        '买卖论文': 5,
        '买卖论文、不当署名': 8,
        '买卖论文数据': 8,
        '买卖论文数据、不当署名': 11,
        '买卖图片数据、不当署名': 6,
        '图片不当使用、伪造数据': 6,
        '图片不当使用、数据造假、未经同意使用他人署名': 9,
        '图片不当使用、数据造假、未经同意使用他人署名、编造研究过程': 13,
        '图片造假、不当署名': 9,
        '图片造假、不当署名、伪造通讯作者邮箱等': 11,
        '买卖数据、不当署名': 6,
        '伪造论文、不当署名': 6,
        '其他轻微不端行为': 1
    }

    # 读取原始数据
    papers_df = pd.read_excel('data3.xlsx', sheet_name='论文')
    projects_df = pd.read_excel('data3.xlsx', sheet_name='项目')

    # ======================
    # 网络构建函数
    # ======================
    def build_networks(papers, projects):
        # 作者-论文网络
        G_papers = nx.Graph()
        for _, row in papers.iterrows():
            authors = [row['姓名']]
            weight = misconduct_weights.get(row['不端原因'], 1)
            G_papers.add_edge(row['姓名'], row['不端内容'], weight=weight)

        # 作者-项目网络
        G_projects = nx.Graph()
        for _, row in projects.iterrows():
            authors = [row['姓名']]
            weight = misconduct_weights.get(row['不端原因'], 1)
            G_projects.add_edge(row['姓名'], row['不端内容'], weight=weight)

        # 作者-作者网络
        G_authors = nx.Graph()
        
        # 共同项目/论文连接
        for df in [papers, projects]:
            for _, row in df.iterrows():
                authors = [row['姓名']]
                weight = misconduct_weights.get(row['不端原因'], 1)
                for i in range(len(authors)):
                    for j in range(i+1, len(authors)):
                        if G_authors.has_edge(authors[i], authors[j]):
                            G_authors[authors[i]][authors[j]]['weight'] += weight
                        else:
                            G_authors.add_edge(authors[i], authors[j], weight=weight)

        # 研究方向相似性连接
        research_areas = papers.groupby('姓名')['研究方向'].apply(lambda x: ' '.join(x)).reset_index()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(research_areas['研究方向'])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        for i in range(len(research_areas)):
            for j in range(i+1, len(research_areas)):
                if similarity_matrix[i,j] > 0.7:
                    a1 = research_areas.iloc[i]['姓名']
                    a2 = research_areas.iloc[j]['姓名']
                    G_authors.add_edge(a1, a2, weight=similarity_matrix[i,j])

        # 共同机构连接
        institution_map = papers.set_index('姓名')['研究机构'].to_dict()
        for a1 in institution_map:
            for a2 in institution_map:
                if a1 != a2 and institution_map[a1] == institution_map[a2]:
                    G_authors.add_edge(a1, a2, weight=1)
        
        return G_authors

    # ======================
    # 基于随机游走的节点嵌入实现
    # ======================
    def random_walk(graph, start_node, walk_length):
        walk = [start_node]
        for _ in range(walk_length - 1):
            neighbors = list(graph.neighbors(walk[-1]))
            if not neighbors:
                break
            walk.append(random.choice(neighbors))
        return walk

    def generate_walks(graph, num_walks, walk_length):
        walks = []
        nodes = list(graph.nodes())
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(random_walk(graph, node, walk_length))
        return walks

    def node2vec_embedding(graph, num_walks=10, walk_length=80, embedding_size=128):
        walks = generate_walks(graph, num_walks, walk_length)
        # 通过计数词频实现简单的嵌入
        from collections import Counter
        node_freq = Counter([node for walk in walks for node in walk])
        node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
        embeddings = np.zeros((len(graph.nodes()), embedding_size))
        for i, node in enumerate(graph.nodes()):
            embeddings[i] = np.random.normal(size=embedding_size) / np.sqrt(node_freq[node])
        return {node: embeddings[node_to_idx[node]] for node in graph.nodes()}

    # ======================
    # 执行计算流程
    # ======================
    with st.spinner('正在构建合作网络...'):
        G_authors = build_networks(papers_df, projects_df)
    
    with st.spinner('正在生成节点嵌入...'):
        embeddings = node2vec_embedding(G_authors)
    
    with st.spinner('正在计算风险指标...'):
        # 构建分类数据集
        X, y = [], []
        for edge in G_authors.edges():
            X.append(np.concatenate([embeddings[edge[0]], embeddings[edge[1]]]))
            y.append(1)
        
        non_edges = list(nx.non_edges(G_authors))
        non_edges = random.sample(non_edges, len(y))
        for edge in non_edges:
            X.append(np.concatenate([embeddings[edge[0]], embeddings[edge[1]]]))
            y.append(0)
        
        # 训练分类器
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
        clf = RandomForestClassifier(n_estimators=100, random_state=random_seed)
        clf.fit(X_train, y_train)
        
        # 计算节点风险值
        risk_scores = {node: np.linalg.norm(emb) for node, emb in embeddings.items()}
    
    return pd.DataFrame({
        '作者': list(risk_scores.keys()),
        '风险值': list(risk_scores.values())
    }), papers_df, projects_df

# ==========================
# 可视化界面模块
# ==========================
def main():
    st.set_page_config(
        page_title="科研诚信分析平台",
        page_icon="🔬",
        layout="wide"
    )
    
    # 自定义CSS样式
    st.markdown("""
    <style>
    .high-risk { color: red; font-weight: bold; animation: blink 1s infinite; }
    @keyframes blink { 0% {opacity:1;} 50% {opacity:0;} 100% {opacity:1;} }
    .metric-box { padding: 20px; border-radius: 10px; background: #f0f2f6; margin: 10px; }
    </style>
    """, unsafe_allow_html=True)

    # 侧边栏控制面板
    with st.sidebar:
        st.title("控制面板")
        if st.button("🔄 重新计算风险值", help="当原始数据更新后点击此按钮"):
            with st.spinner("重新计算中..."):
                risk_df, papers, projects = process_risk_data()
                risk_df.to_excel('risk_scores.xlsx', index=False)
            st.success("风险值更新完成！")
        
    # 尝试加载现有数据
    try:
        risk_df = pd.read_excel('risk_scores.xlsx')
        papers = pd.read_excel('data3.xlsx', sheet_name='论文')
        projects = pd.read_excel('data3.xlsx', sheet_name='项目')
    except:
        with st.spinner("首次运行需要初始化数据..."):
            risk_df, papers, projects = process_risk_data()
            risk_df.to_excel('risk_scores.xlsx', index=False)

    # 主界面
    st.title("🔍 科研人员信用风险分析系统")
    
    # 搜索框
    search_term = st.text_input("输入研究人员姓名：", placeholder="支持模糊搜索...")
    
    if search_term:
        # 模糊匹配
        candidates = risk_df[risk_df['作者'].str.contains(search_term)]
        if len(candidates) == 0:
            st.warning("未找到匹配的研究人员")
            return
        
        # 选择具体人员
        selected = st.selectbox("请选择具体人员：", candidates['作者'])
        
        # 获取详细信息
        author_risk = risk_df[risk_df['作者'] == selected].iloc[0]['风险值']
        paper_records = papers[papers['姓名'] == selected]
        project_records = projects[projects['姓名'] == selected]
        
        # ======================
        # 信息展示
        # ======================
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 论文记录")
            if not paper_records.empty:
                st.dataframe(paper_records, use_container_width=True)
            else:
                st.info("暂无论文不端记录")

        with col2:
            st.subheader("📋 项目记录")
            if not project_records.empty:
                st.dataframe(project_records, use_container_width=True)
            else:
                st.info("暂无项目不端记录")

        # 风险指标
        st.subheader("📊 风险分析")
        risk_level = "high" if author_risk > 2.5 else "low"
        cols = st.columns(4)
        cols[0].metric("信用评分", f"{author_risk:.2f}", 
                      delta_color="inverse" if risk_level == "high" else "normal")
        cols[1].metric("风险等级", 
                      f"{'⚠️ 高风险' if risk_level == 'high' else '✅ 低风险'}",
                      help="高风险阈值：2.5")
        
        # ======================
        # 关系网络可视化
        # ======================
        with st.expander("🕸️ 展开合作关系网络", expanded=True):
            def build_network_graph(author):
                G = nx.Graph()
                G.add_node(author, size=20, color='red')
                
                # 查找关联节点
                related = papers[
                    (papers['研究机构'] == papers[papers['姓名']==author]['研究机构'].iloc[0]) |
                    (papers['研究方向'] == papers[papers['姓名']==author]['研究方向'].iloc[0])
                ]['姓名'].unique()
                
                for person in related:
                    if person != author:
                        G.add_node(person, size=15, color='blue')
                        G.add_edge(author, person, 
                                  title=f"共同研究方向: {papers[papers['姓名']==person]['研究方向'].iloc[0]}")
                
                # Plotly可视化
                pos = nx.spring_layout(G)
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                node_x = [pos[n][0] for n in G.nodes()]
                node_y = [pos[n][1] for n in G.nodes()]
                
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines'),
                        go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            text=list(G.nodes()),
                            textposition="top center",
                            marker=dict(
                                showscale=True,
                                colorscale='YlGnBu',
                                size=[d['size'] for d in G.nodes.values()],
                                color=[d['color'] for d in G.nodes.values()],
                                line_width=2))
                    ],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=0),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                )
                st.plotly_chart(fig, use_container_width=True)
            
            build_network_graph(selected)

if __name__ == "__main__":
    main()
