import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import random
import streamlit as st
import plotly.graph_objects as go

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

# 读取数据
papers_df = pd.read_excel('data3.xlsx', sheet_name='论文')
projects_df = pd.read_excel('data3.xlsx', sheet_name='项目')

# 构建作者—论文网络
G_papers = nx.Graph()
for _, row in papers_df.iterrows():
    authors = [row['姓名']]  # 假设每行只有一个作者
    misconduct = row['不端原因']
    weight = misconduct_weights.get(misconduct, 1)  # 获取不端原因权重，默认为1
    for author in authors:
        G_papers.add_edge(author, row['不端内容'], weight=weight)

# 构建作者—项目网络
G_projects = nx.Graph()
for _, row in projects_df.iterrows():
    authors = [row['姓名']]  # 假设每行只有一个作者
    misconduct = row['不端原因']
    weight = misconduct_weights.get(misconduct, 1)  # 获取不端原因权重，默认为1
    for author in authors:
        G_projects.add_edge(author, row['不端内容'], weight=weight)

# 构建作者—作者网络
G_authors = nx.Graph()
# 1. 共同的项目
for _, row in projects_df.iterrows():
    authors = [row['姓名']]  # 假设每行只有一个作者
    misconduct = row['不端原因']
    weight = misconduct_weights.get(misconduct, 1)  # 获取不端原因权重，默认为1
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            G_authors.add_edge(authors[i], authors[j], weight=weight)
# 2. 共同的论文
for _, row in papers_df.iterrows():
    authors = [row['姓名']]  # 假设每行只有一个作者
    misconduct = row['不端原因']
    weight = misconduct_weights.get(misconduct, 1)  # 获取不端原因权重，默认为1
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            G_authors.add_edge(authors[i], authors[j], weight=weight)
# 3. 共同的研究方向
# 提取每位作者的研究方向
research_areas = papers_df.groupby('姓名')['研究方向'].apply(lambda x: ' '.join(x)).reset_index()
# 使用TF-IDF计算研究方向相似性
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(research_areas['研究方向'])
similarity_matrix = cosine_similarity(tfidf_matrix)
# 添加边（基于研究方向相似性）
threshold = 0.7  # 相似性阈值
for i in range(len(research_areas)):
    for j in range(i + 1, len(research_areas)):
        if similarity_matrix[i, j] > threshold:
            author1 = research_areas.iloc[i]['姓名']
            author2 = research_areas.iloc[j]['姓名']
            G_authors.add_edge(author1, author2, weight=similarity_matrix[i, j])
# 4. 共同的研究机构
institution_dict = papers_df.set_index('姓名')['研究机构'].to_dict()
for author1 in institution_dict:
    for author2 in institution_dict:
        if author1 != author2 and institution_dict[author1] == institution_dict[author2]:
            G_authors.add_edge(author1, author2, weight=1)  # 权重为1表示共同研究机构

# 输出网络信息
print("作者—论文网络信息：")
print(f"Number of nodes: {G_papers.number_of_nodes()}")
print(f"Number of edges: {G_papers.number_of_edges()}")
print("\n作者—项目网络信息：")
print(f"Number of nodes: {G_projects.number_of_nodes()}")
print(f"Number of edges: {G_projects.number_of_edges()}")
print("\n作者—作者网络信息：")
print(f"Number of nodes: {G_authors.number_of_nodes()}")
print(f"Number of edges: {G_authors.number_of_edges()}")

# DeepWalk 实现
def deepwalk(graph, walk_length=30, num_walks=200, embedding_size=64, window_size=10):
    walks = []
    nodes = list(graph.nodes())
    # 生成随机游走序列
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [str(node)]  # 将节点转换为字符串
            current_node = node
            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(current_node))
                if len(neighbors) > 0:
                    current_node = random.choice(neighbors)
                    walk.append(str(current_node))  # 将节点转换为字符串
                else:
                    break
            walks.append(walk)
    # 使用 Word2Vec 训练嵌入
    model = Word2Vec(
        walks,
        vector_size=embedding_size,
        window=window_size,
        min_count=1,
        sg=1,  # 使用 skip-gram
        workers=4
    )
    return model

# 训练 DeepWalk 模型
model = deepwalk(G_authors)
# 提取节点嵌入
embeddings = {node: model.wv[str(node)] for node in G_authors.nodes()}

# 构建训练数据
X = []
y = []
for edge in G_authors.edges():
    X.append(np.concatenate([embeddings[edge[0]], embeddings[edge[1]]]))
    y.append(1)
# 添加负样本（不存在的边）
non_edges = list(nx.non_edges(G_authors))
non_edges_sample = random.sample(non_edges, len(y))  # 保持正负样本平衡
for edge in non_edges_sample:
    X.append(np.concatenate([embeddings[edge[0]], embeddings[edge[1]]]))
    y.append(0)

# 训练分类器
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 评估模型
y_pred = clf.predict_proba(X_test)[:, 1]
auc_roc = roc_auc_score(y_test, y_pred)
auc_pr = average_precision_score(y_test, y_pred)
print(f'AUC-ROC: {auc_roc}, AUC-PR: {auc_pr}')

# 保存结果
authors = list(G_authors.nodes())
risk_scores = [np.linalg.norm(embeddings[author]) for author in authors]  # 使用嵌入向量的 L2 范数作为风险值
result_df = pd.DataFrame({'作者': authors, '风险值': risk_scores})
result_df.to_excel('author_risk_scores4.xlsx', index=False)

# 设置页面标题
st.title("科研人员信用风险预警查询")

# 定义闪烁效果的 CSS
blink_css = """
<style>
@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0; }
    100% { opacity: 1; }
}
.blink {
    animation: blink 1s infinite;
    color: red;
    font-weight: bold;
}
/* 表格样式优化 */
.dataframe {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}
.dataframe th, .dataframe td {
    padding: 8px;
    text-align: left;
    border: 1px solid #ddd;
    max-width: 300px; /* 限制列宽 */
    white-space: normal; /* 允许换行 */
    word-wrap: break-word; /* 允许单词内换行 */
}
.dataframe th {
    background-color: #f2f2f2;
    font-weight: bold;
}
/* 添加滚动条 */
.dataframe-wrapper {
    max-height: 400px; /* 设置最大高度 */
    overflow-y: auto; /* 添加垂直滚动条 */
    margin-bottom: 20px;
}
</style>
"""
# 添加闪烁效果的 CSS
st.markdown(blink_css, unsafe_allow_html=True)

# 添加返回按钮
if st.button("返回主页"):
    st.markdown("[点击这里返回主页](https://chengyi10.wordpress.com/)", unsafe_allow_html=True)

# 输入查询名字
query_name = st.text_input("请输入查询名字：")
if query_name:
    # 在论文表中寻找姓名等于查询输入的名字
    result_paper = papers_df[papers_df['姓名'] == query_name]
    # 在项目表中寻找姓名等于查询输入的名字
    result_project = projects_df[projects_df['姓名'] == query_name]
    # 在风险值表中寻找作者等于查询输入的名字
    result_risk = result_df[result_df['作者'] == query_name]

    # 生成论文查询结果表格
    if not result_paper.empty:
        st.markdown("### 论文查询结果")
        # 将表格转换为 HTML，并添加滚动条
        html_table1 = result_paper.to_html(index=False, escape=False, classes='dataframe')
        st.markdown(f"<div class='dataframe-wrapper'>{html_table1}</div>", unsafe_allow_html=True)

    # 生成项目查询结果表格
    if not result_project.empty:
        st.markdown("### 项目查询结果")
        # 将表格转换为 HTML，并添加滚动条
        html_table2 = result_project.to_html(index=False, escape=False, classes='dataframe')
        st.markdown(f"<div class='dataframe-wrapper'>{html_table2}</div>", unsafe_allow_html=True)

    # 生成风险值查询结果
    if not result_risk.empty:
        st.markdown("### 风险值查询结果")
        risk_value = result_risk.iloc[0]['风险值']
        # 根据风险值显示不同的提示信息
        if risk_value > 2.5:
            st.markdown(f"<p class='blink'>作者: {result_risk.iloc[0]['作者']}, 风险值: {risk_value}（高风险）</p>", unsafe_allow_html=True)
        else:
            st.write(f"作者: {result_risk.iloc[0]['作者']}, 风险值: {risk_value}（低风险）")
    else:
        st.write("暂时没有相关记录。")

    # 构建网络关系图
    if not result_paper.empty or not result_project.empty:
        st.markdown("### 网络关系图")
        # 创建一个空的无向图
        G = nx.Graph()
        # 添加查询作者到图中
        G.add_node(query_name)

        # 查找与查询作者有共同研究机构、研究方向或不端内容的作者
        if not result_paper.empty:
            # 获取查询作者的研究机构、研究方向和不端内容
            research_institution = result_paper.iloc[0]['研究机构']
            research_direction = result_paper.iloc[0]['研究方向']
            misconduct_content = result_paper.iloc[0]['不端内容']
            # 查找与查询作者有共同研究机构、研究方向或不端内容的作者
            related_authors = papers_df[
                (papers_df['研究机构'] == research_institution) |
                (papers_df['研究方向'] == research_direction) |
                (papers_df['不端内容'] == misconduct_content)
            ]
            # 添加相关作者到图中，并建立边
            for _, row in related_authors.iterrows():
                author = row['姓名']
                if author != query_name:
                    G.add_node(author)
                    # 确定边的标签（相连的原因）
                    edge_label = []
                    if row['研究机构'] == research_institution:
                        edge_label.append(f"研究机构: {research_institution}")
                    if row['研究方向'] == research_direction:
                        edge_label.append(f"研究方向: {research_direction}")
                    if row['不端内容'] == misconduct_content:
                        edge_label.append(f"不端内容: {misconduct_content}")
                    edge_label = "\n".join(edge_label)
                    G.add_edge(query_name, author, label=edge_label)

        # 使用 plotly 绘制网络图
        pos = nx.spring_layout(G, k=0.5)  # 布局算法，增加节点间距
        edge_trace = []
        edge_labels = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=0.5, color='#888'),
                hoverinfo='text',
                mode='lines',
                text=edge[2]['label'],  # 边的标签
                hovertext=edge[2]['label']  # 鼠标悬停时显示的文本
            ))
            # 计算边的中点位置，用于显示标签
            edge_labels.append(go.Scatter(
                x=[(x0 + x1) / 2],  # 边的中点
                y=[(y0 + y1) / 2],
                mode='text',
                text=[edge[2]['label']],  # 边的标签
                textposition='middle center',  # 标签位置
                textfont=dict(size=12, color='black'),  # 调整字体大小
                hoverinfo='none'
            ))
        node_trace = go.Scatter(
            x=[], y=[], text=[], mode='markers+text', hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node])
        fig = go.Figure(data=edge_trace + [node_trace] + edge_labels,
                        layout=go.Layout(
                            title='<br>Network graph of related authors',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        st.plotly_chart(fig)
