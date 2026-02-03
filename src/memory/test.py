from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
import sqlite3
import pickle
import numpy as np

# -------------------------- 全局配置（可根据需求调整） --------------------------
# LSH Hash配置：num_perm越大，Hash精度越高，存储体积稍大（推荐128/256，兼顾精度和效率）
LSH_NUM_PERM = 128
# 数据库文件路径（单文件存储，无需额外配置）
DB_PATH = "text_embedding_hash.db"
# 相似度阈值：筛选候选文本的最低相似度（0.5-0.8为宜，根据场景调整）
SIMILARITY_THRESHOLD = 0.7


# -------------------------- 1. 初始化all-MiniLM-L12-v2模型 --------------------------
def init_model():
    """
    初始化模型（单例加载，首次运行自动下载≈150MB模型文件，后续离线加载）
    返回：预训练模型实例
    """
    model = SentenceTransformer('all-MiniLM-L12-v2')
    # 模型配置：编码时直接归一化向量（核心优化，后续余弦相似度等价于点积）
    model.encode_kwargs = {
        "normalize_embeddings": True,
        "batch_size": 16,
        "convert_to_tensor": False,
        "show_progress_bar": False
    }
    return model


# 全局模型实例（仅初始化一次，提升效率）
model = init_model()


# -------------------------- 2. 初始化SQLite数据库 --------------------------
def init_database():
    """
    初始化数据库，创建文本存储表：texts
    字段：text_id(主键)、text_content(原始文本)、minhash_signature(LSH Hash签名，pickle二进制存储)、embedding_vector(嵌入向量，pickle二进制存储)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 创建表（若不存在）
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS texts
                   (
                       text_id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       text_content
                       TEXT
                       NOT
                       NULL
                       UNIQUE,
                       minhash_signature
                       BLOB
                       NOT
                       NULL,
                       embedding_vector
                       BLOB
                       NOT
                       NULL
                   )
                   ''')
    conn.commit()
    conn.close()
    print(f"数据库初始化完成，文件路径：{DB_PATH}")


# 初始化数据库（运行一次即可）
init_database()


# -------------------------- 3. 文本→向量→LSH Hash签名（核心转换） --------------------------
def text2vec_hash(text: str):
    """
    输入单条文本，完成：模型编码生成768维向量 → MinHash LSH生成Hash签名
    参数：text - 原始文本字符串
    返回：embedding(768维numpy数组)、minhash(MinHash对象，含Hash签名)
    """
    # 步骤1：模型编码生成768维归一化Embedding向量
    embedding = model.encode([text])[0]  # [0]取标量，输出形状：(768,)

    # 步骤2：MinHash LSH生成Hash签名（将高维向量映射为LSH Hash）
    minhash = MinHash(num_perm=LSH_NUM_PERM)
    # 将向量的浮点值转换为可哈希的整数（MinHash默认处理离散特征，此处做连续特征适配）
    # 方法：将浮点值乘以1e6取整，保留精度同时转为整数
    vec_int = (embedding * 1e6).astype(np.int64)
    # 更新MinHash，生成签名
    for val in vec_int:
        minhash.update(val.tobytes())  # 按字节更新，保证哈希唯一性

    return embedding, minhash


# -------------------------- 4. 存储文本（向量+Hash签名）到数据库 --------------------------
def store_text(text: str):
    """
    将文本、其Embedding向量、LSH Hash签名存入数据库
    参数：text - 原始文本字符串
    返回：True(存储成功/已存在)、False(存储失败)
    """
    try:
        # 步骤1：生成向量和Hash签名
        embedding, minhash = text2vec_hash(text)

        # 步骤2：将MinHash对象和向量序列化为二进制（pickle），适配SQLite存储
        minhash_blob = pickle.dumps(minhash)
        embedding_blob = pickle.dumps(embedding)

        # 步骤3：存入数据库（若文本已存在，跳过存储）
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                           INSERT INTO texts (text_content, minhash_signature, embedding_vector)
                           VALUES (?, ?, ?)
                           ''', (text, minhash_blob, embedding_blob))
            conn.commit()
            print(f"文本存储成功：{text[:50]}...")
        except sqlite3.IntegrityError:
            # 文本已存在，无需重复存储
            print(f"文本已存在，跳过存储：{text[:50]}...")
        finally:
            conn.close()
        return True
    except Exception as e:
        print(f"文本存储失败：{e}")
        return False


# -------------------------- 5. 检索最相似文本（核心查询功能） --------------------------
def search_similar_text(query_text: str, top_k: int = 1):
    """
    新语句检索最相似文本：Hash粗筛候选 → 余弦相似度精准排序 → 返回Top-K结果
    参数：
        query_text - 待查询的新语句
        top_k - 返回最相似的文本数量（默认1，即最相似的1条）
    返回：
        若有相似文本：列表，每个元素为(相似文本内容, 余弦相似度值)，按相似度降序排列
        若无相似文本：空列表
    """
    # 步骤1：生成查询文本的向量和Hash签名
    query_embedding, query_minhash = text2vec_hash(query_text)

    # 步骤2：从数据库加载所有文本的Hash签名和向量，初始化LSH索引并粗筛候选
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT text_id, text_content, minhash_signature, embedding_vector FROM texts')
    all_data = cursor.fetchall()
    conn.close()

    if not all_data:
        print("数据库中无文本数据，无法检索")
        return []

    # 初始化LSH索引，用于粗筛相似候选
    lsh = MinHashLSH(num_perm=LSH_NUM_PERM)
    text_dict = {}  # 存储：text_id → (text_content, embedding)
    for text_id, text_content, minhash_blob, embedding_blob in all_data:
        # 反序列化Hash签名和向量
        db_minhash = pickle.loads(minhash_blob)
        db_embedding = pickle.loads(embedding_blob)
        # 将数据库中的Hash签名加入LSH索引
        lsh.insert(text_id, db_minhash)
        # 存储文本内容和向量，便于后续计算相似度
        text_dict[text_id] = (text_content, db_embedding)

    # 步骤3：LSH粗筛候选文本ID（相似性初步匹配）
    candidate_ids = lsh.query(query_minhash)
    if not candidate_ids:
        print("LSH粗筛无候选相似文本")
        return []

    # 步骤4：对候选文本计算余弦相似度，精准排序
    similar_list = []
    for text_id in candidate_ids:
        text_content, db_embedding = text_dict[text_id]
        # 计算余弦相似度（归一化后，点积等价于余弦相似度，更快）
        similarity = np.dot(query_embedding, db_embedding)
        # 过滤低于阈值的结果
        if similarity >= SIMILARITY_THRESHOLD:
            similar_list.append((text_content, round(similarity, 4)))

    # 步骤5：按相似度降序排序，取Top-K
    similar_list.sort(key=lambda x: x[1], reverse=True)
    result = similar_list[:top_k]

    return result


# -------------------------- 测试示例 --------------------------
if __name__ == "__main__":
    # 第一步：批量存储测试文本到数据库（模拟已有文本库）
    test_texts = [
        "人工智能技术正在深刻改变各行各业的发展模式",
        "AI技术对各个行业的发展方式产生了深远的影响",
        "美团打车是一款便捷的手机打车软件，支持多种支付方式",
        "网约车APP美团出行可实现一键叫车，支付方式灵活多样",
        "今天的天气格外晴朗，适合去公园放风筝和野餐",
        "深度学习是人工智能的核心技术，基于神经网络实现特征学习",
        "大语言模型是自然语言处理的重要突破，能理解和生成人类语言"
    ]
    # 批量存储
    for text in test_texts:
        store_text(text)

    # 第二步：测试相似文本检索（3个测试查询）
    query1 = "AI技术深刻影响了各个行业的发展"  # 相似于：人工智能技术正在深刻改变各行各业的发展模式
    query2 = "美团出行的网约车APP支持多种支付方式，叫车很方便"  # 相似于：网约车APP美团出行可实现一键叫车...
    query3 = "今天适合去郊外野餐，天气特别好"  # 相似于：今天的天气格外晴朗，适合去公园放风筝和野餐
    query4 = "Python是一门简洁易用的编程语言"  # 无相似文本

    # 检索并输出结果
    print("\n" + "-" * 50)
    for idx, query in enumerate([query1, query2, query3, query4], 1):
        print(f"\n【查询{idx}】：{query}")
        similar = search_similar_text(query, top_k=1)
        if similar:
            print(f"【最相似文本】：{similar[0][0]}")
            print(f"【余弦相似度】：{similar[0][1]}")
        else:
            print(f"【无符合条件的相似文本】")
    print("\n" + "-" * 50)