"""
记忆动力学模块 - 实现认知心理学中的记忆增强和传播机制

核心理论基础：
1. 扩散激活理论 (Spreading Activation Theory)：记忆检索时通过权重向关联节点传播激活
2. 记忆巩固理论：通过多次激活强化长期记忆
3. 遗忘曲线理论：记忆随时间衰减，分快速和缓慢两个阶段

主要功能：
- 计算不同记忆类型间的相互作用强度（跨扇区共鸣）
- 强化检索到的记忆节点的显著性
- 将强化传播到相关联的记忆节点
"""

from typing import List, Dict

from src.core.dml_ops import dml_ops

# ======================== 学习率和衰减系数 ========================

# 召回强化学习率：每次成功检索记忆时提升显著性的比例
ALPHA_LEARNING_RATE_FOR_RECALL_REINFORCEMENT = 0.15

# 情感频率学习率：情感相关记忆的强化速率更快
BETA_LEARNING_RATE_FOR_EMOTIONAL_FREQUENCY = 0.2

# 图距衰减常数：相距越远的节点，强化传播的衰减越快
GAMMA_ATTENUATION_CONSTANT_FOR_GRAPH_DISTANCE = 0.35

# 长期巩固系数：用于计算长期记忆强化的权重
THETA_CONSOLIDATION_COEFFICIENT_FOR_LONG_TERM = 0.4

# 迹学习强化因子：单次检索对记忆显著性的提升幅度（核心参数）
ETA_REINFORCEMENT_FACTOR_FOR_TRACE_LEARNING = 0.18

# 快速衰减率：短期记忆遗忘速度（高）
# 对应遗忘曲线的陡峭部分（检索后前 24 小时）
LAMBDA_ONE_FAST_DECAY_RATE = 0.015

# 缓慢衰减率：长期记忆遗忘速度（低）
# 对应遗忘曲线的平缓部分（1 周后）
LAMBDA_TWO_SLOW_DECAY_RATE = 0.002

# 检索能量阈值：记忆显著性需超过此值才能被成功激活和检索
# 范围通常为 0.3-0.5，代表有意识接近的概率
TAU_ENERGY_THRESHOLD_FOR_RETRIEVAL = 0.4

# ======================== 跨扇区共鸣矩阵 ========================
#
#  5×5 的对称矩阵,用于表示人脑中 5 种记忆类型之间的相互激活强度（0.0-1.0）
# - 对角线值为 1.0（同类记忆自身相互影响最强）
# - 非对角线值表示不同类型记忆间的激活传播强度
# - 值越大，两种记忆类型的关联度越高
#
# 矩阵结构说明：
# 行/列顺序: [episodic, semantic, procedural, emotional, reflective]
#
# 关键关联强度：
# - semantic(1) ↔ emotional(3) = 0.7：知识与情感强关联（如学习开心的事）
# - semantic(1) ↔ reflective(4) = 0.8：知识与反思最强关联
# - episodic(0) ↔ semantic(1) = 0.7：事件与知识的转化关联
# - procedural(2) ↔ reflective(4) = 0.2：技能与反思关联最弱
#
SECTORAL_INTERDEPENDENCE_MATRIX_FOR_COGNITIVE_RESONANCE = [
    [1.0, 0.7, 0.3, 0.6, 0.6],  # 情景记忆（episodic）：事件、时间、地点信息
    [0.7, 1.0, 0.4, 0.7, 0.8],  # 语义记忆（semantic）：知识、概念、规则
    [0.3, 0.4, 1.0, 0.5, 0.2],  # 程序记忆（procedural）：技能、动作、习惯
    [0.6, 0.7, 0.5, 1.0, 0.8],  # 情感记忆（emotional）：情绪、感受、态度
    [0.6, 0.8, 0.2, 0.8, 1.0],  # 反思记忆（reflective）：元认知、思考、觉悟
]

# 扇区索引映射 - 将记忆类型名称映射到矩阵行列坐标
# 用于快速查询两种记忆类型间的共鸣系数
SECTOR_INDEX_MAPPING_FOR_MATRIX_LOOKUP = {
    "episodic": 0,  # 情景记忆：具体的个人经历和事件（如"去年夏天的旅行"）
    "semantic": 1,  # 语义记忆：一般性的知识和概念（如"巴黎是法国首都"）
    "procedural": 2,  # 程序记忆：如何做某事的知识（如"如何骑自行车"）
    "emotional": 3,  # 情感记忆：与情感相关的记忆（如"失恋的悲伤"）
    "reflective": 4,  # 反思记忆：关于自己想法的记忆（如"我是个完美主义者"）
}


async def calculate_cross_sector_resonance_score(ms: str, qs: str, bs: float) -> float:
    """
    计算跨扇区共鸣分数 - 根据记忆类型间的相互作用强度计算激活程度

    理论基础：扩散激活理论
    - 不同类型的记忆之间存在不同程度的关联
    - 当一个记忆被激活时，会向相关记忆传播激活能量
    - 传播强度取决于两种记忆类型的相互作用系数

    参数：
        ms (str)：源记忆扇区类型，如"semantic"、"emotional"等
        qs (str)：目标记忆扇区类型（被激活的记忆类型）
        bs (float)：基础显著性 (baseline salience)，范围 0.0-1.0
                   表示源记忆的初始激活强度

    返回：
        float：共鸣分数 = bs × 矩阵[源扇区][目标扇区]
               范围 0.0-1.0，表示目标记忆被激活的强度

    执行步骤：
        1. 从映射表查找源扇区在矩阵中的行号（默认为semantic=1）
        2. 从映射表查找目标扇区在矩阵中的列号（默认为semantic=1）
        3. 返回：基础显著性 × 矩阵相关系数

    使用示例：
        # 语义记忆激活情感记忆的共鸣分数
        score = await calculate_cross_sector_resonance_score(
            ms="semantic",      # 源：语义记忆
            qs="emotional",     # 目标：情感记忆
            bs=0.5              # 基础显著性
        )
        # 结果：0.5 × MATRIX[1][3] = 0.5 × 0.7 = 0.35
    """
    # 从映射表获取扇区索引，如果类型不存在则默认使用semantic(1)
    si = SECTOR_INDEX_MAPPING_FOR_MATRIX_LOOKUP.get(ms, 1)
    ti = SECTOR_INDEX_MAPPING_FOR_MATRIX_LOOKUP.get(qs, 1)
    # 返回基础显著性与矩阵系数的乘积
    return bs * SECTORAL_INTERDEPENDENCE_MATRIX_FOR_COGNITIVE_RESONANCE[si][ti]


async def apply_retrieval_trace_reinforcement_to_memory(mid: str, sal: float) -> float:
    """
    应用检索迹强化 - 提升成功检索到的记忆节点的显著性

    理论基础：记忆巩固理论（Consolidation Theory）
    - 每次成功检索一个记忆，会强化该记忆的神经通路
    - 显著性越高，下次检索的成功率越高
    - 但显著性有上限（1.0），不会无限增长

    参数：
        mid (str)：记忆节点ID（当前实现未直接使用，预留扩展接口）
        sal (float)：当前记忆显著性，范围 0.0-1.0
                    0.0 = 完全遗忘，1.0 = 最强记忆

    返回：
        float：强化后的显著性，范围 0.0-1.0

    计算公式：
        新显著性 = min(1.0, 当前显著性 + ETA × (1.0 - 当前显著性))
        其中：
        - ETA = 0.18（迹学习强化因子）
        - (1.0 - sal) = 剩余增长空间
        - 剩余空间越大，增幅越大（凸函数）

    算法特点：
        1. 非线性强化：低显著性的记忆强化幅度大
           例：sal=0.1 → 增幅 = 0.18×0.9 = 0.162（增幅比例为162%）
        2. 高显著性时强化幅度小
           例：sal=0.9 → 增幅 = 0.18×0.1 = 0.018（增幅比例为2%）
        3. 自动收敛到 1.0，且永远达不到 1.0（渐近性质）

    使用示例：
        # 检索一个显著性为 0.5 的记忆
        old_salience = 0.5
        new_salience = await apply_retrieval_trace_reinforcement_to_memory(
            mid="mem_001",
            sal=old_salience
        )
        # 计算：
        min(1.0, 0.5 + 0.18×(1.0-0.5)) = min(1.0, 0.5 + 0.09) = 0.59

        # 多次检索效果
        sal = 0.5
        for i in range(5):
            sal = await apply_retrieval_trace_reinforcement_to_memory("id", sal)
        # 经过 5 次检索：0.5 → 0.59 → 0.664 → 0.725 → 0.774 → 0.815
    """
    # 返回强化后的显著性，不超过 1.0 的上限
    return min(1.0, sal + ETA_REINFORCEMENT_FACTOR_FOR_TRACE_LEARNING * (1.0 - sal))


async def propagate_associative_reinforcement_to_linked_nodes(sid: str, salience: float, wps: List[Dict]) -> List[Dict]:
    """
    将检索强化传播到关联节点 - 实现记忆网络中的扩散激活

    理论基础：扩散激活理论（Spreading Activation Theory）
    - 当一个记忆被激活时，激活能量会沿着关联路径扩散
    - 相邻节点受到的激活强度与连接权重和源节点强度成正比
    - 这解释了为什么想起一件事时，相关的记忆也会被唤醒

    应用场景：
    - 用户检索到记忆 A 后，系统自动激活与 A 关联的 B、C、D 等记忆
    - 权重越高的关联，传播的强化越强
    - 使记忆网络更加活跃，增强记忆间的互联程度

    参数：
        sid (str)：源节点 ID，即被检索激活的记忆节点
        salience (float)：源节点的显著性，范围 0.0-1.0，越高的源节点显著性，对关联节点的强化越强
        wps (List[Dict])：权重路径列表，每个元素是一个字典包含：
                         - "target_id": 目标节点 ID
                         - "weight": 连接权重，范围 0.0-1.0，权重代表两个记忆间的关联强度

    返回：
        List[Dict]：更新列表，每个元素包含：
                   - "node_id": 目标节点 ID
                   - "new_salience": 强化后的显著性
                   只返回成功更新的节点（即在数据库中存在的节点）

    执行步骤详解：

        步骤1：遍历所有权重路径
            for wp in wps:
                tid = wp["target_id"]  # 目标节点ID
                wt = wp["weight"]      # 连接权重

        步骤2：从数据库查询目标节点的当前数据
            ld = dml_ops.get_mem(tid)  # 如果节点不存在，跳过此节点
            if ld: ...

        步骤3：获取目标节点的当前显著性（无则为 0）
            curr = ld["salience"] or 0

        步骤4：计算传播强化值（核心公式）
            pr = ETA × wt × salience
               = 0.18 × 连接权重 × 源显著性

            含义解释：
            - 0.18：基础强化系数
            - wt：权重衰减，权重越小激活越弱
            - salience：源强度倍增，源越强传播越强

        步骤5：计算新显著性
            new_sal = min(1.0, curr + pr)
            确保不超过上限 1.0

        步骤6：记录更新到结果列表
            ups.append({"node_id": tid, "new_salience": new_sal})

    使用示例：

        # 场景：用户检索了"北京"这个记忆
        source_id = "mem_beijing"
        source_salience = 0.8  # 检索强化后变为0.8

        # 与"北京"关联的节点
        weighted_paths = [
            {"target_id": "mem_great_wall", "weight": 0.9},    # 高度关联：长城
            {"target_id": "mem_forbidden_city", "weight": 0.85}, # 高度关联：故宫
            {"target_id": "mem_cn_history", "weight": 0.6},    # 中度关联：中国历史
            {"target_id": "mem_chinese_food", "weight": 0.3},  # 低关联：中餐
        ]

        updates = await propagate_associative_reinforcement_to_linked_nodes(
            sid=source_id,
            salience=source_salience,
            wps=weighted_paths
        )

        # 计算过程：
        # 长城：new_sal = min(1.0, curr + 0.18×0.9×0.8) = min(1.0, curr + 0.1296)
        # 故宫：new_sal = min(1.0, curr + 0.18×0.85×0.8) = min(1.0, curr + 0.1224)
        # 历史：new_sal = min(1.0, curr + 0.18×0.6×0.8) = min(1.0, curr + 0.0864)
        # 中餐：new_sal = min(1.0, curr + 0.18×0.3×0.8) = min(1.0, curr + 0.0432)

        # 返回结果示例：
        # [
        #     {"node_id": "mem_great_wall", "new_salience": 0.5296},
        #     {"node_id": "mem_forbidden_city", "new_salience": 0.4224},
        #     {"node_id": "mem_cn_history", "new_salience": 0.3864},
        #     {"node_id": "mem_chinese_food", "new_salience": 0.2432},
        # ]

    关键特性：
        1. 差异化激活：权重高的关联激活强度大
        2. 衰减传播：距离越远的节点激活越弱
        3. 非线性效果：弱小的关联贡献有限，避免无关节点激活
        4. 自动饱和：显著性不会无限增长，最多达 1.0
    """
    ups = []
    for wp in wps:
        # 获取目标节点信息
        tid = wp["target_id"]  # 目标节点 ID
        wt = wp["weight"]  # 连接权重（0.0-1.0）

        # 从数据库查询目标节点
        ld = dml_ops.get_mem(tid)
        if ld:
            # 获取目标节点的当前显著性
            curr = ld["salience"] or 0

            # 计算传播强化值：基础系数 × 权重 × 源显著性
            # 权重越高、源越强，传播的强化越大
            pr = ETA_REINFORCEMENT_FACTOR_FOR_TRACE_LEARNING * wt * salience

            # 计算新显著性，确保不超过 1.0
            new_sal = min(1.0, curr + pr)

            # 记录更新
            ups.append({"node_id": tid, "new_salience": new_sal})

    return ups
