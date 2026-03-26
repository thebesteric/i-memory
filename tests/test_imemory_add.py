import asyncio
import unittest

from src.imemory import IMemory
from src.memory.models.memory_models import IMemoryConfig, IMemoryUserIdentity


# @unittest.skip
class TestIMemoryAdd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.user_identity: IMemoryUserIdentity = IMemoryUserIdentity(
            user_key="test_user",
            tenant_key="test_tenant",
            project_key="test_project"
        )
        cls.mem = IMemory(user_identity=cls.user_identity)
        cls.meta = {"source": "unit_test"}
        cls.tags = ["test", "memory"]

    @classmethod
    def tearDownClass(cls):
        # asyncio.run(cls.mem.clear(user_identity=cls.user_identity))
        pass

    # @unittest.skip
    def test_add_memory(self):
        contents = [
            ("你好", "human"),
            ("你好呀，有什么可以帮助的么？", "assistant"),
            ("OpenClaw是什么，和普通大模型区别在哪？", "human"),
            ("它是本地部署的AI智能体执行框架，核心是执行任务，不是只聊天。", "assistant"),
            ("支持哪些平台？", "human"),
            ("Windows、macOS、Linux 全平台都能跑。", "assistant"),
            ("交互方式呢？要装专门客户端吗？", "human"),
            ("不用，直接在Telegram、钉钉等50多个主流IM里发指令就行。", "assistant"),
            ("它是模型绑定的吗？", "human"),
            ("完全模型无关，GPT、通义千问、本地Ollama都能接。", "assistant"),
            ("架构大概分几块？", "human"),
            ("核心是Gateway、Agent、Skills、Memory、Security。", "assistant"),
            ("谁开发的？曾用名是什么？", "human"),
            ("奥地利工程师Peter Steinberger，曾用名Clawdbot、Moltbot。", "assistant"),
            ("什么时候开源的？协议是啥？", "human"),
            ("2025年11月正式开源，MIT协议，个人企业都能用。", "assistant"),
            ("核心优势是什么？隐私性强吗？", "human"),
            ("核心是本地运行+真实执行+多渠道接入；数据存本地，还有白名单和沙箱机制。", "assistant"),
            ("适合个人、企业还是开发者？", "human"),
            ("都适合，个人办公、企业自动化、开发辅助全覆盖。", "assistant"),
            ("和传统RPA、AutoGPT有何区别？", "human"),
            ("比传统RPA省流程，比AutoGPT更稳定、轻量、易落地。", "assistant"),
            ("国内能正常用吗？有中文文档吗？", "human"),
            ("能，支持国产模型，官方和社区都有中文教程。", "assistant"),
            ("部署难度大吗？低配电脑能跑吗？", "human"),
            ("有一键脚本，新手半小时搞定，低配电脑也能跑。", "assistant"),
            ("个人用免费吗？有付费版吗？", "human"),
            ("个人开源免费，企业有付费版，提供技术支持。", "assistant"),
            ("有明显缺点吗？", "human"),
            ("复杂逻辑偶尔出错，需要人工确认。", "assistant"),
            ("如何部署和测试？", "human"),
            ("Docker一行命令部署，配置API Key和Token；测试可新建test文件夹。", "assistant"),
            ("进阶玩法有哪些？", "human"),
            ("多Agent协同、自定义技能、定时任务等。", "assistant"),
            ("它的定位是什么？", "human"),
            ("本地开源、模型无关、多渠道、强执行的AI智能体。", "assistant"),
            ("谢谢，我知道了", "human"),
            ("不客气", "assistant"),
        ]

        async def batch_add_memories():
            results = []
            for content in contents:
                res = await self.mem.add(
                    content[0],
                    cfg=IMemoryConfig(force_root=False),
                    meta={**self.meta, "talker": content[1]},
                    tags=self.tags,
                    qa_role=content[1]
                )
                results.append(res)
                print("Memory added:", res)
            return results

        all_results = asyncio.run(batch_add_memories())
        for r in all_results:
            print(r)

    # @unittest.skip
    def test_add_long_memory(self):
        content = """
带着项目推进的期许，奔赴北京与项目组汇合，一场深耕细节、凝聚共识的线下对接，就此展开。不同于线上的远程沟通，面对面的交流更能碰撞出思维的火花，也更能精准捕捉每一个需求的核心，高效破解前期推进中的疑点与难点。

抵达后便迅速投入工作，与项目组的伙伴们围坐一堂，从项目整体规划、阶段性目标，到具体执行细节、潜在风险预案，逐一深入探讨。大家各抒己见，坦诚交流，既有对现有方案的优化建议，也有对关键节点的反复推敲，每一个观点的碰撞，都只为让项目落地更顺畅、成果更优质。
我们一同梳理项目推进中的堵点，明确各岗位职责分工，细化后续执行时间表，针对前期线上沟通中模糊的细节，逐一核对、确认，确保每一项要求都清晰可落地、每一个环节都衔接无疏漏。从需求拆解到流程优化，从进度把控到质量保障，每一个话题都围绕项目核心，每一份努力都朝着共同的目标。
线下对接的时光虽紧凑，却收获满满。不仅高效解决了前期积累的问题，更深化了彼此的协作默契，让跨区域配合更具凝聚力。看着大家为了同一个目标并肩探讨、全力以赴的模样，更坚定了我们把项目做好的信心。

此次北京之行，既是一次工作对接，更是一次思维的同频与能力的提升。未来，我们将带着此次沟通达成的共识，各司其职、紧密配合，稳步推进项目每一个环节，不负信任、不负期许，全力以赴交出满意的项目答卷。
        """
        res = asyncio.run(self.mem.add(content,
                                       cfg=IMemoryConfig(force_root=False),
                                       meta=self.meta,
                                       tags=self.tags))
        print("Memory added:", res)


if __name__ == '__main__':
    unittest.main(verbosity=2)
