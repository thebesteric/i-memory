import asyncio
import unittest

from src.memory.graph.semantic_split import Dialogue, SemanticSplit, SemanticsOutput


class TestSemanticSplit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dialogues = [
            Dialogue(id="0", content="你好", role="human", created_at="2026-03-23 15:09"),
            Dialogue(id="1", content="OpenClaw是什么，和普通大模型区别在哪？", role="human", created_at="2026-03-23 15:10"),
            Dialogue(id="2", content="它是本地部署的AI智能体执行框架，核心是执行任务，不是只聊天。", role="assistant", created_at="2026-03-23 15:11"),
            Dialogue(id="3", content="支持哪些平台？", role="human", created_at="2026-03-23 15:12"),
            Dialogue(id="4", content="Windows、macOS、Linux 全平台都能跑。", role="assistant", created_at="2026-03-23 15:13"),
            Dialogue(id="5", content="交互方式呢？要装专门客户端吗？", role="human", created_at="2026-03-23 15:14"),
            Dialogue(id="6", content="不用，直接在Telegram、钉钉等50多个主流IM里发指令就行。", role="assistant", created_at="2026-03-23 15:15"),
            Dialogue(id="7", content="它是模型绑定的吗？", role="human", created_at="2026-03-23 15:16"),
            Dialogue(id="8", content="完全模型无关，GPT、通义千问、本地Ollama都能接。", role="assistant", created_at="2026-03-23 15:17"),
            Dialogue(id="9", content="架构大概分几块？", role="human", created_at="2026-03-23 15:18"),
            Dialogue(id="10", content="核心是Gateway、Agent、Skills、Memory、Security。", role="assistant", created_at="2026-03-23 15:19"),
            Dialogue(id="11", content="谁开发的？曾用名是什么？", role="human", created_at="2026-03-23 15:20"),
            Dialogue(id="12", content="奥地利工程师Peter Steinberger，曾用名Clawdbot、Moltbot。", role="assistant", created_at="2026-03-23 15:21"),
            Dialogue(id="13", content="什么时候开源的？协议是啥？", role="human", created_at="2026-03-23 15:22"),
            Dialogue(id="14", content="2025年11月正式开源，MIT协议，个人企业都能用。", role="assistant", created_at="2026-03-23 15:23"),
            Dialogue(id="15", content="核心优势是什么？隐私性强吗？", role="human", created_at="2026-03-23 15:24"),
            Dialogue(id="16", content="核心是本地运行+真实执行+多渠道接入；数据存本地，还有白名单和沙箱机制。", role="assistant", created_at="2026-03-23 15:25"),
            Dialogue(id="17", content="适合个人、企业还是开发者？", role="human", created_at="2026-03-23 15:26"),
            Dialogue(id="18", content="都适合，个人办公、企业自动化、开发辅助全覆盖。", role="assistant", created_at="2026-03-23 15:27"),
            Dialogue(id="19", content="和传统RPA、AutoGPT有何区别？", role="human", created_at="2026-03-23 15:28"),
            Dialogue(id="20", content="比传统RPA省流程，比AutoGPT更稳定、轻量、易落地。", role="assistant", created_at="2026-03-23 15:29"),
            Dialogue(id="21", content="国内能正常用吗？有中文文档吗？", role="human", created_at="2026-03-23 15:30"),
            Dialogue(id="22", content="能，支持国产模型，官方和社区都有中文教程。", role="assistant", created_at="2026-03-23 15:31"),
            Dialogue(id="23", content="部署难度大吗？低配电脑能跑吗？", role="human", created_at="2026-03-23 15:32"),
            Dialogue(id="24", content="有一键脚本，新手半小时搞定，低配电脑也能跑。", role="assistant", created_at="2026-03-23 15:33"),
            Dialogue(id="25", content="个人用免费吗？有付费版吗？", role="human", created_at="2026-03-23 15:34"),
            Dialogue(id="26", content="个人开源免费，企业有付费版，提供技术支持。", role="assistant", created_at="2026-03-23 15:35"),
            Dialogue(id="27", content="有明显缺点吗？", role="human", created_at="2026-03-23 15:36"),
            Dialogue(id="28", content="复杂逻辑偶尔出错，需要人工确认。", role="assistant", created_at="2026-03-23 15:37"),
            Dialogue(id="29", content="如何部署和测试？", role="human", created_at="2026-03-23 15:38"),
            Dialogue(id="30", content="Docker一行命令部署，配置API Key和Token；测试可新建test文件夹。", role="assistant", created_at="2026-03-23 15:39"),
            Dialogue(id="31", content="进阶玩法有哪些？", role="human", created_at="2026-03-23 15:40"),
            Dialogue(id="32", content="多Agent协同、自定义技能、定时任务等。", role="assistant", created_at="2026-03-23 15:41"),
            Dialogue(id="33", content="它的定位是什么？", role="human", created_at="2026-03-23 15:42"),
            Dialogue(id="34", content="本地开源、模型无关、多渠道、强执行的AI智能体。", role="assistant", created_at="2026-03-23 15:43"),
            Dialogue(id="35", content="好的。谢谢", role="human", created_at="2026-03-23 15:44"),
        ]
        cls.semantic_split = SemanticSplit()

    def test_split(self):
        output: SemanticsOutput = asyncio.run(self.semantic_split.invoke(self.dialogues))
        for topic in output.topics:
            print(topic)
