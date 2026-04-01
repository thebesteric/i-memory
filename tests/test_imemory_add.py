import asyncio
import unittest

from src.imemory import IMemory
from src.memory.memory_models import IMemoryConfig, IMemoryUserIdentity


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
            ("我叫林晓，我最好的朋友苏晴下个月要结婚了，你说我该送她什么礼物呀？", "human"),
            ("你好林晓，先别急～ 你朋友苏晴的婚礼定在什么时候、什么地方呀？", "assistant"),
            ("定在6月18号，在市中心的悦来酒店举行，就因为她和她老公都喜欢这家酒店的装修风格", "human"),
            ("原来是这样，那你和苏晴认识多久啦？为什么想送一份特别的礼物呀？", "assistant"),
            ("我们认识8年了，从大学室友到现在，她是我最亲近的朋友，所以想送一份有心意、不敷衍的礼物", "human"),
            ("太有心啦！那苏晴和她老公平时有什么爱好吗？送贴合他们喜好的礼物会更合适", "assistant"),
            ("她老公喜欢喝茶，苏晴喜欢收集精致的家居摆件，所以我在纠结送茶具还是定制摆件", "human"),
            ("这两个都很合适～ 对了，婚礼当天你会去现场送礼物吗？还是提前送给她？", "assistant"),
            ("我会去现场，打算在婚礼敬酒环节，亲手把礼物交给她，顺便说一句新婚祝福", "human"),
            ("这个方式很有仪式感！那你预算大概多少呀？避免选到超出预算的礼物", "assistant"),
            ("预算大概800-1000元，这个价位既能体现心意，也不会让我有太大经济压力", "human"),
            ("那我给你个建议，定制一套刻字茶具吧，既贴合她老公的爱好，刻上他们的名字和婚礼日期，也有纪念意义", "assistant"),
            ("这个主意太好了！刻字内容就写‘苏晴&陈阳，2026.06.18，新婚快乐’，你觉得怎么样？", "human"),
            ("特别好！既包含了他们的名字，又标注了婚礼日期，还传递了祝福，完全符合你的需求", "assistant"),
            ("那我明天就去网上找定制茶具的店铺，争取在婚礼前一周拿到礼物，避免耽误时间", "human"),
            ("很稳妥～ 记得确认店铺的制作周期，别错过时间，另外可以让商家包装精致一点，更显用心", "assistant"),
            ("好的好的，太谢谢你啦！有你帮忙，我终于不用再纠结送什么礼物了", "human"),
            ("不用客气～ 相信苏晴收到这份定制礼物，一定会感受到你的心意，也提前祝他们新婚幸福", "assistant"),
            ("谢谢！也希望我能顺利拿到礼物，在婚礼上给她一个小惊喜", "human"),
            ("一定会的！如果后续选店铺有疑问，或者想调整刻字内容，随时来问我就好", "assistant"),
            ("好嘞，太感谢啦！对了，婚礼当天悦来酒店的签到时间是上午9点，我得提前半小时到", "human"),
            ("没错，提前到既能避免迟到，也能帮苏晴搭把手，毕竟婚礼当天她会很忙", "assistant"),
            ("嗯嗯，我已经记在备忘录里了，到时候还会带上我的相机，帮他们拍点婚礼现场的照片", "human"),
            ("太贴心了！这些照片以后也是他们珍贵的回忆，苏晴肯定会很感动的", "assistant"),
            ("希望如此吧～ 毕竟我们认识这么多年，看着她找到幸福，我也特别开心", "human"),
            ("能感受到你的心意～ 对了，定制茶具的时候，可以让商家加个小卡片，写上你想对他们说的话", "assistant"),
            ("这个提醒太及时了！我打算写‘愿你们三餐四季，岁岁相依，往后余生，皆是圆满’", "human"),
            ("写得真好，既浪漫又真诚，苏晴看到一定会很暖心", "assistant"),
            ("嘿嘿，借你吉言～ 那我先去选定制店铺啦，有问题再找你", "human"),
            ("好的，祝你顺利选到满意的店铺，也提前祝你朋友婚礼圆满，你们的友谊长长久久", "assistant"),
            ("谢谢啦！回头我把婚礼现场的照片发给你看看", "human"),
            ("太好啦，非常期待！快去选店铺吧，不打扰你啦", "assistant"),
            ("好嘞，再见～", "human"),
            ("再见，有问题随时找我！", "assistant")
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
