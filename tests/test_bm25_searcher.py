import shutil
import tempfile

from agile.search import BM25Searcher
from whoosh.qparser import OrGroup, AndGroup



def test_bm25_searcher_or():
    # 构造测试数据，字段名自定义
    docs = [
        {"docid": 1, "text": "今天早上，我坐公交车上班，看到了彩虹，特别美"},
        {"docid": 2, "text": "周二下雨了，路上有点堵车"},
        {"docid": 3, "text": "我喜欢吃苹果和香蕉"},
        {"docid": 4, "text": "公交车上遇到老同学，聊得很开心"},
    ]
    # 用临时目录，避免污染
    tmp_dir = tempfile.mkdtemp()
    try:
        searcher = BM25Searcher(index_dir=tmp_dir, id_field="docid", content_field="text")

        result = searcher.search("我在公交车上看到了什么", docs=docs, group=OrGroup)
        assert result[:2] == ["1", "4"]

        result = searcher.search("周二 堵车", docs=docs, group=AndGroup)
        assert result[:1] == ["2"]

        result = searcher.search("公交车上遇到谁", docs=docs, group=AndGroup)
        assert result == []
    finally:
        shutil.rmtree(tmp_dir)
