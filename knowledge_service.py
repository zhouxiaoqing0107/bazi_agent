"""古籍资料、知识库初始化、硬规则检索与 BM25。"""
from pathlib import Path
import json
import chromadb
import dashscope
import jieba
from openai import OpenAI
from lunar_python import Solar
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from retrieval_rules import collect_rule_chunks, resolve_retrieval_periods

api_key = 'sk-ws-H.PEDPEID.Dhp0.MEQCIBv7Qpqy48Tbw6LqM3Uc6H_UoC2J88AeoQLYFRiUzfGAAiABGtTQdWELfPHeQ6rdfhxzS1qLilKaQ7inov1le0ea0Q'

def get_year(year: int) -> str:
    '如果用户提到比如我xx年的运势如何，那么需用用这个工具获取指定公历年份对应的干支年份，如 2026 → 丙午年。"""'
    lunar = Solar.fromYmdHms(year, 3, 1, 0, 0, 0).getLunar()
    return lunar.getYearInGanZhi() + "年"

def get_year_and_month(year: int, month: int) -> str:
    '如果用户提到比如我xx年xx月的运势如何，那么需用用这个工具获取指定公历年份和月份对应的干支年份和月份，如 2026年10月 → 丙午年戊戌月。"""'
    lunar = Solar.fromYmdHms(year, month, 8, 0, 0, 0).getLunar()
    return lunar.getYearInGanZhi() + "年" + lunar.getMonthInGanZhi() + "月"

############################## 事前资料准备 ###############################
r'''
《渊海子平》 
执行命令：使用Codex会话提供的网页工具 web.run，由 OpenAI 托管，Python 脚本不能直接调用这个内部工具。
保存位置：C:/Users/zxq_0/Desktop/bazi_agent/渊海子平评注_公开页面提取.txt

《三命通会》 
执行命令：python C:/Users/zxq_0/Documents/Codex/2026-10-28/ni-ka/outputs/crawl_sanming.py
保存位置：C:/Users/zxq_0/Desktop/bazi_agent/sanmingtonghui
 

《滴天髓》 
保存位置：C:/Users/zxq_0/Desktop/bazi_agent/dixiansui/滴天髓_全文.txt

《子平真诠》 
《穷通宝鉴》 
'''
def get_yuanhaiziping():
    with open('C:/Users/zxq_0/Desktop/bazi_agent/渊海子平评注_公开页面提取.txt', 'r', encoding='utf-8') as file:
        content = file.read()
        return content
 
def get_ditiansui():
    with open('C:/Users/zxq_0/Desktop/bazi_agent/dixiansui/滴天髓_全文.txt', 'r', encoding='utf-8') as file:
        content = file.read()
        return content 

def get_sanmingtonghui():
    return [
        file.read_text(encoding='utf-8')
        for file in sorted(Path('C:/Users/zxq_0/Desktop/bazi_agent/sanmingtonghui').glob('[0-9]*_*.md'))
    ]

 
############################## Hybrid Search - Embedding 向量库 ###############################
## 持久化向量库
# 选择chroma作为向量库
db = chromadb.PersistentClient(path=str(Path.cwd() / "chroma_db"))

# 检索时通过书名选择集合
collections = {
    "渊海子平": db.get_or_create_collection(
        name="yuanhaiziping", embedding_function=None
    ),
    "滴天髓": db.get_or_create_collection(
        name="ditiansui", embedding_function=None
    ),
    "三命通会": db.get_or_create_collection(
        name="sanmingtonghui", embedding_function=None
    ),
}

# 删除指定集合
#db.delete_collection(name="yuanhaiziping")
#db.delete_collection(name="ditiansui")
#db.delete_collection(name="sanmingtonghui")

# 定义embeding模型
embedding_client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    
def embed(texts: list[str]) -> list[list[float]]:
    response = embedding_client.embeddings.create(
        model="text-embedding-v4",
        input=texts,
    )
    return [
        item.embedding
        for item in sorted(response.data, key=lambda item: item.index)
    ]

## 初始化文本分割器
# 使用langchain的递归字符分割器
splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "；", "，", "",],
    )

def init_knowledge():
    for book, collection in collections.items():
        if (collection.metadata or {}).get("ready"):
            continue
        loader = {
            "渊海子平": get_yuanhaiziping,
            "滴天髓": get_ditiansui,
            "三命通会": get_sanmingtonghui,
        }[book]
        records = [
            (str(i), text)
            for i, text in enumerate(loader() if book == "三命通会" else splitter.split_text(loader()))
        ]
        records = [(key, text) for key, text in records if text]
        if not records:
            raise ValueError(f"《{book}》没有可入库的正文")
        # 每批最多 10 条；每条正文分别生成一个向量。
        for i in tqdm(range(0, len(records), 10), desc=book):
            batch = records[i:i + 10]
            texts = [text for _, text in batch]
            collection.upsert(
                ids=[key for key, _ in batch],
                documents=texts,
                embeddings=embed(texts),
                metadatas=[
                    {"book": book, "chapter": key}
                    for key, _ in batch
                ],
            )
        collection.modify(metadata={"ready": True})

# 查看embedding向量库状态
#for book, collection in collections.items():
#    print(book, "条数:", collection.count(), "状态:", collection.metadata) #  {'ready': True}
 
# 查看书籍里的前3条：
if __name__ == "__main__":
    book = "滴天髓"
    data = collections[book].get(
        limit=3,
        include=["documents", "metadatas", "embeddings"],
    )
    for i, key in enumerate(data["ids"]):
        print("ID:", key)
        print("元数据:", data["metadatas"][i])
        print("正文:", data["documents"][i])
        #print("向量维度:", len(data["embeddings"][i]))
        print()

# 硬规则召回 + rerank；不调用 embed 或 collection.query。
def retrieve(query: str, books: list[str] | None = None, top_k: int = 10, *,
             bazi: dict | None = None, question: str | None = None,
             periods: dict | None = None) -> str:
    if bazi is None:
        raise ValueError("硬规则检索需要传入 bazi，不能从自由文本猜测四柱")
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    selected = list(collections) if books is None else list(dict.fromkeys(books))
    periods = periods or resolve_retrieval_periods(question or "")
    chunks, context = collect_rule_chunks(
        {book: collections[book] for book in selected}, bazi, periods,
        get_year, get_year_and_month, question or "",
    )
    if not chunks:
        return "暂无硬规则匹配的参考资料。"
    candidates = [
        f"【《{item['book']}》｜原块ID：{item['id']}｜规则：{'；'.join(item['rules'])}】\n{item['text']}"
        for item in chunks
    ]
    response = dashscope.TextReRank.call(
        api_key=api_key,
        model="qwen3.7-text-rerank",
        query=query + "\n目标岁运：" + json.dumps(context, ensure_ascii=False)
              + "\n请优先选择与完整四柱及问题相符的资料，注意月令、日时和行运条件。",
        documents=candidates,
        top_n=min(top_k, 10, len(candidates)),
        return_documents=False,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Rerank 调用失败：{response.code} - {response.message}")
    ranked = sorted(response.output["results"], key=lambda item: item["relevance_score"], reverse=True)
    return "\n\n".join(candidates[item["index"]] for item in ranked[:min(top_k, 10)])

# 测试
if __name__ == "__main__":
    print(retrieve('癸巳日主', books=['三命通会'], bazi={'四柱': '癸酉 乙丑 癸巳 戊午'}))

############################## Hybrid Search - BM25 关键词检索 ###############################
bm25_index = None # 存储 BM25 索引
bm25_documents = [] # 存储 BM25 对应的原始文档

def init_bm25():
    global bm25_index, bm25_documents
    # 直接从 ChromaDB 读取已存储的文档
    for book, collection in collections.items():
        data = collection.get(
            include=["documents"]
        )
        for doc_id, text in zip(
            data["ids"],
            data["documents"]
        ):
            if not text:
                continue
            bm25_documents.append({
                "id": doc_id,
                "book": book,
                "text": text
            })
    if not bm25_documents:
        raise ValueError("知识库为空，请先执行 init_knowledge()")
    # 中文分词
    tokenized_docs = [
        list(jieba.cut(item["text"]))
        for item in bm25_documents
    ]
    # 构建 BM25 索引
    bm25_index = BM25Okapi(tokenized_docs)
    print(f"BM25 初始化完成，共 {len(bm25_documents)} 个 Chunks")


def bm25_retrieve(
    query: str,
    books: list[str] | None = None,
    top_k: int = 10
):
    if bm25_index is None:
        raise RuntimeError("请先执行 init_bm25()")
    # 中文分词
    query_tokens = list(jieba.cut(query))
    # BM25 打分
    scores = bm25_index.get_scores(query_tokens)
    # 过滤指定书籍
    selected = set(collections) if books is None else set(books)
    ranked_indices = sorted(
        (
            i for i, item in enumerate(bm25_documents)
            if item["book"] in selected
        ),
        key=lambda i: scores[i],
        reverse=True
    ) 
    results = []
    for i in ranked_indices[:top_k]:
        item = bm25_documents[i]
        results.append({
            "id": item["id"],
            "book": item["book"],
            "text": item["text"],
            "score": float(scores[i])
        })
    return results
 
 
