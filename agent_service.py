from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic import BaseModel, field_validator
from datetime import datetime
from pydantic import Field
from typing import List
from pprint import pprint
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.alibaba import AlibabaProvider
from langfuse import get_client
import asyncio
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from langfuse import Langfuse,observe
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.capabilities import Instrumentation
from concurrent.futures import ProcessPoolExecutor
from pydantic_ai.settings import ToolOrOutput
from pydantic_ai import capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior
import pymysql
from pydantic_ai import ModelRetry
import serpapi
from lunar_python import Solar
from bs4 import BeautifulSoup
from typing import Literal
from langfuse.types import MaskOtelSpansResult, OtelSpanPatch
import json
from retrieval_rules import resolve_retrieval_periods
from knowledge_service import (
    api_key, db, collections, embedding_client, splitter,
    get_yuanhaiziping, get_ditiansui, get_sanmingtonghui,
    embed, init_knowledge, retrieve, init_bm25, bm25_retrieve,
    get_year, get_year_and_month,
)

############################## langfuse  ###############################
MAX_LOG_CHARS = 2000

def shorten(value):
    """仅截取日志里的 system 消息和 instructions，其他字段原样保留。"""
    def trim(text):
        if isinstance(text, str) and len(text) > MAX_LOG_CHARS:
            return text[:MAX_LOG_CHARS] + f"...[已截断，原长 {len(text)} 字符]"
        if isinstance(text, list):
            return [trim(item) for item in text]
        if isinstance(text, dict):
            return {key: trim(item) if key in ('text', 'content', 'parts') else item
                    for key, item in text.items()}
        return text

    if isinstance(value, dict):
        system = value.get('role') == 'system' or value.get('part_kind') in ('system-prompt', 'instruction')
        return {
            key: trim(item) if (system and key in ('content', 'text', 'parts'))
            or key in ('instructions', 'system_prompt') else shorten(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [shorten(item) for item in value]
    return value


def mask_otel_spans(*, params):
    patches = {}

    for identifier, span in params.spans.items():
        replacements = {}

        for key, value in span.attributes.items():
            if not isinstance(value, str):
                continue

            # 对 JSON 字段先解析，避免直接截断导致 JSON 损坏
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                changed = (
                    shorten({'instructions': value})['instructions']
                    if key in ('gen_ai.system_instructions', 'system_prompt', 'instructions')
                    else value
                )
            else:
                trimmed = shorten({'instructions': parsed})['instructions'] if key == 'gen_ai.system_instructions' else shorten(parsed)
                changed = (
                    json.dumps(trimmed, ensure_ascii=False)
                    if trimmed != parsed
                    else value
                )

            if changed != value:
                replacements[key] = changed

        if replacements:
            patches[identifier] = OtelSpanPatch(
                set_attributes=replacements
            )

    return MaskOtelSpansResult(span_patches=patches)

langfuse = Langfuse(
    public_key="pk-lf-41ba02c0-7bfa-4751-aba3-79037ad55f04",
    secret_key="sk-lf-916c5f2e-5284-4eb4-bcd8-53a1eb480a31",
    base_url="https://us.cloud.langfuse.com",
    mask_otel_spans=mask_otel_spans,
    timeout=30
)

############################## Bazi Agent ###############################
# 八字排盘：使用输入的公历时间，不做真太阳时校正
class Bazi(BaseModel):
    date: str = Field(description="阳历 YYYY-MM-DD")
    hour: int = Field(ge=0, le=23, description="小时")
    minute: int = Field(ge=0, le=59, description="分钟")
    gender: Literal["男", "女"]
    @field_validator("date")
    @classmethod
    def check_date(cls, value: str) -> str:
        datetime.strptime(value, "%Y-%m-%d")
        return value


def get_bazi(bazi: Bazi):
    birth = datetime.strptime(bazi.date, "%Y-%m-%d").replace(
        hour=bazi.hour, minute=bazi.minute
    )
    solar = Solar.fromYmdHms(
        birth.year, birth.month, birth.day, birth.hour, birth.minute, 0
    )
    eight_char = solar.getLunar().getEightChar()
    yun = eight_char.getYun(1 if bazi.gender == "男" else 0, 2)
    return {
        "性别": bazi.gender,
        "出生时间": solar.toYmdHms(),
        "四柱": eight_char.toString(),
        "起运日期": yun.getStartSolar().toYmdHms(),
        "大运": [
            {
                "干支": item.getGanZhi(),
                "开始年份": item.getStartYear(),
                "结束年份": item.getEndYear(),
            }
            for item in yun.getDaYun()
            if item.getIndex() != 0
        ],
    }

 
# 获取用户提问的年份和月份
def create_chat_model(model_name: str = "qwen3.8-flash"):
    if model_name not in ("qwen3.8-flash", "qwen3.8-max", "deepseek-v4.1-flash"):
        raise ValueError("不支持的模型")
    return OpenAIChatModel(
        model_name,
        provider=AlibabaProvider(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )


model = create_chat_model()

model_settings = {
    "extra_body": {
        "enable_thinking": False,
        "enable_search": False,
        },
    "tool_choice": ToolOrOutput(function_tools=["get_year", "get_year_and_month"]),
    }

class DepType(BaseModel):
    bazi: dict
    gender: Literal["男", "女"]
    target: Literal["整体命局", "事业财运", "婚姻感情", "流年运势", "大运分析"]
    turn: int = 1
    references: str = ""
    
# 启动时读取一次；所有会话共用相同的规则、古籍内容和排列顺序。
SYSTEM_PROMPT = """
保持排盘信息一致，不得自行改算四柱、大运和起运日期。
不要编造客户没有提供的经历。
需要你仔细参考三本古籍进行内部分析
参考资料按文本规则召回并重排，命中标签不代表条文全部条件成立。日时专条须结合完整四柱分析；月令、行运、合化条件不可省略。
回答时只呈现通俗解释
第一轮结合所选主题回答用户的具体问题；后续结合历史对话和本轮问题，不重复首次的完整解读。
不要提到内部实现或资料读取函数名称。
当用户问题中提到年份或月份时，使用工具 get_year, get_year_and_month 获取对应的干支年份和月份，不要自行计算。
"""

# 固定顺序：《渊海子平》→《滴天髓》→《三命通会》。
FIXED_PROMPT = SYSTEM_PROMPT


bazi_agent = Agent(
    model=model,
    tools=[get_year, get_year_and_month],
    model_settings=model_settings,
    capabilities=[Instrumentation()],
    deps_type=DepType,
)


@bazi_agent.instructions
def dynamic_system_prompt(ctx: RunContext[DepType]) -> str:
    deps = ctx.deps
    # 易变的信息全部放在固定前缀之后。
    return FIXED_PROMPT + "\n\n<参考古籍>\n" + deps.references + "\n</参考古籍>\n\n<本次客户资料>\n" + json.dumps(
        {
            "性别": deps.gender,
            "八字排盘": deps.bazi,
            "主题": deps.target,
            "轮次": deps.turn,
        },
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n</本次客户资料>"


async def main():
    # 初始化知识库
    #init_knowledge()  # 首次入库，以后启动直接复用
    history = []
    periods = None
    date, hour, minute, gender = [
        item.strip()
        for item in input(
            "请输入八字排盘信息(阳历 YYYY-MM-DD,小时,分钟,性别):"
        ).split(",")
        ]
    bazi = get_bazi(Bazi(date=date, hour=hour, minute=minute, gender=gender))
    print(bazi)
    deps = DepType(bazi=bazi, gender=gender, target=input("请选择主题(整体命局,事业财运,婚姻感情,流年运势,大运分析):"))
    for turn in range(1, 6):
        question = input(
            "请输入你的问题：" if turn == 1 else f"第 {turn}/5 轮，请继续提问："
        ).strip()
        while not question:
            question = input("问题不能为空，请输入你的问题：").strip()
        deps.turn = turn
        periods = resolve_retrieval_periods(question, periods)
        deps.references = await asyncio.to_thread(
            retrieve, f"{deps.bazi}\n主题：{deps.target}\n问题：{question}", bazi=deps.bazi, question=question, periods=periods
        )

        async with bazi_agent.run_stream(
            user_prompt=question,
            deps=deps,
            message_history=history,
            usage_limits=UsageLimits(request_limit=5),
        ) as result:
            printed = 0
            async for text in result.stream_text():
                print(text[printed:], end="", flush=True)
                printed = len(text)
            print()
            usage = result.usage()
            print("缓存命中 tokens:", usage.cache_read_tokens)
            history = result.all_messages()
        langfuse.flush()
 

if __name__ == "__main__":
    asyncio.run(main())

  

