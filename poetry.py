import asyncio
import os
import edge_tts
import time
import random
from pydub import AudioSegment
from scipy.io.wavfile import read
import sounddevice as sd
import numpy as np
from openai import OpenAI  # 使用 OpenAI 包兼容阿里 LLM
from config import ConfigLoader
import requests

cfg = ConfigLoader().config

# 配置
BASE_DIR = "./"
POETRY_LIST_FILE = os.path.join(BASE_DIR, "poetry_list.txt")  # 古诗词/古文列表文件
OUTPUT_AUDIO_FILE_POETRY = os.path.join(BASE_DIR, "poetry_audio.mp3")  # 古诗词音频文件
OUTPUT_AUDIO_FILE_EXPLANATION = os.path.join(BASE_DIR, "explanation_audio.mp3")  # 讲解音频文件

# 语音设置
VOICE_CN = "zh-CN-XiaoxiaoNeural"  # 中文女声
SPEECH_RATE_CN = "+0%"  # 中文保持默认语速

# 是否使用 Ollama（False 表示使用阿里 LLM）
USE_OLLAMA = False

# Ollama API 配置
OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT = 180  # 超时时间

# 初始化 OpenAI 客户端（用于阿里 LLM）
def init_openai_client():
    return OpenAI(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
    )

# 从文件中读取古诗词并随机打乱
def load_poetry(file_path):
    if not os.path.exists(file_path):
        print(f"古诗词文件 {file_path} 不存在！")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        poetry = [line.strip() for line in f if line.strip()]
    random.shuffle(poetry)  # 随机打乱诗词列表
    return poetry

# 调用 Ollama 获取古诗词讲解
def call_ollama(prompt):
    payload = {
        "model": "qwen2.5:0.5b",
        "prompt": prompt,
        "stream": False,
        "temperature": 0.8,
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"Ollama 请求错误：{e}")
        return None

# 调用阿里 LLM 获取古诗词讲解
def call_ali_llm(prompt):
    try:
        client = init_openai_client()
        response = client.chat.completions.create(
            model=cfg.llm.model,
            messages=[
                {"role": "system", "content": "你是一个古诗词学习助手，提供准确的古诗词释义和讲解。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"阿里 LLM 请求错误：{e}")
        return None

# 获取古诗词的详细信息和讲解
def get_poetry_explanation(poem):
    prompt = (
        f"请为这首古诗词 '{poem}' 提供以下信息：\n"
        f"1. 诗名\n"
        f"2. 作者\n"
        f"3. 简单的背景介绍\n"
        f"4. 逐句解释\n"
        f"5. 整体赏析\n"
        f"6. 诗歌内容\n"
        f"输出格式如下（直接输出内容，不要多余解释）：\n"
        f"Title: [诗名]\n"
        f"Author: [作者]\n"
        f"Background: [背景介绍]\n"
        f"Explanation: [逐句解释]\n"
        f"Appreciation: [整体赏析]\n"
        f"Content: [诗歌内容]"
    )

    if USE_OLLAMA:
        response = call_ollama(prompt)
    else:
        response = call_ali_llm(prompt)

    if not response:
        return None

    try:
        lines = response.split("\n")
        info = {}
        current_section = None
        for line in lines:
            if line.startswith("Title:"):
                info["title"] = line.replace("Title:", "").strip()
                current_section = "title"
            elif line.startswith("Author:"):
                info["author"] = line.replace("Author:", "").strip()
                current_section = "author"
            elif line.startswith("Background:"):
                info["background"] = line.replace("Background:", "").strip()
                current_section = "background"
            elif line.startswith("Explanation:"):
                info["explanation"] = line.replace("Explanation:", "").strip()
                current_section = "explanation"
            elif line.startswith("Appreciation:"):
                info["appreciation"] = line.replace("Appreciation:", "").strip()
                current_section = "appreciation"
            elif line.startswith("Content:"):
                info["content"] = line.replace("Content:", "").strip()
                current_section = "content"
            elif current_section and line.strip():
                info[current_section] += "\n" + line.strip()

        return info
    except Exception as e:
        print(f"解析古诗词信息失败：{e}")
        return None

# 使用 edge-tts 生成音频
async def generate_audio(text, voice, rate, output_file):
    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_file)
        return True
    except Exception as e:
        print(f"生成音频失败：{e}")
        return False

# 播放音频 (使用 sounddevice 和 numpy)
def play_audio(file_path):
    if not os.path.exists(file_path):
        print(f"音频文件未找到：{file_path}")
        return

    # 将 MP3 文件转换为 WAV 格式
    wav_file = file_path.replace(".mp3", ".wav")
    audio = AudioSegment.from_file(file_path, format="mp3")
    audio.export(wav_file, format="wav")

    # 使用 sounddevice 播放 WAV 文件
    sample_rate, data = read(wav_file)
    sd.play(data, sample_rate)
    sd.wait()

    # 删除临时 WAV 文件
    os.remove(wav_file)

# 主函数
async def main():
    poetry_list = load_poetry(POETRY_LIST_FILE)
    if not poetry_list:
        print("没有加载到古诗词，程序退出。")
        return

    print(f"共加载 {len(poetry_list)} 首古诗词，开始随机播放...")

    for i, poem in enumerate(poetry_list, 1):
        print(f"\n📖 第 {i}/{len(poetry_list)} 首古诗词：\n{poem}")

        explanation = get_poetry_explanation(poem)
        if not explanation:
            print(f"无法获取 {poem} 的讲解信息，跳过。")
            continue

        print(f"Title: {explanation['title']}")
        print(f"Author: {explanation['author']}")
        print(f"Background: {explanation['background']}")
        print(f"Explanation: {explanation['explanation']}")
        print(f"Appreciation: {explanation['appreciation']}")
        print(f"Content: {explanation['content']}")

        full_poetry_text = f"{explanation['title']}\n{explanation['author']}\n{explanation['content']}"
        full_explanation_text = f"{explanation['background']}\n{explanation['explanation']}\n{explanation['appreciation']}"

        print("生成古诗词音频...")
        await generate_audio(full_poetry_text, VOICE_CN, SPEECH_RATE_CN, OUTPUT_AUDIO_FILE_POETRY)

        print("生成讲解音频...")
        await generate_audio(full_explanation_text, VOICE_CN, SPEECH_RATE_CN, OUTPUT_AUDIO_FILE_EXPLANATION)

        print("播放古诗词音频...")
        play_audio(OUTPUT_AUDIO_FILE_POETRY)

        print("播放讲解音频...")
        play_audio(OUTPUT_AUDIO_FILE_EXPLANATION)

        time.sleep(1)

    print("\n🎉 所有古诗词播放完毕！")

if __name__ == "__main__":
    asyncio.run(main())
