import asyncio
import os
import edge_tts
import time
import requests
import json
import random
from datetime import datetime
from openai import OpenAI  # 使用 OpenAI 包兼容阿里 LLM
from config import ConfigLoader
from pydub import AudioSegment
from scipy.io.wavfile import read
import sounddevice as sd
import numpy as np

cfg = ConfigLoader().config

# 配置
BASE_DIR = "./"
WORD_LIST_FILE = os.path.join(BASE_DIR, "word_list.txt")  # 单词列表文件
OUTPUT_AUDIO_FILE_EN = os.path.join(BASE_DIR, "word_audio_en.mp3")  # 英文音频文件
OUTPUT_AUDIO_FILE_CN = os.path.join(BASE_DIR, "word_audio_cn.mp3")  # 中文音频文件

# 语音设置
VOICE_EN = "en-US-JennyNeural"  # 英文女声
VOICE_CN = "zh-CN-XiaoxiaoNeural"  # 中文女声
SPEECH_RATE_EN = "-20%"  # 英文降低 20% 语速
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

# 从文件中读取单词并随机打乱
def load_words(file_path):
    if not os.path.exists(file_path):
        print(f"单词文件 {file_path} 不存在！")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    random.shuffle(words)  # 随机打乱单词列表
    return words

# 调用 Ollama 获取单词信息
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

# 调用阿里 LLM 获取单词信息
def call_ali_llm(prompt):
    try:
        client = init_openai_client()
        response = client.chat.completions.create(
            model=cfg.llm.model,
            messages=[
                {"role": "system", "content": "你是一个语言学习助手，提供准确的单词释义和例句。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"阿里 LLM 请求错误：{e}")
        return None

# 获取单词的英文、中文释义和例句
def get_word_info(word):
    prompt = (
        f"请为单词 '{word}' 提供以下信息：\n"
        f"1. 英文单词\n"
        f"2. 英文字母拼读（如 C A T）\n"
        f"3. 中文释义（简洁明了）\n"
        f"4. 一个简短的英文例句（不超过15个单词）\n"
        f"5. 该例句的中文翻译\n"
        f"输出格式如下（直接输出内容，不要多余解释）：\n"
        f"English: [英文单词]\n"
        f"Spelling: [字母拼读]\n"
        f"Chinese: [中文释义]\n"
        f"Example: [英文例句]\n"
        f"Translation: [中文翻译]"
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
        for line in lines:
            if line.startswith("English:"):
                info["english"] = line.replace("English:", "").strip()
            elif line.startswith("Spelling:"):
                info["spelling"] = line.replace("Spelling:", "").strip()
            elif line.startswith("Chinese:"):
                info["chinese"] = line.replace("Chinese:", "").strip()
            elif line.startswith("Example:"):
                info["example"] = line.replace("Example:", "").strip()
            elif line.startswith("Translation:"):
                info["translation"] = line.replace("Translation:", "").strip()
        return info
    except Exception as e:
        print(f"解析单词信息失败：{e}")
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
    words = load_words(WORD_LIST_FILE)
    if not words:
        print("没有加载到单词，程序退出。")
        return

    print(f"共加载 {len(words)} 个单词，开始随机播放...")

    for i, word in enumerate(words, 1):
        print(f"\n📖 第 {i}/{len(words)} 个单词：{word}")

        word_info = get_word_info(word)
        if not word_info:
            print(f"无法获取 {word} 的信息，跳过。")
            continue

        print(f"English: {word_info['english']}")
        print(f"Spelling: {word_info['spelling']}")
        print(f"Chinese: {word_info['chinese']}")
        print(f"Example: {word_info['example']}")
        print(f"Translation: {word_info['translation']}")

        full_english_text = f"{word_info['spelling']}. {word_info['english']}. {word_info['example']}"
        full_chinese_text = f"{word_info['chinese']}. {word_info['translation']}"

        print("生成英文音频...")
        await generate_audio(full_english_text, VOICE_EN, SPEECH_RATE_EN, OUTPUT_AUDIO_FILE_EN)

        print("生成中文音频...")
        await generate_audio(full_chinese_text, VOICE_CN, SPEECH_RATE_CN, OUTPUT_AUDIO_FILE_CN)

        print("播放英文音频...")
        play_audio(OUTPUT_AUDIO_FILE_EN)

        print("播放中文音频...")
        play_audio(OUTPUT_AUDIO_FILE_CN)

        time.sleep(1)

    print("\n🎉 所有单词播放完毕！")

if __name__ == "__main__":
    asyncio.run(main())

