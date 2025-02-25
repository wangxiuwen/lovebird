# 背单词小脚本

## 📌 功能介绍

本脚本用于帮助用户通过 TTS（文本转语音）和 LLM（大语言模型）学习英语单词。它能够：

1. **随机加载单词**：从 `word_list.txt` 读取单词并随机排序。
2. **获取单词信息**：调用 Ollama 或阿里 LLM 获取单词的拼读、释义和例句。
3. **生成语音**：使用 edge-tts 生成英文和中文语音。
4. **播放音频**：合并中英文音频并自动播放。

---

## 📦 安装依赖

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 FFmpeg（用于合并音频）

**Mac 用户**（推荐使用 Homebrew）：

```bash
brew install ffmpeg
```

**Ubuntu/Debian 用户**：

```bash
sudo apt update && sudo apt install ffmpeg -y
```

**Windows 用户**：

1. 下载安装 FFmpeg：https://ffmpeg.org/download.html
2. 配置环境变量，使 `ffmpeg` 命令可用。

---

## 🚀 使用方法

### 1. 准备单词列表

在脚本同级目录创建 `word_list.txt`，每行一个单词，例如：

```
apple
banana
computer
```

### 2. 配置 LLM（可选）

如果使用 **阿里 LLM**，需在 `config.py` 中配置 API 相关信息。

如果使用 **Ollama**，请确保本地 Ollama 服务器运行，并修改 `USE_OLLAMA = True`。

### 3. 运行脚本

```bash
python main.py
```

运行后，程序会：
- 随机选取单词
- 获取拼读、释义和例句
- 生成音频并播放

---

## ❗ 注意事项

1. 确保 `word_list.txt` 存在，否则程序无法运行。
2. 阿里 LLM 需要配置 API 密钥，Ollama 需要本地运行。
3. FFmpeg 需正确安装，否则无法合并音频。

🎉 **Enjoy Learning!** 🚀

