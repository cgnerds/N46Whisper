file_names = ["G:/AIGC/ssni-658/ssni-658-2.mp4"]

# **通用参数**
# 选择上传的文件类型(视频-video/音频-audio）
file_type = "video"  # @param ["audio","video"]

# 模型大小将影响转录时间和质量, 默认使用最新发布的large-v2模型以节省试错时间
# 默认识别语言为日语，若使用其它语言的视频请自行输入即可。请注意：使用两字母语言代码如'en'，'ja'
# 请注意：large-v2在某些情况下可能未必优于large-v1，请用户自行选择
model_size = "base"  # @param ["base","small","medium", "large-v1","large-v2"]
language = "ja"  # @param {type:"string"}

# 默认只导出ass，若需要srt则选择Yes
# 导出时浏览器会弹出允许同时下载多个文件的请求，需要同意
export_srt = "No"  # @param ["No","Yes"]

# **其他选项**

# 将存在空格的单行文本分割为多行（多句）。分割后的若干行均临时采用相同时间戳，且添加了adjust_required标记提示调整时间戳避免叠轴
# 普通分割（Modest): 当空格后的文本长度超过5个字符，则另起一行
# 全部分割（Aggressive): 只要遇到空格即另起一行
is_split = "Yes"  # @param ["No","Yes"]
split_method = "Modest"  # @param ["Modest","Aggressive"]

# 当前支持生成字幕格式：
# -- ikedaCN - 特蕾纱熊猫观察会字幕组
# -- sugawaraCN - 坂上之月字幕组
# -- kaedeCN - 三番目の枫字幕组
# -- taniguchiCN - 泪痣愛季応援団
# -- asukaCN - 暗鳥其实很甜字幕组
sub_style = "default"  # @param ["default", "ikedaCN", "kaedeCN","sugawaraCN","taniguchiCN","asukaCN"]

# **使用VAD过滤**

# 使用[Silero VAD model](https://github.com/snakers4/silero-vad)以检测并过滤音频中的无声段落（推荐小语种使用）
# 【注意】使用VAD filter有优点亦有缺点，请用户自行根据音频内容决定是否启用. [关于VAD filter](https://github.com/Ayanaminn/N46Whisper/blob/main/FAQ.md)

is_vad_filter = "False" # @param ["True", "False"]
# @markdown  <font size="2"> *  The default <font size="3">  ```min_silence_duration``` <font size="2"> is set at 1000 ms in N46Whisper

print('语音识别库配置完毕，将开始转换')
import os
import ffmpeg
import subprocess
import torch
from faster_whisper import WhisperModel
from tqdm import tqdm
import time
import pandas as pd
import requests
from urllib.parse import quote_plus
from pathlib import Path
import sys
import pysubs2
import gc

# assert file_name != ""
# assert language != ""
file_basenames = ["G:/AIGC/ssni-658/ssni-658-2"]


torch.cuda.empty_cache()
print('加载模型 Loading model...')
model = WhisperModel(model_size)

for i in range(len(file_names)):
  file_name = file_names[i]
  #Transcribe
  file_basename = file_basenames[i]
  if file_type == "video":
    print('提取音频中 Extracting audio from video file...')
    os.system(f'ffmpeg -i {file_name} -f mp3 -ab 192000 -vn {file_basename}.mp3')
    print('提取完毕 Done.') 
  # print(file_basename)
  tic = time.time()
  print('识别中 Transcribe in progress...')
  segments, info = model.transcribe(audio = f'{file_name}',
                                      beam_size=5,
                                      language=language,
                                      vad_filter=is_vad_filter,
                                      vad_parameters=dict(min_silence_duration_ms=1000))
  
  # segments is a generator so the transcription only starts when you iterate over it
  # to use pysubs2, the argument must be a segment list-of-dicts
  total_duration = round(info.duration, 2)  # Same precision as the Whisper timestamps.
  results= []
  with tqdm(total=total_duration, unit=" seconds") as pbar:
      for s in segments:
          segment_dict = {'start':s.start,'end':s.end,'text':s.text}
          results.append(segment_dict)
          segment_duration = s.end - s.start
          pbar.update(segment_duration)


  #Time comsumed
  toc = time.time()
  print('识别完毕 Done')
  print(f'Time consumpution {toc-tic}s')

  subs = pysubs2.load_from_whisper(results)
  subs.save(file_basename+'.srt')

  from srt2ass import srt2ass
  ass_sub = srt2ass(file_basename + ".srt", sub_style, is_split,split_method)
  print('ASS subtitle saved as: ' + ass_sub)
  # files.download(ass_sub)

  # if export_srt == 'Yes':
  #   files.download(file_basename+'.srt')

  print('第',i+1,'个文件字幕生成完毕/',i+1, 'file(s) was completed!')
  torch.cuda.empty_cache()
print('所有字幕生成完毕 All done!')

#@title **【实验功能】Experimental Features:**

# @markdown **AI文本翻译/AI Translation:**
# @markdown **</br>**<font size="2"> 此功能允许用户使用AI翻译服务对识别的字幕文件做逐行翻译，并以相同的格式生成双语对照字幕。
# @markdown **</br>**阅读项目文档以了解更多。</font>
# @markdown **</br>**<font size="2"> This feature allow users to translate previously transcribed subtitle text line by line using AI translation.
# @markdown **</br>**Then generate bilingual subtitle files in same sub style.Read documentaion to learn more.</font>

# @markdown **</br>**希望在本地使用字幕翻译功能的用户，推荐尝试 [subtitle-translator-electron](https://github.com/gnehs/subtitle-translator-electron)

# @markdown **</br><font size="3">Select subtitle file source</br>
# @markdown <font size="3">选择字幕文件(使用上一步的转录-use_transcribed/新上传-upload_new）</br>**
# @markdown <font size="2">支持SRT与ASS文件
sub_source = "use_transcribed"  # @param ["use_transcribed","upload_new"]

# @markdown **chatGPT:**
# @markdown **</br>**<font size="2"> 要使用chatGPT翻译，请填入你自己的OpenAI API Key，目标语言，输出类型，然后执行单元格。</font>
# @markdown **</br>**<font size="2"> Please input your own OpenAI API Key, then execute this cell.</font>
# @markdown **</br>**<font size="2">【注意】 免费的API对速度有所限制，需要较长时间，用户可以自行考虑付费方案。</font>
# @markdown **</br>**<font size="2">【Note】There are limitaions on usage for free API, consider paid plan to speed up.</font>
openai_key = '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' # @param {type:"string"}
target_language = 'zh-hans'# @param ["zh-hans","english"]
prompt = "You are a language expert.Your task is to translate the input subtitle text, sentence by sentence, into the user specified target language.However, please utilize the context to improve the accuracy and quality of translation.Please be aware that the input text could contain typos and grammar mistakes, utilize the context to correct the translation.Please return only translated content and do not include the origin text.Please do not use any punctuation around the returned text.Please do not translate people's name and leave it as original language.\"" # @param {type:"string"}
temperature = 0.6 #@param {type:"slider", min:0, max:1.0, step:0.1}
# @markdown <font size="4">Default prompt: </br>
# @markdown ```You are a language expert.```</br>
# @markdown ```Your task is to translate the input subtitle text, sentence by sentence, into the user specified target language.```</br>
# @markdown ```Please utilize the context to improve the accuracy and quality of translation.```</br>
# @markdown ```Please be aware that the input text could contain typos and grammar mistakes, utilize the context to correct the translation.```</br>
# @markdown ```Please return only translated content and do not include the origin text.```</br>
# @markdown ```Please do not use any punctuation around the returned text.```</br>
# @markdown ```Please do not translate people's name and leave it as original language.```</br>
output_format = "ass"  # @param ["ass","srt"]

import sys
import os
import re
import time
import codecs
import regex as re
from pathlib import Path
from tqdm import tqdm
# from google.colab import files
# from IPython.display import clear_output 

clear_output()

# if sub_source == 'upload_new':
#   uploaded = files.upload()
#   sub_name = list(uploaded.keys())[0]
#   sub_basename = Path(sub_name).stem
if sub_source == 'use_transcribed':
  sub_name = file_basenames[0] +'.ass'
  sub_basename = file_basenames[0]

# !pip install openai
# !pip install pysubs2
import openai
import pysubs2

clear_output()

class ChatGPTAPI():
    def __init__(self, key, language, prompt, temperature):
        self.key = key
        # self.keys = itertools.cycle(key.split(","))
        self.language = language
        self.key_len = len(key.split(","))
        self. prompt = prompt
        self.temperature = temperature


    # def rotate_key(self):
    #     openai.api_key = next(self.keys)

    def translate(self, text):
        # print(text)
        # self.rotate_key()
        openai.api_key = self.key
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        # english prompt here to save tokens
                        "content": f'{self.prompt}'
                    },
                    {
                        "role":"user",
                        "content": f"Original text:`{text}`. Target language: {self.language}"
                    }
                ],
                temperature=self.temperature
            )
            t_text = (
                completion["choices"][0]
                .get("message")
                .get("content")
                .encode("utf8")
                .decode()
            )
            total_tokens = completion['usage']['total_tokens'] # include prompt_tokens and completion_tokens
        except Exception as e:
            # TIME LIMIT for open api , pay to reduce the waiting time
            sleep_time = int(60 / self.key_len)
            time.sleep(sleep_time)
            print(e, f"will sleep  {sleep_time} seconds")
            # self.rotate_key()
            openai.api_key = self.key
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f'{self.prompt}'
                    },
                    {
                        "role": "user",
                        "content": f"Original text:`{text}`. Target language: {self.language}"
                    }
                ],
                temperature=self.temperature
            )
            t_text = (
                completion["choices"][0]
                .get("message")
                .get("content")
                .encode("utf8")
                .decode()
            )
        total_tokens = completion['usage']['total_tokens']
        return t_text, total_tokens


class SubtitleTranslator():
    def __init__(self, sub_src, model, key, language, prompt,temperature):
        self.sub_src = sub_src
        self.translate_model = model(key, language,prompt,temperature)
        self.translations = []
        self.total_tokens = 0

    def calculate_price(self,num_tokens):
        price_per_token = 0.000002 #gpt-3.5-turbo	$0.002 / 1K tokens
        return num_tokens * price_per_token

    def translate_by_line(self):
        sub_trans = pysubs2.load(self.sub_src)
        total_lines = len(sub_trans)
        for line in tqdm(sub_trans,total = total_lines):
            line_trans, tokens_per_task = self.translate_model.translate(line.text)
            line.text += (r'\N'+ line_trans)
            print(line_trans)
            self.translations.append(line_trans)
            self.total_tokens += tokens_per_task

        return sub_trans, self.translations, self.total_tokens


clear_output()

translate_model = ChatGPTAPI

assert translate_model is not None, "unsupported model"
OPENAI_API_KEY = openai_key

if not OPENAI_API_KEY:
    raise Exception(
        "OpenAI API key not provided, please google how to obtain it"
    )
# else:
#     OPENAI_API_KEY = openai_key

t = SubtitleTranslator(
    sub_src=sub_name,
    model= translate_model,
    key = OPENAI_API_KEY,
    language=target_language,
    prompt=prompt,
    temperature=temperature)

translation, _, total_token = t.translate_by_line()
total_price = t.calculate_price(total_token)
#Download ass file

if output_format == 'ass':
  translation.save(sub_basename + '_translation.ass')
#   files.download(sub_basename + '_translation.ass')
elif output_format == 'srt':
  translation.save(sub_basename + '_translation.srt')
#   files.download(sub_basename + '_translation.srt')



print('双语字幕生成完毕 All done!')
print(f"Total number of tokens used: {total_token}")
print(f"Total price (USD): ${total_price:.4f}")