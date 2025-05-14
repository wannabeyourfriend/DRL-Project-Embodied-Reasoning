import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
# 这一行导入了一个工具脚本，你需要确保 qwen_omni_utils.py 文件和你运行的脚本在同一个目录下
# 或者 qwen_omni_utils.py 在你的 Python 环境可以找到的路径中。
# 你通常可以从 Qwen 官方的 GitHub 仓库或提供该 demo 的地方找到这个文件。
from qwen_omni_utils import process_mm_info
local_model_path = "/cluster/home2/yueyang/wangzixuan/Qwen2.5-Omni-7B"
print(f"Loading model from: {local_model_path}")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    local_model_path, 
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True 
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# 如果你想使用 flash_attention_2，并且你的环境和硬件支持它 (通常需要自己编译安装 flash-attn库):
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     local_model_path, # 使用本地路径
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="flash_attention_2",
#     trust_remote_code=True # 从本地加载自定义代码时，通常需要这个
# )
print("Model loaded successfully.")

print(f"Loading processor from: {local_model_path}")
processor = Qwen2_5OmniProcessor.from_pretrained(
    local_model_path, # 使用本地路径
    trust_remote_code=True # 从本地加载自定义代码时，通常需要这个
)
print("Processor loaded successfully.")
# 修改为使用图片而非视频
conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            # 这里使用一张图片而非视频
            {"type": "image", "image": "/cluster/home2/yueyang/wangzixuan/data/demo/clipboard-image-1747206005.png"},
            {"type": "text", "text": "请描述一下这张图片中的内容。"}
        ],
    },
]

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
# 由于我们不使用视频，所以不需要提取音频
audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs, use_audio_in_video=False, max_new_tokens=512)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)

if audio is not None:
    sf.write(
        "output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )
    print("音频已保存到 output.wav")
else:
    print("模型未生成音频输出")