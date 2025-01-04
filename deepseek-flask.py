from flask import Flask, request, jsonify
import argparse
import os
import sys
import torch
from threading import Thread
from transformers import TextIteratorStreamer
from PIL import Image
from deepseek_vl.utils.io import load_pretrained_model

# Flask 앱 초기화
app = Flask(__name__)

# 모델 초기화
def initialize_model(model_path, max_gen_len, temperature, top_p, repetition_penalty):
    tokenizer, vl_chat_processor, vl_gpt = load_pretrained_model(model_path)
    generation_config = dict(
        pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
        bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
        eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
        max_new_tokens=max_gen_len,
        use_cache=True,
    )
    if temperature > 0:
        generation_config.update(
            {
                "do_sample": True,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
            }
        )
    else:
        generation_config.update({"do_sample": False})

    return tokenizer, vl_chat_processor, vl_gpt, generation_config


# 모델 로드
MODEL_PATH = "deepseek-ai/deepseek-vl-7b-chat"
MAX_GEN_LEN = 512
TEMPERATURE = 0.2
TOP_P = 0.95
REPETITION_PENALTY = 1.1

tokenizer, vl_chat_processor, vl_gpt, generation_config = initialize_model(
    MODEL_PATH, MAX_GEN_LEN, TEMPERATURE, TOP_P, REPETITION_PENALTY
)


# 이미지 로드 함수
def load_image(image_file):
    return Image.open(image_file).convert("RGB")


# 모델 응답 생성 함수
@torch.inference_mode()
def generate_response(user_input, pil_images):
    # 대화 초기화
    conv = vl_chat_processor.new_chat_template()
    conv.append_message(conv.roles[0], user_input)
    conv.append_message(conv.roles[1], None)

    # 입력 데이터 준비
    prompt = conv.get_prompt()
    prepare_inputs = vl_chat_processor(
        prompt=prompt, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # 이미지 임베딩 생성
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_config["inputs_embeds"] = inputs_embeds
    generation_config["attention_mask"] = prepare_inputs.attention_mask
    generation_config["streamer"] = streamer

    # 생성 스레드 시작
    thread = Thread(target=vl_gpt.language_model.generate, kwargs=generation_config)
    thread.start()

    # 응답 수집
    response = ""
    for char in streamer:
        response += char

    return response


# Flask 라우터
@app.route('/process', methods=['POST'])
def process_request():
    try:
        # JSON 입력 받기
        data = request.json
        text_input = data.get("text", "")
        image_paths = data.get("images", [])

        if not text_input:
            return jsonify({"error": "No input text provided"}), 400

        # 이미지 로드
        pil_images = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                pil_images.append(load_image(img_path))
            else:
                return jsonify({"error": f"Image file not found: {img_path}"}), 400

        # 모델 응답 생성
        response = generate_response(text_input, pil_images)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Flask 서버 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10103)