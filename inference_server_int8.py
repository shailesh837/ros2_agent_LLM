#!/usr/bin/env python3
"""
OpenVINO INT8 GPU Inference Server
Provides OpenAI-compatible API for Qwen2.5-Coder-7B-INT8
"""

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from flask import Flask, request, jsonify, Response
import torch
import json
import time
import os

print("=" * 70)
print("Loading INT8 Model on Intel GPU...")
print("=" * 70)

# Model path - adjust if needed
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/spandey2/ros_control_agent/qwen2.5-coder-7b-openvino-int8")

# Load INT8 model on GPU
model = OVModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device="GPU",
    ov_config={
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": "1",
        "CACHE_DIR": "/tmp/ov_cache"
    }
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    trust_remote_code=True
)

print("✅ INT8 Model loaded on GPU!")
print("✅ Tokenizer loaded on CPU!")
print("=" * 70)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """Root endpoint with API information"""
    return jsonify({
        "status": "online",
        "service": "OpenVINO INT8 Inference Server",
        "model": "qwen_coder_int8",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions (POST)"
        },
        "example": {
            "curl": 'curl -X POST http://localhost:9001/v1/chat/completions -H "Content-Type: application/json" -d \'{"messages":[{"role":"user","content":"Hello"}],"stream":false}\''
        }
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "qwen_coder_int8",
        "device": "GPU"
    })

@app.route("/v1/models", methods=["GET"])
def list_models():
    """List available models"""
    return jsonify({
        "object": "list",
        "data": [{
            "id": "qwen_coder_int8",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openvino"
        }]
    })

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint"""
    data = request.json
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    max_tokens = data.get("max_tokens", 2000)
    temperature = data.get("temperature", 0.7)
    
    # Build prompt using chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        return jsonify({"error": f"Prompt building failed: {str(e)}"}), 400
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if stream:
        def generate():
            """Streaming generation"""
            generated_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            past_key_values = None
            
            for i in range(max_tokens):
                try:
                    with torch.no_grad():
                        outputs = model(
                            input_ids=generated_ids[:, -1:] if past_key_values else generated_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                    
                    # Sample next token
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    if temperature > 0:
                        probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Check for EOS
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    
                    # Decode token
                    token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                    
                    # Send chunk
                    if token_text:
                        chunk = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "qwen_coder_int8",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": token_text},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                    # Update for next iteration
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long)], dim=-1)
                    past_key_values = outputs.past_key_values
                
                except Exception as e:
                    error_chunk = {
                        "error": str(e)
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    break
            
            # Send done
            final_chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "qwen_coder_int8",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
    else:
        # Non-streaming
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return jsonify({
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "qwen_coder_int8",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": inputs["input_ids"].shape[1],
                    "completion_tokens": len(outputs[0]) - inputs["input_ids"].shape[1],
                    "total_tokens": len(outputs[0])
                }
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 OpenVINO INT8 GPU Inference Server")
    print("=" * 70)
    print("Server: http://0.0.0.0:9001")
    print("Endpoint: http://0.0.0.0:9001/v1/chat/completions")
    print("Model: INT8 quantization on Intel GPU")
    print("Tokenizer: CPU")
    print("=" * 70 + "\n")
    
    # Get host and port from environment
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "9001"))
    
    app.run(host=host, port=port, threaded=True)
