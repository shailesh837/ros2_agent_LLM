#!/usr/bin/env python3
"""
ROS Control Agent - Natural Language to ROS Commands
Converts natural language to Python script execution and ROS topic publishing
"""

import os
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

import sys
import json
import yaml
import asyncio
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, request, send_from_directory, Response, jsonify
from flask_cors import CORS

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

# Configuration
CONFIG_FILE = "ros_config.yaml"

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
if not os.path.exists(config_path):
    print(f"ERROR: Config file not found: {config_path}")
    sys.exit(1)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

MODEL_URL = config['llm']['model_url']
MODEL_NAME = config['llm']['model_name']

openai_client = AsyncOpenAI(base_url=MODEL_URL, api_key="not-needed")

print("="*70)
print("ROS Control Agent - Natural Language to ROS Commands")
print("="*70)
print(f"LLM: {MODEL_URL} / {MODEL_NAME}")
print(f"Actions loaded: {len(config['actions'])}")
print("="*70)

class AsyncLoopManager:
    def __init__(self):
        self.loop = None
        self.thread = None

    def start(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

loop_manager = AsyncLoopManager()

# Build system prompt from config
def build_system_prompt():
    actions_list = []
    for action, details in config['actions'].items():
        keywords = ", ".join(details['keywords'][:3])  # Show first 3 keywords
        actions_list.append(f"- {action}: {details['description']} (e.g., {keywords})")
    
    actions_text = "\n".join(actions_list)
    
    return f"""You are a ROS Control Agent that converts natural language to robot control commands.

Your job is to identify which action the user wants to execute based on their message.

Available actions:
{actions_text}

CRITICAL RULES:
1. Respond with ONLY the action name: circle, square, dot, cpu, npu, or gpu
2. If multiple actions mentioned, respond with comma-separated list: "circle,square"
3. If no clear action found, respond with: "unknown"
4. Never explain, never add extra text, ONLY the action name(s)

Examples:
"Do a circle" -> circle
"Make it square" -> square
"Show me dots" -> dot
"Use GPU" -> gpu
"Draw circle then square" -> circle,square
"Hello" -> unknown

Respond ONLY with action name(s) or "unknown"."""

SYSTEM_PROMPT = build_system_prompt()

async def identify_action(message: str):
    """Use LLM to identify action from natural language"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message}
    ]
    
    try:
        response = await openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=config['llm']['temperature'],
            max_tokens=config['llm']['max_tokens']
        )
        
        action = response.choices[0].message.content.strip().lower()
        return action
    except Exception as e:
        print(f"LLM error: {e}")
        return "unknown"

def execute_action(action_name: str):
    """Execute the Python script for the given action"""
    if action_name not in config['actions']:
        return {
            "success": False,
            "error": f"Unknown action: {action_name}"
        }
    
    action = config['actions'][action_name]
    script_path = action['script']
    ros_topic = action.get('ros_topic', '')
    ros_message = action.get('ros_message', action_name)
    
    # Log command
    if config['settings'].get('log_commands', True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Action: {action_name} | Topic: {ros_topic} | Message: {ros_message}\n"
        log_file = config['settings'].get('log_file', 'ros_agent.log')
        with open(log_file, 'a') as f:
            f.write(log_entry)
    
    # Simulate or execute
    if config['settings'].get('simulate_ros', True):
        print(f"[SIMULATED] Action: {action_name}")
        print(f"            Topic: {ros_topic}")
        print(f"            Message: {ros_message}")
        print(f"            Script: {script_path}")
        
        return {
            "success": True,
            "action": action_name,
            "description": action['description'],
            "ros_topic": ros_topic,
            "ros_message": ros_message,
            "simulated": True
        }
    
    # Execute real script
    try:
        if not os.path.exists(script_path):
            return {
                "success": False,
                "error": f"Script not found: {script_path}"
            }
        
        result = subprocess.run(
            [sys.executable, script_path, ros_topic, ros_message],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        return {
            "success": result.returncode == 0,
            "action": action_name,
            "description": action['description'],
            "ros_topic": ros_topic,
            "ros_message": ros_message,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "simulated": False
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Script execution timeout"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

async def process_command_stream(message: str):
    """Process natural language command and execute actions"""
    # Step 1: Identify action(s)
    yield f"data: {json.dumps({'type': 'thinking', 'content': 'Analyzing command...'})}\n\n"
    
    action_str = await identify_action(message)
    
    if action_str == "unknown":
        yield f"data: {json.dumps({'type': 'chunk', 'content': '❌ Could not identify action.\\n\\nAvailable commands: circle, square, dot, cpu, npu, gpu'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return
    
    # Handle multiple actions
    actions = [a.strip() for a in action_str.split(',') if a.strip()]
    
    # Step 2: Execute each action
    for action in actions:
        yield f"data: {json.dumps({'type': 'chunk', 'content': f'\\n**Executing: {action.upper()}**\\n'})}\n\n"
        
        result = execute_action(action)
        
        if result['success']:
            mode = "SIMULATED" if result.get('simulated') else "EXECUTED"
            response = f"""✅ {mode}
- Action: **{result['action']}**
- Description: {result['description']}
- ROS Topic: `{result['ros_topic']}`
- Message: `{result['ros_message']}`
"""
            yield f"data: {json.dumps({'type': 'chunk', 'content': response})}\n\n"
        else:
            error_msg = f"❌ Failed: {result.get('error', 'Unknown error')}\n"
            yield f"data: {json.dumps({'type': 'chunk', 'content': error_msg})}\n\n"
    
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

app = Flask(__name__, static_folder='.')
CORS(app)

@app.route("/")
def index():
    return send_from_directory('.', 'ros_agent_simple.html')

@app.route("/ros_agent.js")
def app_js():
    return send_from_directory('.', 'ros_agent.js')

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    
    if not message:
        return {"error": "No message"}, 400
    
    def generate():
        try:
            async_gen = process_command_stream(message)
            while True:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        async_gen.__anext__(),
                        loop_manager.loop
                    )
                    chunk = future.result(timeout=30)
                    yield chunk
                except StopAsyncIteration:
                    break
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                    break
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route("/config", methods=["GET"])
def get_config():
    """Return available actions for UI"""
    actions_info = []
    for action, details in config['actions'].items():
        actions_info.append({
            "name": action,
            "description": details['description'],
            "keywords": details['keywords'][:3]
        })
    return jsonify({"actions": actions_info})

if __name__ == "__main__":
    loop_manager.start()
    
    host = os.getenv("WEBAPP_HOST", "0.0.0.0")
    port = int(os.getenv("WEBAPP_PORT", "5001"))
    
    print(f"\n▶ Open: http://localhost:{port}")
    print(f"▶ Mode: {'SIMULATION' if config['settings'].get('simulate_ros') else 'LIVE ROS'}")
    print("="*70)
    
    app.run(host=host, port=port, debug=False, threaded=True)
