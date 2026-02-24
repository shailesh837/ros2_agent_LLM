# 🤖 AI-Powered ROS Control Agent

<div align="center">

![ROS Control Agent](https://img.shields.io/badge/ROS2-Humble-blue)
![OpenVINO](https://img.shields.io/badge/OpenVINO-2024.4.0-green)
![Python](https://img.shields.io/badge/Python-3.11+-yellow)
![License](https://img.shields.io/badge/License-MIT-red)

**Natural Language Command Interface for ROS 2 Robots**

Transform natural language into ROS 2 commands using local LLM inference powered by Intel OpenVINO

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Contributing](#contributing)

</div>

---

## 📸 **Demo**

![ROS Control Agent Demo](screenshots/demo.png)

*Natural language commands being translated to ROS 2 actions in real-time*

---

## ✨ **Features**

- 🗣️ **Natural Language Interface** - Control robots using plain English commands
- 🚀 **Local LLM Inference** - Runs entirely on-device using Intel OpenVINO
- ⚡ **Intel GPU Acceleration** - INT8 quantization for fast inference on Intel integrated GPUs
- 🔄 **Real-time Processing** - Streaming responses with low latency
- 🎯 **Action Library** - Extensible action system for robot behaviors
- 🌐 **Web-Based UI** - Clean, responsive interface accessible from any browser
- 🔌 **ROS 2 Integration** - Direct publishing to ROS topics and services
- 🛠️ **Simulation Mode** - Test commands without real robot hardware
- 📊 **Device Management** - Switch between CPU, GPU, and NPU processing modes

---

## 🏗️ **Architecture**
```
┌─────────────────┐
│   Web Browser   │
│  (User Input)   │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────────────────────────┐
│   Flask Server (Port 5001)          │
│   - Command parsing                 │
│   - Action routing                  │
│   - ROS topic publishing (future)   │
└────────┬────────────────────────────┘
         │ HTTP POST
         ▼
┌─────────────────────────────────────┐
│   OpenVINO Inference (Port 9001)    │
│   - Qwen2.5-Coder-7B-INT8          │
│   - Intel GPU acceleration          │
│   - OpenAI-compatible API           │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Intel GPU (OpenVINO Runtime)      │
│   - Model: 7.1GB INT8 quantized     │
│   - Inference: ~15-20 tokens/sec    │
└─────────────────────────────────────┘
```

---

## 📋 **Requirements**

### **Hardware**
- Intel CPU with integrated GPU (tested on Intel Core Ultra)
- Minimum 8GB RAM (16GB recommended)
- ~10GB disk space for model and dependencies

### **Software**
- Ubuntu 22.04 / 24.04
- Python 3.11+
- ROS 2 Humble (for real robot control)

---

## 🚀 **Installation**

### **1. Clone Repository**
```bash
git clone https://github.com/shailesh837/ros2_agent_LLM.git
cd ros2_agent_LLM
git clone https://huggingface.co/shailesh83/qwen2.5-coder-7b-openvino-int8
```

### **2. Setup Python Environment**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### **3. Install Dependencies**
```bash
# Install PyTorch (CPU-only, no CUDA needed)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cpu

# Install OpenVINO and optimum
pip install transformers==4.45.0
pip install optimum==1.22.0
pip install optimum-intel==1.18.2
pip install openvino==2024.4.0

# Install web server and utilities
pip install Flask==3.0.0 flask-cors==6.0.2
pip install tiktoken==0.8.0 sentencepiece==0.2.0
pip install accelerate==1.1.0 huggingface-hub==0.26.0
pip install requests==2.32.0 numpy==1.26.4 openai
```

## 🎮 **Usage**

### **Start the System (2 Terminals)**

#### **Terminal 1: Start Inference Server**
```bash
cd ros-control-agent
source venv/bin/activate

# Edit inference_server_int8.py and set MODEL_PATH if needed
# MODEL_PATH = "./qwen2.5-coder-7b-openvino-int8"

python inference_server_int8.py
```

**Expected output:**
```
======================================================================
Loading INT8 Model on Intel GPU...
======================================================================
Compiling the model to GPU ...
✅ INT8 Model loaded on GPU!
✅ Tokenizer loaded on CPU!
======================================================================
Server: http://0.0.0.0:9001
======================================================================
```

#### **Terminal 2: Start ROS Agent Server**
```bash
cd ros-control-agent
source venv/bin/activate

python ros_agent_server.py
```

**Expected output:**
```
======================================================================
ROS Control Agent - Natural Language to ROS Commands
======================================================================
LLM: http://localhost:9001/v1 / qwen_coder_int8
Actions loaded: 6
======================================================================
▶ Open: http://localhost:5001
▶ Mode: SIMULATION
======================================================================
```

### **Access Web Interface**

Open browser: **http://localhost:5001**

---

## 💬 **Example Commands**

| Command | Action | ROS Topic |
|---------|--------|-----------|
| "do a circle" | Execute circular motion pattern | `/demo/shape/circle` |
| "make a square" | Execute square motion pattern | `/demo/shape/square` |
| "show dots" | Execute dot pattern | `/demo/shape/dot` |
| "use GPU" | Switch processing to GPU | `/demo/device/mode` |
| "use CPU" | Switch processing to CPU | `/demo/device/mode` |
| "use NPU" | Switch processing to NPU | `/demo/device/mode` |

---

## ⚙️ **Configuration**

### **ros_config.yaml**
```yaml
llm:
  base_url: "http://localhost:9001/v1"
  model: "qwen_coder_int8"
  temperature: 0.7
  max_tokens: 500

actions:
  - name: "circle"
    keywords: ["circle", "round", "circular"]
    script: "scripts/do_circle.py"
    topic: "/demo/shape/circle"
    
  - name: "square"
    keywords: ["square", "box", "rectangle"]
    script: "scripts/do_square.py"
    topic: "/demo/shape/square"
    
  - name: "cpu"
    keywords: ["cpu", "processor"]
    script: "scripts/set_cpu.py"
    topic: "/demo/device/mode"

mode: "SIMULATION"  # or "REAL"
```

---

## 📁 **Project Structure**
```
ros-control-agent/
├── inference_server_int8.py      # OpenVINO inference server
├── ros_agent_server.py            # Flask backend server
├── ros_agent.html                 # Web UI
├── ros_agent.js                   # Frontend JavaScript
├── ros_config.yaml                # Configuration file
├── requirements.txt               # Python dependencies
├── README.md                      # Documentation
├── screenshots/                   # Demo images
│   └── demo.png
├── scripts/                       # Action scripts
│   ├── do_circle.py              # Circle motion
│   ├── do_square.py              # Square motion
│   ├── do_dot.py                 # Dot pattern
│   ├── set_cpu.py                # Set CPU mode
│   ├── set_gpu.py                # Set GPU mode
│   └── set_npu.py                # Set NPU mode
└── qwen2.5-coder-7b-openvino-int8/  # Model files (7.5GB)
    ├── openvino_model.bin
    ├── openvino_model.xml
    └── ...
```

---

## 🔧 **Adding Custom Actions**

### **1. Create Action Script**
```python
# scripts/my_action.py
#!/usr/bin/env python3

def execute():
    print("Executing my custom action")
    # Add your ROS publishing logic here
    return {
        "status": "success",
        "message": "Action completed"
    }

if __name__ == "__main__":
    result = execute()
    print(result)
```

### **2. Add to Configuration**
```yaml
# ros_config.yaml
actions:
  - name: "my_action"
    keywords: ["custom", "my action", "special"]
    script: "scripts/my_action.py"
    topic: "/demo/my_action"
    description: "My custom robot action"
```

### **3. Restart Server**
```bash
python ros_agent_server.py
```

---

## 🐛 **Troubleshooting**

### **Model Loading Issues**
```bash
# Verify OpenVINO installation
python -c "import openvino; print(openvino.__version__)"

# Check GPU availability
python -c "from openvino.runtime import Core; print(Core().available_devices)"
# Should show: ['CPU', 'GPU']
```

### **Performance Issues**
```bash
# Monitor GPU usage
intel_gpu_top

# Check if model using GPU
# Look for "Compiling the model to GPU" in server output
```

### **Network Access Issues**
```bash
# Check firewall
sudo ufw status
sudo ufw allow 5001
sudo ufw allow 9001

# Verify servers running
netstat -tlnp | grep -E '5001|9001'
```

---

## 📊 **Performance**

| Metric | Value |
|--------|-------|
| **Model Size** | 7.5GB (INT8 quantized) |
| **VRAM Usage** | 6-8GB (shared system RAM) |
| **Inference Speed** | 15-20 tokens/sec |
| **Latency** | ~500ms (first token) |
| **Throughput** | ~50ms/token |

*Tested on Intel Core Ultra with integrated GPU*

---

## 🛣️ **Roadmap**

- [ ] Real ROS 2 node integration
- [ ] Voice command support
- [ ] Multi-robot coordination
- [ ] Custom action learning
- [ ] Mobile app interface
- [ ] Docker deployment
- [ ] Cloud deployment option
- [ ] Action history and replay

---

## 🤝 **Contributing**

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📝 **License**

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **[Qwen Team](https://github.com/QwenLM/Qwen)** - Qwen2.5-Coder model
- **[Intel OpenVINO](https://github.com/openvinotoolkit/openvino)** - Inference optimization
- **[Hugging Face](https://huggingface.co/)** - Model hosting and transformers
- **[ROS 2](https://www.ros.org/)** - Robot Operating System

---

## 📧 **Contact**

Project Link: [https://github.com/shailesh837/ros2_agent_LLM](https://github.com/shailesh837/ros2_agent_LLM)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made by [Shailesh Pandey]

</div>
