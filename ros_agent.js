let isProcessing = false;

function sendQuick(text) {
    document.getElementById('messageInput').value = text;
    sendMessage();
}

async function sendMessage() {
    if (isProcessing) return;
    
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message) return;
    
    input.value = '';
    isProcessing = true;
    document.getElementById('sendBtn').disabled = true;
    
    addMessage(message, 'user');
    const assistantDiv = addMessage('', 'assistant');
    const contentDiv = assistantDiv.querySelector('.message-content');
    let text = '';
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: message})
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, {stream: true});
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                
                try {
                    const event = JSON.parse(line.slice(6));
                    
                    if (event.type === 'thinking') {
                        contentDiv.innerHTML = '<em>' + event.content + '</em>';
                    } else if (event.type === 'chunk') {
                        text += event.content;
                        contentDiv.innerHTML = formatText(text);
                        scrollToBottom();
                    } else if (event.type === 'error') {
                        contentDiv.innerHTML = '<div style="color: #d32f2f;">❌ Error: ' + event.content + '</div>';
                    }
                } catch (e) {
                    console.error('Parse error:', e);
                }
            }
        }
    } catch (error) {
        contentDiv.innerHTML = '<div style="color: #d32f2f;">❌ Connection error: ' + error.message + '</div>';
    }
    
    isProcessing = false;
    document.getElementById('sendBtn').disabled = false;
}

function addMessage(text, type) {
    const container = document.getElementById('chatMessages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message ' + type;
    messageDiv.innerHTML = 
        '<div class="message-label">' + (type === 'user' ? 'YOU' : 'AGENT') + '</div>' +
        '<div class="message-content">' + (text || 'Processing...') + '</div>';
    
    container.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

function scrollToBottom() {
    const container = document.getElementById('chatMessages');
    container.scrollTop = container.scrollHeight;
}

function formatText(text) {
    let html = text;
    
    // Code blocks
    html = html.replace(/`([^`]+)`/g, '<code style="background:#f0f0f0;padding:2px 6px;border-radius:3px;font-family:monospace;font-size:12px;">$1</code>');
    
    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    
    return html;
}

// Enter key to send
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('messageInput').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Load config and show available actions
    fetch('/config')
        .then(r => r.json())
        .then(data => {
            console.log('Available actions:', data.actions);
        })
        .catch(e => console.error('Config load error:', e));
    
    console.log('ROS Control Agent loaded');
});
