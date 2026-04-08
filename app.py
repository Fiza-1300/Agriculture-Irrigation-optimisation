import gradio as gr
import requests
import json

API_URL = "http://localhost:7860"

def reset_env():
    response = requests.post(f"{API_URL}/reset")
    return response.json()

def take_step():
    response = requests.post(f"{API_URL}/step")
    return response.json()

def create_interface():
    with gr.Blocks(title="Irrigation AI") as demo:
        gr.Markdown("# 🌾 AI-Powered Irrigation Optimization")
        gr.Markdown("PPO Reinforcement Learning Agent controlling irrigation")
        
        with gr.Row():
            reset_btn = gr.Button("🔄 Reset Environment")
            step_btn = gr.Button("💧 Take Step")
        
        output = gr.JSON(label="Environment State")
        
        reset_btn.click(fn=reset_env, outputs=output)
        step_btn.click(fn=take_step, outputs=output)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)