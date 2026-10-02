import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
# pyrefly: ignore [missing-import]
from context import system_prompt

# pyrefly: ignore [missing-import]
from tools import tools, handle_tool_calls

# pyrefly: ignore [missing-import]
from styles import CSS, JS, EXAMPLES
from dotenv import load_dotenv
import gradio as gr

load_dotenv(override=True)

MODEL_NAME = "openai/gpt-oss-120b"

openai = OpenAI()

system = [{"role": "system", "content": system_prompt}]

def chat(message, history):

    clean_history = []

    for item in history:
        role = item["role"]
        content = item["content"]

        if isinstance(content, list):
            content = "".join(
                part["text"]
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )

        if role in ["user", "assistant"] and isinstance(content, str):
            clean_history.append({
                "role": role,
                "content": content
            })

    messages = (
        [{"role": "system", "content": system_prompt}]
        + clean_history
        + [{"role": "user", "content": message}]
    )

    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools
    )

    while response.choices[0].finish_reason == "tool_calls":

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls

        results = handle_tool_calls(tool_calls)

        messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        messages.extend(results)

        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools
        )

    return response.choices[0].message.content


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        examples=EXAMPLES,
        title="Digital Twin",
        description="Talk to my AI twin about my career",
        chatbot=gr.Chatbot(show_label=False),
    ).launch(css=CSS, js=JS, theme=gr.themes.Base())
