from starlette.responses import JSONResponse
from langchain_core.messages.ai import AIMessage
from app.agent import agent
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class UserRequest(BaseModel):
    input: str
    doc_content_hash: str


@app.post("/chat")
async def chat(user_request: UserRequest) -> str:
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_request.input}]}
        )
        print(response)
        print([msg.pretty_print() for msg in response["messages"]])
        answer = [
            msg.content[0]["text"]
            for msg in response["messages"]
            if isinstance(msg, AIMessage)
            and len(msg.content) > 0
            and msg.content[0].get("type") == "text"
        ][-1]
        return answer
    except Exception as e:
        print(f"[ERROR]: chat, {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
