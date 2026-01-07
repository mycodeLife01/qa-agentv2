import uuid
import asyncio
from starlette.responses import JSONResponse, StreamingResponse
from app.agent import agent, Context
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Feature Functions
def invoke_agent(agent, user_request):
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_request.input}]},
            context=Context(doc_content_hash=user_request.doc_content_hash),
        )
        return response
    except Exception as e:
        print(f"[ERROR]: chat, {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# HTTP Models
class UserRequest(BaseModel):
    input: str
    doc_content_hash: str
    thread_id: str


class AskResponse(BaseModel):
    answer: str


@app.post("/chat")
async def chat(user_request: UserRequest):
    print(
        f"user request, hash: {user_request.doc_content_hash}, thread_id: {user_request.thread_id}, input: {user_request.input}"
    )

    async def event_generator():
        try:
            response = agent.stream(
                {
                    "messages": [{"role": "user", "content": user_request.input}],
                },
                {"configurable": {"thread_id": user_request.thread_id}},
                context=Context(doc_content_hash=user_request.doc_content_hash),
                stream_mode="messages",
            )
            for message, metadata in response:
                if metadata["langgraph_node"] == "model" and message.content:
                    yield f"data: {message.text}\n\n"
                    await asyncio.sleep(0)
        except Exception as e:
            yield f"data: [ERROR]: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
