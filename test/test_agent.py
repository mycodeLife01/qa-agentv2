from app.agent import agent


def test_agent_1():
    resp = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Chronos-X 搭载的 Q-7 芯片采用了什么制程工艺？",
                }
            ]
        }
    )
    assert "messages" in resp
    print([msg.pretty_print() for msg in resp["messages"]])
