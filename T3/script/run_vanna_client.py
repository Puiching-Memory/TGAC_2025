import requests
import os
from pprint import pprint

SERVER_URL = "http://localhost:8000/api/vanna/v2/chat_poll"
AUTHORIZATION_HEADER = {"Authorization": "admin@example.com"}

def main():
    for task in os.listdir("T3/script/prompt/input/V1"):
        print(f"Processing task: {task}")

        with open(os.path.join("T3/script/prompt/input/V1", task), "r", encoding="utf-8") as f:
            task_prompt = f.read()

        response = requests.post(
            SERVER_URL,
            json={
                "message": task_prompt,
                "conversation_id": f"{task}"
            },
            headers=AUTHORIZATION_HEADER
        )

        pprint(response.json())

        break


if __name__ == "__main__":
	main()
