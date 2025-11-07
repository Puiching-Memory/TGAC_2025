import requests
import os
from pprint import pprint
from colorama import Fore, Style
import re

SERVER_URL = "http://localhost:8000/api/vanna/v2/chat_poll"
AUTHORIZATION_HEADER = {"Authorization": "admin@example.com"}

def main():
    task_list = os.listdir(path="T3/script/prompt/input/V1")
    task_list.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    for task in task_list:
        print(f"{Fore.GREEN}Processing task: {task}{Style.RESET_ALL}")

        with open(os.path.join("T3/script/prompt/input/V1", task), "r", encoding="utf-8") as f:
            task_prompt = f.read()

        # print(f"{Fore.YELLOW}Prompt:{Style.RESET_ALL}\n{task_prompt}\n")

        response = requests.post(
            SERVER_URL,
            json={
                "message": task_prompt,
                "conversation_id": f"{task}"
            },
            headers=AUTHORIZATION_HEADER
        )

        pprint(response.json())
        print("\n" + "="*50 + "\n")

        # break


if __name__ == "__main__":
	main()