import os
import openai

os.environ["http_proxy"] = "PROXY"
os.environ["https_proxy"] = "PROXY"

from object_query_prompt import PROMPT as object_query_prompt
from visual_programm_prompt import PROMPT as visual_programm_prompt

openai.api_key = "YOUR_API_KEY"
txt_code_name = "visual_programming_prompt/demo/visual_programming_result.txt"
txt_code_name_chatGPT = (
    "visual_programming_prompt/demo/visual_programming_result_chatGPT.txt"
)
curr_prompt = visual_programm_prompt


def insert_task_into_prompt(task, prompt_base, insert_index="INSERT TASK HERE"):
    full_prompt = prompt_base.replace(insert_index, task)
    return full_prompt


demo_task = "Put the tiles round into the yellow and purple polka dot pan."
full_prompt = insert_task_into_prompt(demo_task, curr_prompt)
all_result = []

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=full_prompt,
    temperature=0.99,
    max_tokens=200,
    n=1,
    stop=".",
)

print("For the prompt Completion, the result is: ")
for r in range(len(response["choices"])):
    result = response["choices"][r]["text"]
    all_result.append(result.replace("\n\n", "") + ".")

with open(txt_code_name, "w") as tfile:
    tfile.write("\n".join(all_result))
print(all_result)


# If you have the access to ChatGPT, you can also use the codes below to generate the code.
# all_result_chatGPT = []
# response_chatGPT = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": curr_prompt}
#     ]
# )

# print("For the prompt ChatCompletion, the result is: ")
# for r in range(len(response_chatGPT["choices"])):
#     result = response_chatGPT["choices"][r]["text"]
#     all_result_chatGPT.append(response_chatGPT.replace("\n\n", "") + ".")

# with open(txt_code_name_chatGPT,'w') as tfile:
# 	tfile.write('\n'.join(all_result_chatGPT))
# print(all_result_chatGPT)
