from querydata import query_rag
# from langchain_community.llms import NLPCloud
# llm = NLPCloud(nlpcloud_api_key="e4c513eeddeda22f6e014b7a39c1124b82c9f668")
from dotenv import load_dotenv
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import os
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm = HuggingFaceHub(repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1',model_kwargs={'temprature':0.5, 'max_length':500})

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_1():
    assert query_and_validate(
        question="Có bao nhiêu loại chỉ nha khoa? (chỉ đưa ra con số)",
        expected_response="2 loại",
    )


def test_2():
    assert query_and_validate(
        question="Nên kiểm tra sức khỏe răng miệng bao nhiêu tháng 1 lần?",
        expected_response="6 tháng 1 lần",
    )
def test_3():
    assert query_and_validate(
        question="Giới thiệu tóm tắt về nha khoa BestSmile?",
        expected_response="Nha khoa BestSmile ra đời với sứ mệnh mang đến những giá trị cốt lõi cho khách hàng là làm đẹp nụ cười – thay đổi cuộc sống. Đây là hệ thống chuỗi nha khoa tiêu chuẩn Pháp đầu tiên tại Việt Nam quy tụ đội ngũ bác sĩ – chuyên gia nha khoa hàng đầu, đồng thời luôn tiên phong trong việc ứng dụng công nghệ nha khoa tân tiến nhất trên thế giới, đem đến những trải nghiệm khác biệt cho khách hàng trên mọi miền đất nước."
    )
def test_4():
    assert query_and_validate(
    question="Nguyên nhân gây viêm lợi là gì?",
    expected_response="Có nhiều nguyên nhân gây bệnh viêm lợi như: Nghiện rượu, thuốc lá, ăn nhiều đồ ngoạt, cay,..."
)
def test_5():
    assert query_and_validate(
    question="Sứ mệnh của nha khoa BestSmile?",
    expected_response="Nha khoa BestSmile ra đời với sứ mệnh mang đến những giá trị cốt lõi cho khách hàng là làm đẹp nụ cười – thay đổi cuộc sống."
)

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    evaluation_results_str = llm.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
test_5()