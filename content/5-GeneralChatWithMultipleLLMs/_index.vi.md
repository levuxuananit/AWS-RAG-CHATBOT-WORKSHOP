---
title : "Trò chuyện chung với nhiều LLMS"
date :  "`r Sys.Date()`" 
weight : 5 
chapter : false
pre : " <b> 5. </b> "
---
### Giải pháp RAG với Amazon Bedrock và LangChain
Giải pháp trong hội thảo này được xây dựng bằng cách sử dụng phương pháp **Retrieval-Augmented Generation (RAG)**. RAG là một kiến ​​trúc mô hình tích hợp các khía cạnh của cả kỹ thuật retrieval và generation để nâng cao chất lượng và tính liên quan của văn bản được tạo ra.
Khi bạn nhập một câu hỏi vào hộp văn bản câu hỏi, các bước sau đây sẽ được chạy để cung cấp cho bạn câu trả lời có nguồn gốc từ tài liệu:

- **Truy xuất**: Quá trình này tìm kiếm trong một khối dữ liệu văn bản lớn để tìm thông tin hoặc ngữ cảnh có liên quan. Trong giai đoạn này, Amazon Kendra lấy câu hỏi từ yêu cầu và tìm kiếm các câu trả lời và tài liệu tham khảo có liên quan.

- **Tăng cường**: Sau khi lấy thông tin có liên quan, mô hình sử dụng ngữ cảnh đã lấy được để tăng cường việc tạo văn bản. Điều này có nghĩa là văn bản được tạo ra chịu ảnh hưởng của thông tin đã lấy được, đảm bảo rằng nội dung được tạo ra phù hợp với ngữ cảnh và mang tính thông tin.

- **Generation**: Generation trong RAG đề cập đến khía cạnh sinh sản truyền thống của mô hình, trong đó nó tạo ra văn bản mới dựa trên ngữ cảnh được truy xuất và tăng cường. Văn bản được tạo ra này có thể ở dạng câu trả lời, phản hồi hoặc giải thích.

- **LangChain**: Để điều phối luồng này, chúng tôi sử dụng tác nhân LangChain trong hội thảo này. Các phép trừu tượng linh hoạt và bộ công cụ toàn diện của LangChain giúp các nhà phát triển khai thác khả năng của các mô hình nền tảng (FM).

Trong nhiệm vụ này, bạn sẽ triển khai hàm Lambda RAG (Lấy, Phân tích, Tạo) để cung cấp trải nghiệm chatbot theo ngữ cảnh với các tập dữ liệu của bạn. Các tập dữ liệu mẫu được lưu trữ trong Amazon S3. Để sắp xếp luồng qua các truy vấn của người dùng, bạn sẽ sử dụng LangChain làm công cụ sắp xếp. Mã Lambda được cung cấp sử dụng API LangChain để tóm tắt logic phức tạp cần thiết cho tích hợp này.

### Triển khai Lambda RAG
1. Mở trình soạn thảo VSCode.
2. Từ dự án **bedrock-serverless-workshop**, mở hàm **/lambdas/ragFunctions/ragfunction.py**, sao chép mã bên dưới và cập nhật mã hàm. Hàm này mang logic để hỗ trợ các mô hình Claud3 (Haiku, Sonnet, v.v.) , Mistral và Llama.
```python
import os
import json
import boto3
from langchain_community.retrievers import AmazonKendraRetriever
from langchain_aws import ChatBedrock
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import traceback

kendra = boto3.client('kendra')
chain_type = 'stuff'

KENDRA_INDEX_ID = os.getenv('KENDRA_INDEX_ID')
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]


refine_prompt_template = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "This is the original question: {question}\n"
    "The existing answer: {existing_answer}\n"
    "Now there are some additional texts, (if needed) you can use them to improve your existing answer."
    "\n\n"
    "{context_str}\n"
    "\\nn"
    "Please use the new passage to further improve your answer.\n\n"
    "### Response: "
)

initial_qa_template = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "The following is background knowledge：\n"
    "{context_str}"
    "\n"
    "Please answer this question based on the background knowledge provided above：{question}。\n\n"
    "### Response: "
)

def lambda_handler(event, context):
    print(f"Event is: {event}")
    
    event_body = json.loads(event["body"])
    question = event_body["query"]
    print(f"Query is: {question}")
    
    model_id = event_body["model_id"]
    temperature = event_body["temperature"]
    max_tokens = event_body["max_tokens"]

    response = ''
    status_code = 200

    PROMPT_TEMPLATE = 'prompt-engineering/claude-prompt-template.txt'

    try:
        if model_id == 'mistral.mistral-7b-instruct-v0:2':
            llm = get_mistral_llm(model_id,temperature,max_tokens)
            PROMPT_TEMPLATE = 'prompt-engineering/mistral-prompt-template.txt'
        elif model_id == 'meta.llama3-1-8b-instruct-v1:0':
            llm = get_llama_llm(model_id,temperature,max_tokens)
            PROMPT_TEMPLATE = 'prompt-engineering/llama-prompt-template.txt'
        else:
            llm = get_claude_llm(model_id,temperature,max_tokens)
            PROMPT_TEMPLATE = 'prompt-engineering/claude-prompt-template.txt'
        
        # Read the prompt template from S3 bucket
        s3 = boto3.resource('s3')
        obj = s3.Object(S3_BUCKET_NAME, PROMPT_TEMPLATE) 
        prompt_template = obj.get()['Body'].read().decode('utf-8')
        print(f"prompt template: {prompt_template}")
        
        retriever = AmazonKendraRetriever(kendra_client=kendra,index_id=KENDRA_INDEX_ID)
        
        
        if chain_type == "stuff":
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            chain_type_kwargs = {"prompt": PROMPT}
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs)
            response = qa(question, return_only_outputs=False)
        elif chain_type == "refine":
            refine_prompt = PromptTemplate(
                input_variables=["question", "existing_answer", "context_str"],
                template=refine_prompt_template,
            )
            initial_qa_prompt = PromptTemplate(
                input_variables=["context_str", "question"],
                template=prompt_template,
            )
            chain_type_kwargs = {"question_prompt": initial_qa_prompt, "refine_prompt": refine_prompt}
            qa = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="refine",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs)
            response = qa(question, return_only_outputs=False)
                
        print('Response', response)
        source_documents = response.get('source_documents')
        source_docs = []
        previous_source = None
        previous_score = None
        response_data = []
        
        #if chain_type == "stuff":
        for source_doc in source_documents:
            source = source_doc.metadata['source']
            score = source_doc.metadata["score"]
            if source != previous_source or score != previous_score:
                source_data = {
                    "source": source,
                    "score": score
                }
                response_data.append(source_data)
                previous_source = source
                previous_score = score
        
        response_with_metadata = {
            "answer": response.get('result'),
            "source_documents": response_data
        }

        return {
            'statusCode': status_code,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps(response_with_metadata)
        }
    
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        stack_trace = traceback.format_exc()
        print(f"stack trace: {stack_trace}")
        print(f"error: {str(e)}")
        
        response = str(e)
        return {
            'statusCode': status_code,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({'error': response})
        }
        
def get_claude_llm(model_id, temperature, max_tokens):
    model_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature, 
        "top_k": 50, 
        "top_p": 0.95
    }
    llm = ChatBedrock(model_id=model_id, model_kwargs=model_kwargs) 
    return llm

def get_llama_llm(model_id, temperature, max_tokens):
    model_kwargs = {
        "max_gen_len": max_tokens,
        "temperature": temperature, 
        "top_p": 0.9
    }
    llm = ChatBedrock(model_id=model_id, model_kwargs=model_kwargs) 
    return llm

def get_mistral_llm(model_id, temperature, max_tokens):
    model_kwargs = { 
        "max_tokens": max_tokens,
        "temperature": temperature, 
        "top_k": 50, 
        "top_p": 0.9
    }
    llm = ChatBedrock(model_id=model_id, model_kwargs=model_kwargs) 
    return llm

def get_memory():

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )
    return memory
```
3. Chạy lệnh sau để xây dựng và triển khai với mã lambda mới được cập nhật.
```bash
cd ~/environment/bedrock-serverless-workshop
sam build && sam deploy
```
4. Bạn đã triển khai thành công ứng dụng chatbot không cần máy chủ Amazon Bedrock và cho phép mô hình truy cập vào các Mô hình ngôn ngữ lớn (LLM) cần thiết bằng bảng điều khiển Amazon Bedrock.

### Giới thiệu các mô hình LLM được sử dụn
Trong nhiệm vụ này, mỗi mô hình cung cấp các chức năng và chuyên môn độc đáo. Các mô hình này bao gồm:
- **anthropic.claude-3-haiku - Claude 3 Haiku** là mô hình nhanh nhất, nhỏ gọn nhất của Anthropic cho khả năng phản hồi gần như tức thời. Haiku là lựa chọn tốt nhất để xây dựng trải nghiệm AI liền mạch mô phỏng tương tác của con người.
- **anthropic.claude-3-5-sonnet** là mô hình thông minh và tiên tiến nhất của Anthropic, Claude 3.5 Sonnet, thể hiện khả năng đặc biệt trong nhiều nhiệm vụ và đánh giá khác nhau.
- **anthropic.claude-3-opus - Opus** là một mô hình cực kỳ thông minh với hiệu suất đáng tin cậy trên các tác vụ phức tạp. Nó có thể điều hướng các lời nhắc mở và các tình huống chưa từng thấy với sự trôi chảy đáng kể và khả năng hiểu biết giống như con người. Sử dụng Opus để tự động hóa các tác vụ và đẩy nhanh quá trình nghiên cứu và phát triển trên nhiều trường hợp sử dụng và ngành công nghiệp khác nhau.
- **mistral.mistral-7b-instruct - Mistral** là một mô hình ngôn ngữ lớn hiệu quả cao được tối ưu hóa cho các tác vụ dựa trên ngôn ngữ có khối lượng lớn, độ trễ thấp. Các trường hợp sử dụng phổ biến cho Mistral là tóm tắt văn bản, cấu trúc hóa, trả lời câu hỏi và hoàn thành mã
- **meta.llama3-1-8b-instruct - Llama 3.1 8B** phù hợp nhất với năng lực tính toán và tài nguyên hạn chế. Mô hình này vượt trội trong việc tóm tắt văn bản, phân loại văn bản, phân tích tình cảm và dịch ngôn ngữ đòi hỏi suy luận độ trễ thấp.