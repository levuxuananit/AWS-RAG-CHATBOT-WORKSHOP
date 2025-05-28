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
2. Từ dự án **bedrock-serverless-workshop**, mở hàm **/lambdas/llmFunctions/llmfunction.py**, sao chép mã bên dưới và cập nhật mã hàm. Hàm này mang logic để hỗ trợ các mô hình Claud3 (Haiku, Sonnet, v.v.) , Mistral và Llama.
```python
import boto3
import json
import traceback


region = boto3.session.Session().region_name

def lambda_handler(event, context):
    boto3_version = boto3.__version__
    print(f"Boto3 version: {boto3_version}")
    
    print(f"Event is: {event}")
    event_body = json.loads(event["body"])
    prompt = event_body["query"]
    temperature = event_body["temperature"]
    max_tokens = event_body["max_tokens"]
    model_id = event_body["model_id"]
    
    response = ''
    status_code = 200
    
    try:
        if model_id == 'mistral.mistral-7b-instruct-v0:2':
            response = invoke_mistral_7b(model_id, prompt, temperature, max_tokens)
        elif model_id == 'meta.llama3-1-8b-instruct-v1:0':
            response = invoke_llama(model_id, prompt, temperature, max_tokens)
        else:
            response = invoke_claude(model_id, prompt, temperature, max_tokens)
            
        return {
            'statusCode': status_code,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({'answer': response})
        }
            
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        stack_trace = traceback.format_exc()
        print(stack_trace)
        return {
            'statusCode': status_code,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({'error': str(e)})
        }


def invoke_claude(model_id, prompt, temperature, max_tokens):
    try:

        instruction = f"Human: {prompt} nAssistant:"
        bedrock_runtime_client = boto3.client(service_name="bedrock-runtime", region_name=region)
        body= {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
        }

        response = bedrock_runtime_client.invoke_model(
            modelId=model_id, body=json.dumps(body)
        )

        response_body = json.loads(response["body"].read())
        outputs = response_body.get("content")
        completions = [output["text"] for output in outputs]
        print(f"completions: {completions[0]}")

        return completions[0]

    except Exception as e:
        raise
        
def invoke_mistral_7b(model_id, prompt, temperature, max_tokens):
    try:
        instruction = f"<s>[INST] {prompt} [/INST]"
        bedrock_runtime_client = boto3.client(service_name="bedrock-runtime", region_name=region)

        body = {
            "prompt": instruction,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = bedrock_runtime_client.invoke_model(
            modelId=model_id, body=json.dumps(body)
        )
        response_body = json.loads(response["body"].read())
        outputs = response_body.get("outputs")
        print(f"response: {outputs}")

        completions = [output["text"] for output in outputs]
        return completions[0]
    except Exception as e:
        raise
        
def invoke_llama(model_id, prompt, temperature, max_tokens):
    print(f"Invoking llam model {model_id}" )
    print(f"max_tokens {max_tokens}" )
    try:
        instruction = f"[INST]You are a very intelligent bot with exceptional critical thinking, help me answering below question.[/INST]"
        total_prompt = f"{instruction}\n{prompt}" 
        
        print(f"Prompt template {total_prompt}" )

        bedrock_runtime_client = boto3.client(service_name="bedrock-runtime", region_name=region)
        
        body = {
            "prompt": total_prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": 0.9
        }

        response = bedrock_runtime_client.invoke_model(
            modelId=model_id, body=json.dumps(body)
        )
        response_body = json.loads(response["body"].read())
        print(f"response: {response_body}")
        return response_body ['generation']
    except Exception as e:
        raise
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

### Kiểm tra kết quả trả lời của Chatbot
1. Mở trang web chatbot. Nếu bạn chưa đăng nhập, hãy sử dụng thông tin đăng nhập mà bạn đã lấy trước đó để đăng nhập. Trang web được hiển thị như sau:
2. Đặt bất kỳ câu hỏi nào bằng cách nhập vào hộp văn bản **Query** và nhấp vào nút **Ask Question**. Bạn có thể thử với một câu hỏi mẫu từ bảng bên phải.
3. Dưới đây một số câu hỏi mẫu và câu trả lời. Bạn có thể thử với các LLM khác và so sánh kết quả.
- Thử nghiệm với mô hình **Claude 3 Haiku**, câu hỏi: ```What is the distance between Earth and Moon?```
![5.1.Test1](/images/5.GeneralChatWithMultipleLLMs/5.1.Test1.png)
- Thử nghiệm với mô hình **Llama 3.1 Instruct 8B**, câu hỏi: ```Tell me a story around a cat and mouse.```
![5.2.Test2](/images/5.GeneralChatWithMultipleLLMs/5.2.Test2.png)