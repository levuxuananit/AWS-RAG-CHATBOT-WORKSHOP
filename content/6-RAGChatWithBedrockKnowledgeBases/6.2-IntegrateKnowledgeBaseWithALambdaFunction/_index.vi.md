---
title : "Tích hợp Knowledge Base với Lambda Function"
date :  "`r Sys.Date()`" 
weight : 2 
chapter : false
pre : " <b> 6.2 </b> "
---
### Thiết lập AWS Lammbda Function
Trong nhiệm vụ này, bạn sẽ tích hợp Amazon Bedrock Knowledge Base với một hàm AWS Lambda, sử dụng thiết lập API Gateway hiện có. Trọng tâm sẽ là thiết lập một hàm Lambda tương tác với Knowledge Base và kết nối với điểm cuối API được thiết lập trước. Tích hợp này sẽ cho phép UI hiện có giao tiếp với Knowledge Base thông qua kiến ​​trúc không máy chủ, cho phép truy vấn và truy xuất thông tin hiệu quả với RAG. Bằng cách tận dụng API Gateway được cấu hình trước và triển khai hàm Lambda, đồng thời hoàn thiện cơ sở hạ tầng phụ trợ, cho phép người dùng truy cập sức mạnh của Knowledge Base thông qua giao diện web quen thuộc, đồng thời duy trì khả năng mở rộng và tính linh hoạt của giải pháp đám mây gốc.

1. Mở trình soạn thảo VSCode.
2.Từ dự án **bedrock-serverless-workshop** , mở **/lambdas/llmFunctions/kbfunction.py** , sao chép mã bên dưới và cập nhật mã hàm. Hàm này chứa logic để gọi Knowledge Base.

```python
import os
import json
import boto3

import traceback


region = boto3.session.Session().region_name
KB_ID = os.environ["KB_ID"]


def lambda_handler(event, context):
    boto3_version = boto3.__version__
    print(f"Boto3 version: {boto3_version}")
    
    print(f"Event is: {event}")
    event_body = json.loads(event["body"])
    prompt = event_body["query"]
    model_id = event_body["model_id"]
    
    response = ''
    status_code = 200
    
    try:
        model_arn = 'arn:aws:bedrock:'+region+'::foundation-model/'+model_id
        print(f"Model arn: {model_arn}")
        
        response = retrieveAndGenerate(prompt, model_arn)["output"]["text"]
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

def retrieveAndGenerate(prompt, model_arn):
    bedrock_agent_runtime = boto3.client(
            service_name = "bedrock-agent-runtime")
    return bedrock_agent_runtime.retrieve_and_generate(
        input={
            'text': prompt
        },
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': KB_ID,
                'modelArn': model_arn
                }
            }
    )
```
3. Chạy các lệnh sau để lấy Knowledge Base ID và cập nhật biến môi trường của hàm Lambda:
```bash
export KB_ID=$(aws bedrock-agent list-knowledge-bases | jq -r '.knowledgeBaseSummaries[0].knowledgeBaseId')
echo "Knowledge Base ID: $KB_ID"
sed -Ei "s|copy_kb_id|${KB_ID}|g" ./template.yaml
```
{{%notice note%}}
 Lưu ý nếu tài khoản của bạn có nhiều Knowledge Base thì hãy thay đổi **knowledgeBaseSummaries[0]** thành **knowledgeBaseSummaries[i]** có **số thứ tự i** tương ứng.
{{%/notice%}}
4. Mở terminal VSCode, chạy lệnh sau để xây dựng và triển khai với mã lambda mới được cập nhật.
```bash
cd ~/environment/bedrock-serverless-workshop
sam build && sam deploy
```
5. Lambda Function đã được tích hợp thành công với Knowledge Base của bạn. Tích hợp này cho phép Lambda Function tương tác trực tiếp với Knowledge Base, cho phép nó truy xuất và xử lý thông tin từ dữ liệu độc quyền của bạn.