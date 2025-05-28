---
title : "Integrate Knowledge Base with a Lambda Function"
date : "`r Sys.Date()`"
weight : 2
chapter : false
pre : " <b> 6.2 </b> "
---
### Setting Up an AWS Lambda Function
In this task, you will integrate Amazon Bedrock Knowledge Base with an AWS Lambda function, using an existing API Gateway setup. The focus will be on setting up a Lambda function that interacts with the Knowledge Base and connects to a pre-configured API endpoint. This integration will allow the existing UI to communicate with the Knowledge Base via a serverless architecture, allowing for efficient querying and retrieval of information with RAG. By leveraging a pre-configured API Gateway and deploying a Lambda function, and completing the backend infrastructure, users will be able to access the power of the Knowledge Base through a familiar web interface, while maintaining the scalability and flexibility of a cloud-native solution.

1. Open VSCode editor.
2.From **bedrock-serverless-workshop** project, open **/lambdas/llmFunctions/kbfunction.py** , copy the code below and update the function code. This function contains the logic to call the Knowledge Base.

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
3. Run the following commands to get the Knowledge Base ID and update the Lambda function's environment variable:
```bash
export KB_ID=$(aws bedrock-agent list-knowledge-bases | jq -r '.knowledgeBaseSummaries[0].knowledgeBaseId')
echo "Knowledge Base ID: $KB_ID"
sed -Ei "s|copy_kb_id|${KB_ID}|g" ./template.yaml
```
{{%notice note%}}
Note that if your account has multiple Knowledge Bases, change **knowledgeBaseSummaries[0]** to **knowledgeBaseSummaries[i]** with the corresponding **order number i**.

{{%/notice%}}
4. Open the VSCode terminal, run the following command to build and deploy with the newly updated lambda code.
```bash
cd ~/environment/bedrock-serverless-workshop
sam build && sam deploy
```
5. The Lambda Function has been successfully integrated with your Knowledge Base. This integration allows the Lambda Function to interact directly with the Knowledge Base, allowing it to retrieve and process information from your proprietary data.

### Check RAG Answer Results with Knowledge Bases
1. Open the chatbot website. If you are not logged in, use the credentials you got earlier to log in. Select **RAG with Knowledge Bases** mode
2. Ask any question by typing in the Query text box and click the Ask Question button. You can try with a sample question from the table on the right.
3. Below are some sample questions and answers. You can try with other LLMs and compare the results.
- Test with Claude 3 Haiku model, question: ```What was the total number of shares of common stock outstanding as of January 22, 2020?```
![6.10-test1kb](/images/6.RAGChatWithBedrockKnowledgeBases/6.10-test1kb.png)
- Test with Claude 3 Haiku model, question: ```What percentage of Amazon’s annual revenue was recognized in Q4 of 2019?```
![6.12-test2kb](/images/6.RAGChatWithBedrockKnowledgeBases/6.12-test2kb.png)
- Test with Claude 3 Haiku model, question: ```How many shareholders of record did Amazon have as of January 22, 2020?```
![6.14-test3kb](/images/6.RAGChatWithBedrockKnowledgeBases/6.14-test3kb.png)
4. The result shows that the Chatbot has answered correctly the results retrieved from the Knowledge Bases (private data)

### Test the General chat answer result with LLM
Reuse the examples in the **Test RAG answer result with Knowledge Bases** section to test the General chat answer result with LLM
1. Open the chatbot website. If you are not logged in, use the login information you got earlier to log in. Select the **General chat with LLM** mode
2. Use the questions from the previous section to enter in the Query text box and click the Ask Question button. You can try with a sample question from the right panel.
3. Below are some sample questions and answers. You can try with other LLMs and compare the results.
- Test with Claude 3 Haiku model, question: ```What was the total number of shares of common stock outstanding as of January 22, 2020?```
![6.11-test1llm](/images/6.RAGChatWithBedrockKnowledgeBases/6.11-test1llm.png)
- Test with Claude 3 Haiku model, question: ```What percentage of Amazon’s annual revenue was recognized in Q4 of 2019?```
![6.13-test2llm](/images/6.RAGChatWithBedrockKnowledgeBases/6.13-test2llm.png)
- Test with Claude 3 Haiku model, question: ```How many shareholders of record did Amazon have as of January 22, 2020?```
![6.15-test3llm](/images/6.RAGChatWithBedrockKnowledgeBases/6.15-test3llm.png)
4. The results show that the Chatbot does not have the information to answer internal, private questions like this.