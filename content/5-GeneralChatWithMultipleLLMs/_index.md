---
title : "General Chat with Multiple LLMs"
date :  "`r Sys.Date()`" 
weight : 5 
chapter : false
pre : " <b> 5. </b> "
---
### RAG Solution with Amazon Bedrock and LangChain
The solution in this workshop is built using the **Retrieval-Augmented Generation (RAG)** approach. RAG is a model architecture that integrates aspects of both retrieval and generation techniques to enhance the quality and relevance of the generated text.

When you enter a question in the question text box, the following steps are run to provide you with a document-derived answer:

- **Retrieval**: This process searches a large corpus of text data for relevant information or context. During this phase, Amazon Kendra retrieves the question from the request and searches for relevant answers and references.

- **Enhanced**: After retrieving relevant information, the model uses the retrieved context to enhance the text generation. This means that the generated text is influenced by the retrieved information, ensuring that the generated content is contextual and informative.

- **Generation**: Generation in RAG refers to the traditional generative aspect of the model, where it generates new text based on the retrieved and augmented context. This generated text can be in the form of answers, responses, or explanations.

- **LangChain**: To orchestrate this flow, we use the LangChain agent in this workshop. LangChain's flexible abstractions and comprehensive toolkit enable developers to tap into the capabilities of platform models (FM).

In this assignment, you will implement a Lambda RAG (Get, Analyze, Generate) function to deliver a contextual chatbot experience with your datasets. The sample datasets are stored in Amazon S3. To stream through user queries, you will use LangChain as the sorter. The provided Lambda code uses the LangChain API to abstract away the complex logic required for this integration.

### Deploying Lambda RAG
1. Open VSCode editor.
2. From **bedrock-serverless-workshop** project, open **/lambdas/llmFunctions/llmfunction.py** function, copy the code below and update the function code. This function carries the logic to support Claud3 models (Haiku, Sonnet, etc.), Mistral and Llama.
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
3. Run the following command to build and deploy with the updated lambda code.
```bash
cd ~/environment/bedrock-serverless-workshop
sam build && sam deploy
```
4. You have successfully deployed your Amazon Bedrock serverless chatbot application and enabled the model to access the required Large Language Models (LLMs) using the Amazon Bedrock console.

### Introducing the LLM models used
In this task, each model provides unique capabilities and expertise. These models include:
- **anthropic.claude-3-haiku - Claude 3 Haiku** is Anthropic's fastest, most compact model for near-instantaneous response. Haiku is the best choice for building seamless AI experiences that simulate human interactions.
- **anthropic.claude-3-5-sonnet** is Anthropic's most advanced and intelligent model, Claude 3.5 Sonnet, which demonstrates exceptional performance across a wide range of tasks and assessments.
- **anthropic.claude-3-opus - Opus** is an extremely intelligent model with reliable performance on complex tasks. It can navigate open-ended prompts and unseen scenarios with remarkable fluency and human-like understanding. Use Opus to automate tasks and accelerate research and development across a wide range of use cases and industries.
- **mistral.mistral-7b-instruct - Mistral** is a high-performance large language model optimized for high-volume, low-latency language-based tasks. Common use cases for Mistral are text summarization, structuring, question answering, and code completion
- **meta.llama3-1-8b-instruct - Llama 3.1 8B** is best suited for limited computing power and resources. The model excels at text summarization, text classification, sentiment analysis, and language translation that require low-latency inference.

### Check the Chatbot's response
1. Open the chatbot website. If you are not logged in, use the credentials you got earlier to log in. The website is displayed as follows:
2. Ask any question by typing in the **Query** text box and click the **Ask Question** button. You can try with a sample question from the table on the right.
3. Below are some sample questions and answers. You can try with other LLMs and compare the results.
- Test with **Claude 3 Haiku** model, question: ```What is the distance between Earth and Moon?```
![5.1.Test1](/images/5.GeneralChatWithMultipleLLMs/5.1.Test1.png)
- Test with **Llama 3.1 Instruct 8B** model, question: ```Tell me a story around a cat and mouse.```
![5.2.Test2](/images/5.GeneralChatWithMultipleLLMs/5.2.Test2.png)