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
2. From **bedrock-serverless-workshop** project, open **/lambdas/ragFunctions/ragfunction.py** function, copy the code below and update the function code. This function carries the logic to support Claud3 models (Haiku, Sonnet, etc.), Mistral and Llama.
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