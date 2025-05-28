---
title : "AWS RAG CHATBOT WORKSHOP"
date :  "`r Sys.Date()`" 
weight : 1 
chapter : false
---
# Personalized PDF Chatbots with Amazon Bedrock

Welcome to the Personalized PDF Chatbots with Amazon Bedrock workshop! In this hands-on session, you will build an intelligent, AI-powered chatbot that can understand and answer user questions based on documents they upload themselves, such as PDFs, CSVs, and text files.

The solution leverages a scalable, serverless architecture powered by Amazon Bedrock, integrating leading large language models, such as Claude 3, Mistral, and Llama, with Retrieval Augmentation (RAG) techniques. You will use AWS Lambda to run core functions for document parsing, knowledge retrieval, and chatbot responses. Knowledge is securely stored and queried from Amazon S3, OpenSearch, and a custom Knowledge Base, enhancing the chatbot's ability to provide relevant and document-aware answers.

The UI is built using Vue.js and AWS Amplify, providing a modern, responsive interface, while Amazon Cognito ensures secure authentication and personalized access. Users interact through the Web UI, with all backend processing handled seamlessly through Amazon API Gateway and Lambda.

By the end of this workshop, you will have a fully functional chatbot that combines LLM’s general intelligence with your own private, domain-specific data—perfect for enterprise, educational, or internal use cases.

Let’s get started!