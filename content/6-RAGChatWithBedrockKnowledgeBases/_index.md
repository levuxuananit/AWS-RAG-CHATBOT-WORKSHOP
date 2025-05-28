---
title : "RAG Chat with Bedrock Knowledge Bases"
date : "`r Sys.Date()`"
weight : 6
chapter : false
pre : " <b> 6. </b> "
---
### Amazon Bedrock Knowledge Bases
**Amazon Bedrock Knowledge Bases** is a fully managed capability that enables you to deploy Retrieval Augmented Generation (RAG) workflows using your organization's proprietary data sources. Amazon Bedrock Knowledge Bases automates the entire RAG workflow, from data collection to retrieval and reminder augmentation, without requiring custom integrations or data stream management. This enables you to equip your platform models (FMs) and agents with proprietary and up-to-date information to provide more relevant and accurate responses.

Amazon Bedrock Knowledge Bases rely on two key components to enable efficient retrieval of relevant information: **embeddings** and **vector repositories** . These components work together to transform textual data into a searchable format and retrieve it quickly based on semantic similarity.

- **Embeddings** are digital representations of text that capture semantic meaning. In Amazon Bedrock, embedding models such as Amazon Titan Embeddings or Cohere Embed convert text from your documents into dense vectors. This process enables efficient comparison and retrieval of semantically similar content, forming the foundation for a knowledge base’s understanding of your data.

- **Vector data stores** are specialized databases designed to efficiently index and query these vector embeddings. Amazon Bedrock offers options such as Amazon-managed Serverless OpenSearch or custom solutions such as Amazon Aurora PostgreSQL with pgvector. These data stores enable fast similarity searches, allowing systems to quickly identify and retrieve the most relevant information in response to queries, thereby improving the performance of Retrieval Augmented Generation (RAG) workflows.

### Features and Benefits
Features and Benefits Include:

- **Seamless RAG Deployment**: Amazon Bedrock Knowledge Bases automates the entire RAG workflow, from data collection to retrieval and rapid enhancement, without requiring custom integrations or data stream management. This allows you to equip platform models (FMs) and agents with proprietary and up-to-date information to provide more accurate and relevant responses.

- **Secure Data Connection**: The service automatically retrieves documents from data sources you specify, including Amazon S3, Web Crawler, Salesforce, and SharePoint. It then processes the content by breaking it into text blocks, converting them into embeds, and storing them in a vector database.

- **Customization Options**: You can fine-tune both the retrieval and collection processes to improve accuracy for different use cases. Advanced analytics options are available to understand complex unstructured data, and you can choose from a variety of segmentation strategies or even write custom segmentation code.

### Knowledge Bases and Vector Search Components
- **Knowledge Bases**: Central repository for structured information
- **Amazon S3**: Document storage (pdf, csv, txt, etc.)
- **Amazon OpenSearch**: Vector database and search engine for efficient similarity searching
- **Amazon Bedrock**: Provides access to embedded models, including Amazon Titan Embeddings

### Knowledge Bases and Vector Search Processes
- Documents are stored in Amazon S3
- Text is extracted and processed from documents
- Amazon Bedrock Titan Embeddings models generate vector representations of text
- Vectors are stored in Amazon OpenSearch, which acts as a vector database
- Knowledge Bases collect and structure information, combining vector representations
- AWS Lambda functions (KB/LLM functions) interact with Knowledge Base and Vector Search
- RAG (Retrieval Augmented Generation) leverages vector similarity search to retrieve relevant information

### Key Features of Knowledge Bases and Vector Search Integration
- Semantic search capabilities using vector representations of text
- Efficient similarity search via Amazon OpenSearch vector database functionality
- Amazon Titan Embeddings integration for high-quality text vectorization
- Improved context and accuracy in chatbot responses using vector-based retrieval
- Improved relevance in information retrieval for RAG operations
- Ability to process and search through large volumes of unstructured text data
- Seamless combination of traditional keyword search and vector-based semantic search

### Summary
This setup enables chatbots to perform advanced semantic search on business knowledge. By using Amazon Titan Embeddings to create vector representations of text and storing them in Amazon OpenSearch as a vector database, the system can find contextually similar information even without an exact keyword match. This significantly improves the chatbot's ability to understand and respond to user queries with relevant information from the business's knowledge base.