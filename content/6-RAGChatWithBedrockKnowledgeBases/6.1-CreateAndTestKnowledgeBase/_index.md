---
title : "Create and Test Knowledge Base"
date : "`r Sys.Date()`"
weight : 1
chapter : false
pre : " <b> 6.1 </b> "
---
### Creating Knowledge Bases
1. Open the [Amazon Bedrock console](https://console.aws.amazon.com/bedrock/home).
2. In the upper-left corner of the Amazon Bedrock console, select the **menu** icon. In the navigation pane, select **Knowledge Bases**.
3. Select **Create Knowledge Bases**.
![6.1-CreateKB](/images/6.RAGChatWithBedrockKnowledgeBases/6.1-CreateKB.png)
4. Leave the default name.
5. Under IAM permissions, select Use an existing role.
6. From the drop-down list, select the role ```xxxx-BedrockExecutionRoleForKBs-xxxx```.

![6.2-KBDetail](/images/6.RAGChatWithBedrockKnowledgeBases/6.2-KBDetail.png)
7. Select Amazon S3 as the data source.
8. Select **Next**.
9. Select **Browse S3**.
![6.3-SelectS3](/images/6.RAGChatWithBedrockKnowledgeBases/6.3-SelectS3.png)
10. Select the ```sample-documents``` folder.
12. Leave the default sharding configuration.
13. Select **Next**.
14. For the Embeddings model, select Titan Text Embeddings v2 .
![6.4-SelectModel](/images/6.RAGChatWithBedrockKnowledgeBases/6.4-SelectModel.png)
15. For Vector databases, select Quick Create... .
16. SelectNext.
17. Select **Create Knowledge Bases**.
18. This process may take a few minutes to complete.

### Sync Data to Knowledge Bases
This process converts text from data sources into vector embeddings using models such as Amazon Titan Embeddings, storing them in a vector database such as OpenSearch Serverless. Once set up, the Knowledge Base can be queried to retrieve relevant information, enhancing prompts for the underlying models.
1. Under Data Sources, select your knowledge source ( knowledge-base-quick-... ).
![6.5-Sync](/images/6.RAGChatWithBedrockKnowledgeBases/6.5-Sync.png)
2. Select **Sync**.
3. This process may take a few minutes to complete.

### Test the vector database in OpenSearch
This process helps test that the Amazon Titan Embeddings model has converted the data in S3 into a vector database.
1. Open the **OpenSearch** dashboard, select the **menu** icon. In the navigation pane, select **Collections** Base.
2. Open the **OpenSearch Dashboards URL**.
![6.6-Collections](/images/6.RAGChatWithBedrockKnowledgeBases/6.6-Collections.png)
3. Select the **menu** icon. In the navigation pane, select **Dev Tools** Base.
![6.7-DevTools](/images/6.RAGChatWithBedrockKnowledgeBases/6.7-DevTools.png)
4. Select the **run** icon.
![6.8-Run](/images/6.RAGChatWithBedrockKnowledgeBases/6.8-Run.png)
5. The result is the data converted to a vector database.
![6.9-Show](/images/6.RAGChatWithBedrockKnowledgeBases/6.9-Show.png)