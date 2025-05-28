---
title : "Walkthrough Amazon Bedrock Console"
date :  "`r Sys.Date()`" 
weight : 4 
chapter : false
pre : " <b> 4. </b> "
---
### Introducing Amazon Bedrock
Use Amazon Bedrock to build and scale generative AI applications with FM. Amazon Bedrock is a fully managed service that makes FMs from leading AI startups and Amazon available via API. You can choose from a variety of FMs to find the model that best fits your use case. With Amazon Bedrock’s serverless experience, you can get started quickly, customize private FMs with your own data, and quickly integrate and deploy them into your applications using AWS tools without managing any infrastructure.

### Enable model access
1. Open the [Amazon Bedrock console](https://console.aws.amazon.com/bedrock/home).
2. In the upper left corner of the Amazon Bedrock console, select the **menu** icon. In the navigation pane, select **Model access**, then select **Enable specific models**.
![3.3-ModifyModelAccess](/images/3.DeployAmazonBedrockServerlessApplication/3.3-ModifyModelAccess.png)
3. Select the LLMs to use:
- ```Embed English```
- ```Claude 3.5 Sonnet```
- ```Claude 3 Haiku```
- ```Claude 3 Opu```
- ```Llama 3.1 8B Instruct```
- ```Mistral 7B Instruct```
![3.4-SelectFM1](/images/3.DeployAmazonBedrockServerlessApplication/3.4-SelectFM1.png)
![3.5-SelectFM2](/images/3.DeployAmazonBedrockServerlessApplication/3.5-SelectFM2.png)
4. Select **Submit**.