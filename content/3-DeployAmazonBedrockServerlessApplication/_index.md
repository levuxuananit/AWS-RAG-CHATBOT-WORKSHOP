---
title : "Deploy Amazon Bedrock Serverless Application"
date :  "`r Sys.Date()`" 
weight : 3 
chapter : false
pre : " <b> 3. </b> "
---
### Deploying a Serverless Application
In this task, you will build and deploy both the backend and frontend components of the application
- **Backend**: Deployed as a serverless application using **AWS SAM**, creating **Amazon API Gateway**, the required **AWS Lambda** functions, **Cognito** user pool, and storing credentials in **AWS Secrets Manager**.
- **Frontend**: Built using **Vue.js** and deployed using **AWS Amplify**. The frontend will invoke REST-based API calls hosted using **Amazon API Gateway**, **API Gateway** calls **AWS Lambda** functions, and **Amazon Bedrock API** calls.
- This serverless architecture allows the application to scale efficiently and reduces the operational costs of managing the underlying infrastructure. The deployment process is automated through the **startup.sh** script, which handles end-to-end provisioning for the entire application stack.

### Source Code Settings
1. From the VSCode editor, run the following command to set the primary environment variable used throughout the workshop:
```bash
export CFNStackName=chatbot-startup-stack
export S3BucketName=$(aws cloudformation describe-stacks --stack-name ${CFNStackName} --query "Stacks[0].Outputs[?OutputKey=='S3BucketName'].OutputValue" --output text)
echo "S3 bucket name: $S3BucketName"
```
![3.1-BuildChatBot](/images/3.DeployAmazonBedrockServerlessApplication/3.1-BuildChatBot.png)
- **CFNStackName**: Referenced after interacting with resources provided as part of the application deployment.
- **S3BucketName**: Used throughout the workshop to store various data sources and knowledge bases required by the application. Backend services will interact with the contents of this S3 bucket to retrieve and process information required for the application to function. Ensuring that the S3BucketName environment variable is set correctly will allow workshop tasks to seamlessly access and use the required data stored in this central location.

2. To download the source code for this workshop, run the following command:
```bash
cd ~/environment
git clone https://github.com/levuxuananit/AWS-RAG-CHATBOT.git
```

### Deploying the backend and frontend
1. To build and deploy the backend and frontend of the serverless application, run the following command.
```bash
cd ~/environment/bedrock-serverless-workshop
python3 aws-creds.py
chmod +x startup.sh
./startup.sh
```
2. This script performs the following tasks:
- Upgrade the operating system, install jq software.
- Build the backend using sam build.
- Deploy the backend using sam deploy.
- Install Amplify and build the frontend.
- Publish the frontend application using Amplify.
- This script will take about 2 to 5 minutes to complete.

{{%notice note%}}
During the amplify add host process, you will be prompted to select twice, keep the default selection and press **Enter**. The default is the first option **Hosting with Amplify Console** (Managed Hosting with Custom Domain, Continuous Deployment) and the second option is Manual Deployment.
{{%/notice%}}

3. Once completed, you will see the following image.
![3.2-CopyLink](/images/3.DeployAmazonBedrockServerlessApplication/3.2-CopyLink.png)
4. Copy and save the values ​​for **amplifyapp url**, **user_id** and **password** in a text editor. You will use this information to log in to the application.

### Set environment variables for SAM and Amplify build commands
1. To set the required environment variables for SAM and Amplify build commands, run the commands below:
```bash
export SAMStackName="sam-$CFNStackName"
export AWS_REGION=$(aws configure get region)
export BedrockApiUrl=$(aws cloudformation describe-stacks --stack-name ${SAMStackName} --query "Stacks[0].Outputs[?OutputKey=='BedrockApiUrl'].OutputValue" --output text)
export SecretName=$(aws cloudformation describe-stacks --stack-name ${SAMStackName} --query "Stacks[0].Outputs[?OutputKey=='SecretsName'].OutputValue" --output text)
export BedrockApiUrl=$(aws cloudformation describe-stacks --stack-name ${SAMStackName} --query "Stacks[0].Outputs[?OutputKey=='BedrockApiUrl'].OutputValue" --output text)
export UserPoolId=$(aws cloudformation describe-stacks --stack-name ${SAMStackName} --query "Stacks[0].Outputs[?OutputKey=='CognitoUserPool'].OutputValue" --output text)
export UserPoolClientId=$(aws cloudformation describe-stacks --stack-name ${SAMStackName} --query "Stacks[0].Outputs[?OutputKey=='CongnitoUserPoolClientID'].OutputValue" --output text)
export SecretName=$(aws cloudformation describe-stacks --stack-name ${SAMStackName} --query "Stacks[0].Outputs[?OutputKey=='SecretsName'].OutputValue" --output text)
export PATH=~/.npm-global/bin:$PATH
```