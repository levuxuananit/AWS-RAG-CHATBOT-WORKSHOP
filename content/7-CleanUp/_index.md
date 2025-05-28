+++
title = "Clean up resources"
date = 2022
weight = 7
chapter = false
pre = "<b> 7. </b>"
+++
### Delete the AWS SAM application and start the CloudFormation stack
1. Navigate to VSC on AWS, then run the following commands:
2. To delete the SAM stack, run the following command.
```bash
cd ~/environment/bedrock-serverless-workshop
sam delete
```
![7.1CleanUp1](/images/7.CleanUp/7.1CleanUp1.png)
3. To delete the startup stack, run the following command.
```bash
aws cloudformation delete-stack --stack-name $CFNStackName
```