---
title : "Launch VSC on AWS and Set Up the Environment"
date :  "`r Sys.Date()`" 
weight : 2 
chapter : false
pre : " <b> 2. </b> "
---
### Launch a workshop using a CloudFormation template
AWS CloudFormation templates are used to set up resources in the AWS Region you choose. CloudFormation templates create the following AWS resources:
- **VSCode**: VSCode on Amazon EC2 is a cloud-based integrated development environment (IDE) that you can use to write, run, and debug your code using just your browser. It includes a code editor, a debugger, and a terminal. In this workshop, you use the VSCode editor to deploy a backend application, which is built using the **AWS Serverless Application Model (AWS SAM)** and also implements the **AWS Amplify** interface.
- **Amazon S3**: Amazon Simple Storage Service (Amazon S3) is an object storage service that provides industry-leading scalability, data availability, security, and performance. Customers of all sizes and industries can store and protect any amount of data for virtually any use case, such as data lakes, cloud-native applications, and mobile applications. In this workshop, you use S3 buckets to upload documents focused on your use case.
  
1. Download [CloudFormation template](https://static.us-east-1.prod.workshops.aws/public/c5c516a7-10ce-444b-a0c5-1e60794fdb7c/static/template/chatbot-startup-stack.yaml)
2. Save the YAML template file to a folder on your local machine.
3. Navigate to [AWS CloudFormation Console](https://console.aws.amazon.com/cloudformation/home).
4. In the CloudFormation console, select Upload Template File.
5. Select the template you just downloaded, and then select **Next**.
![2.3-CreateStack](/images/2.LaunchVSCAndSetUpTheEnvironment/2.3-CreateStack.png)
6. Name the stack ```chatbot-startup-stack```.
7.Leave the other values ​​as they are and select **Next**.
![2.4_ProvideStackName](/images/2.LaunchVSCAndSetUpTheEnvironment/2.4-ProvideStackName.png)
8. For **Stack Configuration Options**, select the I confirm... option, then select **Next**.
![2.5-TickAllow](/images/2.LaunchVSCAndSetUpTheEnvironment/2.5-TickAllow.png)
9. To deploy the template, select **Submit**
10.  Once the template is deployed, to review the resources you created, navigate to [CloudFormation Resources](https://console.aws.amazon.com/cloudformation/home?/stacks/resources?) , and then select the CloudFormation stack you created.
![2.6-OpenStacks](/images/2.LaunchVSCAndSetUpTheEnvironment/2.6-OpenStacks.png)
{{%notice note%}}
The template deployment takes about 15 minutes to complete the entire AWS resource provisioning process.
{{%/notice%}}

### Launch VSC on AWS
1. Open the [AWS CloudFormation Console](https://console.aws.amazon.com/cloudformation/home).
2. Open the ```chatbot-startup-stack```.
3. Copy the passwords **VSCodeWorkspaceURL** and **VSCodeWorkspacePassword** into a notebook, where:

- **VSCodeWorkspaceURL**: VSCode environment (replacement for AWS Cloud9) hosted on Amazon EC2
- **VSCodeWorkspacePassword**: Password to access the VSCode environment
![2.1-VSCOnAWS](/images/2.LaunchVSCAndSetUpTheEnvironment/2.1-VSCOnAWS.png)
4. Open a new browser window or tab, enter **VSCodeWorkspaceURL** and enter **VSCodeWorkspacePassword** to launch VSCode.
![2.2-LoginCodeServer](/images/2.LaunchVSCAndSetUpTheEnvironment/2.2-LoginCodeServer.png)
5. After successfully launching, your VSCode Editor is ready.

![2.7-VSC](/images/2.LaunchVSCAndSetUpTheEnvironment/2.7-VSC.png)

{{%notice note%}}
The default theme is white, you can optionally change to other color themes. For example: Settings icon -> Theme -> Color theme -> Dark (Visual Studio)
{{%/notice%}}