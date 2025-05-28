---
title : "Triển khai ứng dụng không máy chủ Amazon Bedrock"
date :  "`r Sys.Date()`" 
weight : 3 
chapter : false
pre : " <b> 3. </b> "
---
### Triển khai ứng dụng không máy chủ
Trong nhiệm vụ này, bạn sẽ xây dựng và triển khai cả thành phần backend và frontend của ứng dụng.
- **Backend**: Được triển khai dưới dạng ứng dụng không có máy chủ bằng **AWS SAM**, tạo **Amazon API Gateway**, các hàm **AWS Lambda** cần thiết, nhóm người dùng **Cognito** và lưu trữ thông tin đăng nhập trong **AWS Secrets Manager**.
- **Frontend**: Được xây dựng bằng **Vue.js** và triển khai bằng **AWS Amplify**. Giao diện người dùng sẽ gọi các lệnh gọi API dựa trên REST được lưu trữ bằng **Amazon API Gateway**, **API Gateway** gọi hàm **AWS Lambda**, hàm gọi **API Amazon Bedrock**.
- Kiến trúc không máy chủ này cho phép ứng dụng mở rộng hiệu quả và giảm chi phí vận hành để quản lý cơ sở hạ tầng cơ bản. Quy trình triển khai được tự động hóa thông qua tập lệnh **startup.sh**, xử lý việc cung cấp đầu cuối cho toàn bộ ngăn xếp ứng dụng.

### Cài đặt mã nguồn
1. Từ trình soạn thảo VSCode, hãy chạy lệnh sau để đặt biến môi trường chính được sử dụng trong suốt hội thảo: 
```bash
export CFNStackName=chatbot-startup-stack
export S3BucketName=$(aws cloudformation describe-stacks --stack-name ${CFNStackName} --query "Stacks[0].Outputs[?OutputKey=='S3BucketName'].OutputValue" --output text)
echo "S3 bucket name: $S3BucketName"
```
![3.1-BuildChatBot](/images/3.DeployAmazonBedrockServerlessApplication/3.1-BuildChatBot.png)
- **CFNStackName**: Được tham chiếu sau khi tương tác với các tài nguyên được cung cấp như một phần của triển khai ứng dụng.
- **S3BucketName**: Được sử dụng trong suốt hội thảo để lưu trữ nhiều nguồn dữ liệu và cơ sở kiến ​​thức cần thiết cho ứng dụng. Các dịch vụ phụ trợ sẽ tương tác với nội dung của thùng S3 này để truy xuất và xử lý thông tin cần thiết cho chức năng của ứng dụng. Đảm bảo biến môi trường S3BucketName được thiết lập đúng sẽ cho phép các tác vụ hội thảo truy cập và sử dụng liền mạch dữ liệu cần thiết được lưu trữ trong vị trí trung tâm này.
  
2. Để tải mã nguồn cho hội thảo này, hãy chạy lệnh sau:
```bash
cd ~/environment
git clone https://github.com/levuxuananit/AWS-RAG-CHATBOT.git
```

### Triển khai backend và frontend
1. Để xây dựng và triển khai backend và frontend của ứng dụng không có máy chủ, hãy chạy lệnh sau.
```bash
cd ~/environment/bedrock-serverless-workshop
python3 aws-creds.py
chmod +x startup.sh
./startup.sh
```
2. Tập lệnh này thực hiện các nhiệm vụ sau:
- Nâng cấp hệ điều hành, cài đặt phần mềm jq.
- Xây dựng phần phụ trợ bằng sam build.
- Triển khai phần phụ trợ bằng lệnh sam deploy.
- Cài đặt Amplify và xây dựng giao diện.
- Xuất bản ứng dụng giao diện người dùng bằng Amplify.
- Kịch bản này sẽ mất khoảng 2 đến 5 phút để hoàn thành.

{{%notice note%}}
Trong quá trình amplify add host, bạn sẽ được nhắc lựa chọn hai lần, hãy giữ nguyên lựa chọn mặc định và nhấn **Enter**. Mặc định là tùy chọn đầu tiên **Hosting with Amplify Console** (Hosting được quản lý với tên miền tùy chỉnh, Triển khai liên tục) và tùy chọn thứ 2 là Triển khai thủ công.
{{%/notice%}}

3. Sau khi hoàn tất, bạn sẽ thấy hình ảnh sau.
![3.2-CopyLink](/images/3.DeployAmazonBedrockServerlessApplication/3.2-CopyLink.png)
4. Sao chép và lưu trữ giá trị cho **url amplifyapp**, **user_id** và **password** trong trình soạn thảo văn bản. Bạn sử dụng thông tin này để đăng nhập vào ứng dụng.

### Thiết lập biến môi trường cho lệnh xây dựng SAM và Amplify
1. Để thiết lập các biến môi trường cần thiết cho lệnh xây dựng SAM và Amplify, hãy chạy các lệnh bên dưới:
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