+++
title = "Dọn dẹp tài nguyên"
date = 2021
weight = 7
chapter = false
pre = "<b> 7. </b>"
+++
### Xóa ứng dụng AWS SAM và khởi động ngăn xếp CloudFormation
1. Điều hướng đến VSC trên AWS, sau đó chạy các lệnh sau:
2. Để xóa ngăn xếp SAM, hãy chạy lệnh sau.
```bash
cd ~/environment/bedrock-serverless-workshop
sam delete
```
![7.1CleanUp1](/images/7.CleanUp/7.1CleanUp1.png)
1. Để xóa ngăn xếp khởi động, hãy chạy lệnh sau.
```bash
aws cloudformation delete-stack --stack-name $CFNStackName
```