---
title : "Hướng dẫn bảng điều kiển Amazon Bedrock"
date :  "`r Sys.Date()`" 
weight : 4 
chapter : false
pre : " <b> 4. </b> "
---
### Giới thiệu Amazon Bedrock
Sử dụng Amazon Bedrock để xây dựng và mở rộng các ứng dụng AI tạo ra với FM. Amazon Bedrock là một dịch vụ được quản lý hoàn toàn giúp FM, từ các công ty khởi nghiệp AI hàng đầu và Amazon, có sẵn thông qua API. Bạn có thể chọn từ nhiều FM khác nhau để tìm mô hình phù hợp nhất với trường hợp sử dụng của mình. Với trải nghiệm không cần máy chủ của Amazon Bedrock, bạn có thể bắt đầu nhanh chóng, tùy chỉnh FM riêng tư bằng dữ liệu của riêng bạn và nhanh chóng tích hợp và triển khai chúng vào các ứng dụng của mình bằng cách sử dụng các công cụ AWS mà không cần phải quản lý bất kỳ cơ sở hạ tầng nào.

### Kích hoạt quyền truy cập mô hình
1. Mở [bảng điều khiển Amazon Bedrock](https://console.aws.amazon.com/bedrock/home).
2. Ở góc trên bên trái của bảng điều khiển Amazon Bedrock, hãy chọn biểu tượng **menu**. Trong ngăn điều hướng, hãy chọn **Model access**, rồi chọn **Enable specific models**.
![3.3-ModifyModelAccess](/images/3.DeployAmazonBedrockServerlessApplication/3.3-ModifyModelAccess.png)
3. Chọn các LLMs sử dụng:
- ```Embed English```
- ```Claude 3.5 Sonnet```
- ```Claude 3 Haiku```
- ```Claude 3 Opu```
- ```Llama 3.1 8B Instruct```
- ```Mistral 7B Instruct```
![3.4-SelectFM1](/images/3.DeployAmazonBedrockServerlessApplication/3.4-SelectFM1.png)
![3.5-SelectFM2](/images/3.DeployAmazonBedrockServerlessApplication/3.5-SelectFM2.png)
4. Chọn **Nộp**.