---
title : "Giới thiệu"
date :  "`r Sys.Date()`" 
weight : 1 
chapter : false
pre : " <b> 1. </b> "
---
### Kiến trúc giải pháp 
![1.AWSServerlessChatbotArchitecture](/images/1.Introduction/1.AWSServerlessChatbotArchitecture.png)

### Thành phần
1. Người dùng: Người dùng cuối của chatbot
2. Web UI: Giao diện người dùng cho tương tác chatbot
3. AWS Amplify: Quản lý giao diện người dùng và xác thực, cụ thể là sử dụng Vue.js
4. Amazon API Gateway: Xử lý các yêu cầu API
5. AWS Lambda (Chức năng KB/LLM): Thực hiện các chức năng không có máy chủ cho Retrieval-Augmented Generation, hoạt động Cơ sở kiến ​​thức và tương tác LLM
6. Amazon Bedrock: Cung cấp quyền truy cập vào các mô hình và dịch vụ AI
7. Mô hình ngôn ngữ lớn (Claude 3, Mistral, Llama, v.v.): Mô hình AI hỗ trợ phản hồi
8. Knowledge Bases: Lưu trữ thông tin có cấu trúc
9. Amazon S3: Lưu trữ đối tượng cho tài liệu và dữ liệu
10. Tài liệu (pdf, csv, txt, v.v.): Nhiều loại tệp khác nhau để thu thập
11. Amazon OpenSearch: Cơ sở dữ liệu vector và công cụ tìm kiếm để tìm kiếm sự tương đồng hiệu quả
12. Amazon Cognito : Xác thực và ủy quyền người dùng

### Quy trình làm việc
1. Người dùng tương tác với Giao diện người dùng web
2. Các yêu cầu được định tuyến qua API Gateway đến các chức năng Lambda
3. Các hàm Lambda sử dụng Bedrock để truy cập LLM, hoạt động RAG và tương tác Knowledge Bases
4. Kiến thức được lấy từ Knowledge Bases, S3, OpenSearch
5. Hệ thống kết hợp nhiều loại tài liệu khác nhau để nâng cao Knowledge Bases
   
### Các tính năng chính
1. Kiến trúc có khả năng mở rộng, không cần máy chủ
2. Tận dụng nhiều mô hình LLM khác nhau bao gồm Claude 3, Mistral và Llama
3. Kết hợp Knowledge Bases tùy chỉnh cùng dữ liệu cá nhân 
4. Xác thực an toàn với Cognito
5. Khả năng tìm kiếm và thu thập tài liệu linh hoạt
6. Front-end được xây dựng bằng các framework hiện đại (Vue.js) sử dụng AWS Amplify

### Tóm lại 
Kiến trúc này cho phép giải pháp chatbot hỗ trợ AI tạo ra có thể mở rộng theo nhu cầu trong khi cung cấp phản hồi thông minh dựa trên cả kiến ​​thức của LLM và dữ liệu cá nhân (thường là dữ liệu mật của doanh nghiệp, cán nhân). Việc sử dụng Vue.js với AWS Amplify đảm bảo giao diện người dùng phản hồi nhanh và hiệu quả.