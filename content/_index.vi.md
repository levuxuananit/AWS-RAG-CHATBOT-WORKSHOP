---
title : "AWS RAG CHATBOT WORKSHOP"
date :  "`r Sys.Date()`" 
weight : 1 
chapter : false
---
# Chatbot PDF được cá nhân hóa với Amazon Bedrock 
Chào mừng đến với hội thảo Chatbot PDF được cá nhân hóa với Amazon Bedrock! Trong buổi thực hành này, bạn sẽ xây dựng một chatbot thông minh, hỗ trợ AI có thể hiểu và trả lời các câu hỏi của người dùng dựa trên các tài liệu mà họ tự tải lên, chẳng hạn như PDF, CSV và tệp văn bản.

Giải pháp tận dụng kiến ​​trúc không cần máy chủ, có thể mở rộng được hỗ trợ bởi Amazon Bedrock, tích hợp các mô hình ngôn ngữ lớn hàng đầu, chẳng hạn như Claude 3, Mistral và Llama, với các kỹ thuật Tăng cường truy xuất (RAG). Bạn sẽ sử dụng AWS Lambda để chạy các chức năng cốt lõi để phân tích cú pháp tài liệu, truy xuất kiến ​​thức và phản hồi chatbot. Kiến thức được lưu trữ và truy vấn an toàn từ Amazon S3, OpenSearch và Cơ sở kiến ​​thức tùy chỉnh, nâng cao khả năng cung cấp các câu trả lời có liên quan và nhận biết tài liệu của chatbot.

Giao diện người dùng được xây dựng bằng Vue.js và AWS Amplify, cung cấp giao diện hiện đại, phản hồi nhanh, trong khi Amazon Cognito đảm bảo xác thực an toàn và quyền truy cập được cá nhân hóa. Người dùng tương tác thông qua Web UI, với tất cả quá trình xử lý phụ trợ được xử lý liền mạch thông qua Amazon API Gateway và Lambda.

Đến cuối hội thảo này, bạn sẽ có một chatbot đầy đủ chức năng kết hợp trí thông minh chung của LLM với dữ liệu riêng, dành riêng cho miền của bạn—hoàn hảo cho các trường hợp sử dụng doanh nghiệp, giáo dục hoặc nội bộ.

Hãy bắt đầu thôi!