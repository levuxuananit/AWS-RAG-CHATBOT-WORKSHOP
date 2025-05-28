---
title : "Tạo và kiểm thử kKnowledge Base"
date :  "`r Sys.Date()`" 
weight : 1 
chapter : false
pre : " <b> 6.1 </b> "
---
### Tạo Knowledge Bases
1. Mở [bảng điều khiển Amazon Bedrock](https://console.aws.amazon.com/bedrock/home).
2. Ở góc trên bên trái của bảng điều khiển Amazon Bedrock, hãy chọn biểu tượng **menu**. Trong ngăn điều hướng, hãy chọn Cơ sở **Knowledge Bases**.
3. Chọn **Tạo Knowledge Bases**.
![6.1-CreateKB](/images/6.RAGChatWithBedrockKnowledgeBases/6.1-CreateKB.png)
4. Giữ nguyên tên mặc định.
5. Trong quyền IAM , chọn Sử dụng vai trò hiện có.
6. Từ danh sách thả xuống, chọn vai trò ```xxxx-BedrockExecutionRoleForKBs-xxxx```.
![6.2-KBDetail](/images/6.RAGChatWithBedrockKnowledgeBases/6.2-KBDetail.png)
7. Chọn Amazon S3 làm nguồn dữ liệu.
8. Chọn **Kế tiếp**.
9. Chọn **Duyệt S3**.
![6.3-SelectS3](/images/6.RAGChatWithBedrockKnowledgeBases/6.3-SelectS3.png)
10.  Chọn thư mục ```sample-documents```.
12.  Giữ nguyên cấu hình phân đoạn mặc định.
13.  Chọn **Kế tiếp**.
14.  Đối với mô hình Embeddings , hãy chọn Titan Text Embeddings v2 .
![6.4-SelectModel](/images/6.RAGChatWithBedrockKnowledgeBases/6.4-SelectModel.png)
15.  Đối với cơ sở dữ liệu Vector, chọn Tạo nhanh... .
16.  ChọnKế tiếp.
17.  Chọn **Tạo Knowledge Bases**.
18.  Quá trình này có thể mất vài phút để hoàn tất.

### Đồng bộ dữ liệu với Knowledge Bases
Quá trình này chuyển đổi văn bản từ các nguồn dữ liệu thành nhúng vector bằng các mô hình như Amazon Titan Embeddings, lưu trữ chúng trong cơ sở dữ liệu vector như OpenSearch Serverless. Sau khi thiết lập, Cơ sở tri thức có thể được truy vấn để truy xuất thông tin có liên quan, tăng cường lời nhắc cho các mô hình nền tảng.
1. Trong Nguồn dữ liệu , hãy chọn nguồn kiến ​​thức của bạn ( knowledge-base-quick-... ).
![6.5-Sync](/images/6.RAGChatWithBedrockKnowledgeBases/6.5-Sync.png)
2. Chọn **Đồng bộ**.
3. Quá trình này có thể mất vài phút để hoàn tất.

### Kiểm tra cơ sở dữ liệu vector trong OpenSearch
Quá trình này giúp kiểm tra việc mô hình Amazon Titan Embeddings đã chuyển dữ liệu trong S3 thành cơ sở dữ liệu vector.
1. Mở bảng điều kiển **OpenSearch**, hãy chọn biểu tượng **menu**. Trong ngăn điều hướng, hãy chọn Cơ sở **Collections**.
2. Mở **OpenSearch Dashboards URL**.
![6.6-Collections](/images/6.RAGChatWithBedrockKnowledgeBases/6.6-Collections.png)
3. Chọn biểu tượng **menu**. Trong ngăn điều hướng, hãy chọn Cơ sở **Dev Tools**.
![6.7-DevTools](/images/6.RAGChatWithBedrockKnowledgeBases/6.7-DevTools.png)
4. Chọn biểu tương **run**. 
![6.8-Run](/images/6.RAGChatWithBedrockKnowledgeBases/6.8-Run.png)
5. Kết quả ta nhận được dữ liệu được chuyển đổi sang dạng cơ sở dữ liệu vector.
![6.9-Show](/images/6.RAGChatWithBedrockKnowledgeBases/6.9-Show.png)