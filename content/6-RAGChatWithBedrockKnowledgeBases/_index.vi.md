---
title : "RAG Chat với Bedrock Knowledge Bases"
date :  "`r Sys.Date()`" 
weight : 6 
chapter : false
pre : " <b> 6. </b> "
---
### Amazon Bedrock Knowledge Bases
**Amazon Bedrock Knowledge Bases** là một khả năng được quản lý hoàn toàn cho phép bạn triển khai quy trình làm việc Retrieval Augmented Generation (RAG) bằng cách sử dụng các nguồn dữ liệu độc quyền của tổ chức bạn. Amazon Bedrock Knowledge Bases tự động hóa toàn bộ quy trình làm việc RAG, từ thu thập dữ liệu đến truy xuất và tăng cường nhắc nhở, mà không yêu cầu tích hợp tùy chỉnh hoặc quản lý luồng dữ liệu. Điều này cho phép bạn trang bị cho các mô hình nền tảng (FM) và tác nhân thông tin độc quyền và cập nhật để cung cấp phản hồi có liên quan và chính xác hơn.

Cơ sở tri thức Amazon Bedrock dựa vào hai thành phần chính để cho phép truy xuất hiệu quả thông tin có liên quan: **nhúng** và **kho lưu trữ vector** . Các thành phần này hoạt động cùng nhau để chuyển đổi dữ liệu văn bản thành định dạng có thể tìm kiếm và truy xuất nhanh chóng dựa trên sự tương đồng về mặt ngữ nghĩa.

- **Nhúng** là các biểu diễn số của văn bản nắm bắt ý nghĩa ngữ nghĩa. Trong Amazon Bedrock, các mô hình nhúng như Amazon Titan Embeddings hoặc Cohere Embed chuyển đổi văn bản từ tài liệu của bạn thành các vectơ dày đặc. Quá trình này cho phép so sánh và truy xuất hiệu quả nội dung có ngữ nghĩa tương tự, tạo thành nền tảng cho sự hiểu biết của cơ sở kiến ​​thức về dữ liệu của bạn.

- **Kho dữ liệu vector** là cơ sở dữ liệu chuyên biệt được thiết kế để lập chỉ mục và truy vấn các nhúng vector này một cách hiệu quả. Amazon Bedrock cung cấp các tùy chọn như OpenSearch Serverless do Amazon quản lý hoặc các giải pháp tùy chỉnh như Amazon Aurora PostgreSQL với pgvector. Các kho dữ liệu này cho phép tìm kiếm tương tự nhanh chóng, cho phép hệ thống nhanh chóng xác định và truy xuất thông tin có liên quan nhất khi phản hồi các truy vấn, do đó nâng cao hiệu suất của quy trình làm việc Retrieval Augmented Generation (RAG).

### Chức năng và lợi ích
Chức năng và lợi ích bao gồm:
- **Triển khai RAG liền mạch**: Amazon Bedrock Knowledge Bases tự động hóa toàn bộ quy trình làm việc RAG, từ thu thập dữ liệu đến truy xuất và tăng cường nhanh chóng, mà không yêu cầu tích hợp tùy chỉnh hoặc quản lý luồng dữ liệu. Điều này cho phép bạn trang bị cho các mô hình nền tảng (FM) và tác nhân thông tin độc quyền và cập nhật để cung cấp phản hồi chính xác và phù hợp hơn.

- **Kết nối dữ liệu an toàn**: Dịch vụ tự động lấy tài liệu từ các nguồn dữ liệu bạn chỉ định, bao gồm Amazon S3, Web Crawler, Salesforce và SharePoint. Sau đó, nó xử lý nội dung bằng cách chia thành các khối văn bản, chuyển đổi chúng thành nhúng và lưu trữ chúng trong cơ sở dữ liệu vector.

- **Tùy chọn tùy chỉnh**: Bạn có thể tinh chỉnh cả quy trình truy xuất và thu thập để cải thiện độ chính xác trong các trường hợp sử dụng khác nhau. Có sẵn các tùy chọn phân tích nâng cao để hiểu dữ liệu phi cấu trúc phức tạp và bạn có thể chọn từ nhiều chiến lược phân đoạn khác nhau hoặc thậm chí viết mã phân đoạn tùy chỉnh.

### Knowledge Bases và các thành phần tìm kiếm vector
- **Knowledge Bases**: Kho lưu trữ trung tâm cho thông tin có cấu trúc
- **Amazon S3**: Lưu trữ tài liệu (pdf, csv, txt, v.v.)
- **Amazon OpenSearch** : Cơ sở dữ liệu vector và công cụ tìm kiếm để tìm kiếm sự tương đồng hiệu quả
- **Amazon Bedrock** : Cung cấp quyền truy cập vào các mô hình nhúng, bao gồm Amazon Ttan Embeddings 

### Knowledge Bases và quy trình tìm kiếm vector
- Tài liệu được lưu trữ trong Amazon S3
- Văn bản được trích xuất và xử lý từ các tài liệu
- Mô hình Titan Embeddings của Amazon Bedrock tạo ra các biểu diễn vector của văn bản
- Các vectơ được lưu trữ trong Amazon OpenSearch, hoạt động như một cơ sở dữ liệu vectơ
- Cơ sở kiến ​​thức thu thập và cấu trúc thông tin, kết hợp biểu diễn vectơ
- Các hàm AWS Lambda (Các hàm KB/LLM) tương tác với Cơ sở tri thức và tìm kiếm vectơ
- RAG (Retrieval Augmented Generation) tận dụng tìm kiếm tương tự vectơ để truy xuất thông tin có liên quan
  
### Các tính năng chính của tích hợp Knowledge Bases và tìm kiếm vector
- Khả năng tìm kiếm ngữ nghĩa bằng cách sử dụng biểu diễn vector của văn bản
- Tìm kiếm tương tự hiệu quả thông qua chức năng cơ sở dữ liệu vector của Amazon OpenSearch
- Tích hợp Amazon Titan Embeddings để tạo vector hóa văn bản chất lượng cao
- Cải thiện ngữ cảnh và độ chính xác trong phản hồi của chatbot bằng cách sử dụng truy xuất dựa trên vectơ
- Cải thiện tính liên quan trong việc truy xuất thông tin cho các hoạt động RAG
- Khả năng xử lý và tìm kiếm qua khối lượng lớn dữ liệu văn bản phi cấu trúc
- Sự kết hợp liền mạch giữa tìm kiếm từ khóa truyền thống và tìm kiếm ngữ nghĩa dựa trên vector

### Tóm lại
Thiết lập này cho phép chatbot thực hiện tìm kiếm ngữ nghĩa nâng cao trên kiến ​​thức doanh nghiệp. Bằng cách sử dụng Amazon Titan Embeddings để tạo biểu diễn vectơ của văn bản và lưu trữ chúng trong Amazon OpenSearch dưới dạng cơ sở dữ liệu vectơ, hệ thống có thể tìm thông tin tương tự theo ngữ cảnh ngay cả khi không có từ khóa khớp chính xác. Điều này cải thiện đáng kể khả năng hiểu và phản hồi truy vấn của người dùng bằng thông tin có liên quan từ cơ sở kiến ​​thức của doanh nghiệp của chatbot.