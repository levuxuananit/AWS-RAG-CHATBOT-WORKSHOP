---
title : "Khởi chạy VSC trên AWS và thiết lập môi trường"
date :  "`r Sys.Date()`" 
weight : 2 
chapter : false
pre : " <b> 2. </b> "
---
### Khởi chạy hội thảo bằng mẫu CloudFormation
Mẫu AWS CloudFormation được sử dụng để thiết lập tài nguyên trong Vùng AWS mà bạn chọn. Mẫu CloudFormation tạo các tài nguyên AWS sau:
- **VSCode**: VSCode trên Amazon EC2 là môi trường phát triển tích hợp (IDE) dựa trên đám mây mà bạn có thể sử dụng để viết, chạy và gỡ lỗi mã của mình chỉ bằng trình duyệt. Nó bao gồm trình soạn thảo mã, trình gỡ lỗi và thiết bị đầu cuối. Trong hội thảo này, bạn sử dụng trình soạn thảo VSCode để triển khai ứng dụng phụ trợ, được xây dựng bằng **AWS Serverless Application Model (AWS SAM)** và cũng triển khai giao diện **AWS Amplify**.
- **Amazon S3**: Amazon Simple Storage Service (Amazon S3) là dịch vụ lưu trữ đối tượng cung cấp khả năng mở rộng, tính khả dụng của dữ liệu, bảo mật và hiệu suất hàng đầu trong ngành. Khách hàng ở mọi quy mô và ngành nghề đều có thể lưu trữ và bảo vệ bất kỳ lượng dữ liệu nào cho hầu như mọi trường hợp sử dụng, chẳng hạn như data lakes, ứng dụng tập trung vào đám mây và ứng dụng di động. Trong hội thảo này, bạn sử dụng thùng S3 để tải lên các tài liệu tập trung vào trường hợp sử dụng của mình.
  
1. Tải xuống [mẫu CloudFormation](https://static.us-east-1.prod.workshops.aws/public/c5c516a7-10ce-444b-a0c5-1e60794fdb7c/static/template/chatbot-startup-stack.yaml)
2. Lưu tệp mẫu YAML vào một thư mục trên máy cục bộ của bạn.
3. Điều hướng đến [AWS CloudFormation Console](https://console.aws.amazon.com/cloudformation/home).
4. Trên bảng điều khiển CloudFormation, chọn Tải lên tệp mẫu.
5. Chọn mẫu mà bạn vừa tải xuống, sau đó chọn **Kế tiếp**.
![2.3-CreateStack](/images/2.LaunchVSCAndSetUpTheEnvironment/2.3-CreateStack.png)
6. Đặt tên cho ngăn xếp ```chatbot-startup-stack```.
7. Giữ nguyên các giá trị khác và chọn **Kế tiếp**.
![2.4_ProvideStackName](/images/2.LaunchVSCAndSetUpTheEnvironment/2.4-ProvideStackName.png)
8. Đối với **Tùy chọn cấu hình ngăn xếp**, hãy chọn Tôi xác nhận... tùy chọn, sau đó chọn **Kế tiếp**.
![2.5-TickAllow](/images/2.LaunchVSCAndSetUpTheEnvironment/2.5-TickAllow.png)
9. Để triển khai mẫu, hãy chọn **Nộp**
10. Sau khi mẫu được triển khai, để xem lại các tài nguyên đã tạo, hãy điều hướng đến [Tài nguyên CloudFormation](https://console.aws.amazon.com/cloudformation/home?/stacks/resources?) , sau đó chọn ngăn xếp CloudFormation mà bạn đã tạo.
![2.6-OpenStacks](/images/2.LaunchVSCAndSetUpTheEnvironment/2.6-OpenStacks.png)
{{%notice note%}}
Việc triển khai mẫu mất khoảng 15 phút để hoàn tất toàn bộ quá trình cung cấp tài nguyên AWS.
{{%/notice%}}

### Khởi chạy VSC trên AWS
1. Mở [AWS CloudFormation Console](https://console.aws.amazon.com/cloudformation/home).
2. Mở ngăn xếp ```chatbot-startup-stack```.
3. Sao chép mật khẩu **VSCodeWorkspaceURL** và **VSCodeWorkspacePassword** vào sổ tay, trong đó:
- **VSCodeWorkspaceURL**: Môi trường VSCode (thay thế cho AWS Cloud9) được lưu trữ trên Amazon EC2
- **VSCodeWorkspacePassword**: Mật khẩu trên truy cập vào môi trường VSCode 
![2.1-VSCOnAWS](/images/2.LaunchVSCAndSetUpTheEnvironment/2.1-VSCOnAWS.png)
4. Mở cửa sổ trình duyệt hoặc tab mới, nhập **VSCodeWorkspaceURL** và nhập **VSCodeWorkspacePassword** để khởi chạy VSCode.
![2.2-LoginCodeServer](/images/2.LaunchVSCAndSetUpTheEnvironment/2.2-LoginCodeServer.png)
5. Sau khi khởi chạy thành công, Trình soạn thảo VSCode của bạn đã sẵn sàng.
![2.7-VSC](/images/2.LaunchVSCAndSetUpTheEnvironment/2.7-VSC.png)

{{%notice note%}}
Chủ đề mặc định là màu trắng, tùy chọn bạn có thể thay đổi sang các chủ đề màu khác. Ví dụ: Biểu tượng cài đặt -> Chủ đề -> Chủ đề màu -> Tối (Visual Studio)
{{%/notice%}}