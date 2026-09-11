# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.
### Record hạng 1

 - **class\_id:** 468
- **class\_name:** `cab`
- **rank:** 1
- **score:** 0.510915
- **taxonomy\_name:** `ImageNet-1K`

 ### 1. Record này mô tả toàn ảnh như thế nào?

 Record hạng 1 nói rằng **mô hình phân loại toàn ảnh thành lớp `cab` (taxi)** với score **0.510915**.

 Tuy nhiên, ảnh thực tế là một **cảnh giao thông đô thị đông đúc**, có nhiều xe buýt, ô tô, minibus và các phương tiện khác. Vì vậy, `cab` chỉ là **dự đoán lớp nổi bật nhất của model**, không có nghĩa toàn bộ ảnh chỉ chứa một chiếc taxi.

 ### 2. Ai định nghĩa class list mà checkpoint có thể dự đoán?

 Class list được định nghĩa bởi **taxonomy/dataset mà checkpoint được huấn luyện theo**. Ở đây là **ImageNet-1K**, gồm 1.000 lớp được quy định bởi ImageNet. Checkpoint `yolo11n-cls.pt` chỉ có thể dự đoán trong tập lớp mà nó được huấn luyện.

 ### 3. Vì sao cần giữ cả ID, tên lớp và tên taxonomy?

 Vì ba trường này có vai trò khác nhau:

 - **`class_id = 468`**: định danh lớp một cách ổn định trong taxonomy.
- **`class_name = cab`**: giúp con người đọc và hiểu lớp đó.
- **`taxonomy_name = ImageNet-1K`**: cho biết **ID 468 có ý nghĩa trong hệ thống lớp nào**.

 Điều này đặc biệt quan trọng khi nhiều taxonomy có thể dùng những tên lớp giống nhau nhưng ID hoặc định nghĩa khác nhau. Chỉ lưu `cab` có thể gây **mơ hồ hoặc không tái lập được kết quả**.

 ### 4. Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?

 Guideline cần quy định **đơn vị phân loại và quy tắc chọn nhãn cho toàn ảnh**, chẳng hạn:

 - Phân loại **toàn ảnh** hay từng object?
- Nếu có nhiều loại chủ thể thì chọn **chủ thể chính**, **lớp chiếm ưu thế**, hay cho phép **multi-label**?
- Cách xử lý ảnh có nhiều đối tượng ngang nhau.
- Có được dựa vào background/ngữ cảnh để gán nhãn không?
- Trường hợp không có lớp phù hợp thì dùng nhãn nào.

 Trong ảnh này, đây chính là vấn đề: có rất nhiều phương tiện, nên cần guideline rõ ràng về việc **một nhãn ImageNet-1K đại diện cho toàn cảnh theo tiêu chí nào**.

 ### 5. Vì sao model score không phải ground truth?

 `score = 0.510915` là **độ tin cậy/xác suất do model suy ra cho dự đoán `cab`**, không phải nhãn đúng được con người xác nhận.

 Ground truth phải đến từ **annotation/label chuẩn** của dataset hoặc quy trình đánh giá. Model có thể tự tin dự đoán sai — và trường hợp này score chỉ cho biết model **ưu tiên `cab` hơn các lớp khác**, chứ không chứng minh ảnh thực sự thuộc lớp `cab`.

 **Tóm lại:** record này nên được hiểu là **“YOLO11 classification checkpoint dự đoán toàn ảnh là `cab` với score 0.510915, trong taxonomy ImageNet-1K”**, chứ không phải **“ground truth của ảnh là cab”**.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

Dựa trên ảnh và các prediction đã gán nhãn:

 ### 1\. Diễn giải vị trí box bằng lời

 Ảnh kích thước **640 × 427 px**, tọa độ `bbox_xyxy = [x1, y1, x2, y2]`.

 - **Person (0.913)**: người đứng gần **giữa-phải ảnh**, từ khoảng `(385,69)` đến `(499,349)`, chiếm phần lớn chiều cao.
- **Person (0.611)**: một phần người ở **mép trái**, khoảng `(0,263)`–`(62,311)`, bị cắt bởi biên ảnh.
- **Oven (0.687)**: thiết bị/bếp ở **góc trái**, `(0,188)`–`(196,293)`.
- **Oven (0.634)**: thiết bị ở **phía phải**, `(489,201)`–`(618,345)`.
- **Bowl (0.719)**: tô lớn ở **góc dưới-trái**, `(33,342)`–`(100,385)`.
- **Bowl (0.701)**: tô nhỏ ở **phía trên-trái**, `(155,169)`–`(182,184)`.
- **Bowl (0.500)**: tô ở **khu vực dưới-trái/trung tâm**, `(59,289)`–`(133,329)`.
- **Bowl (0.465)**: một tô nhỏ ở **bên trái, phía trên khu vực bàn**, `(156,114)`–`(174,130)`.
- **Bowl (0.381)**: tô/đồ đựng lớn ở **mép dưới-trái**, `(0,305)`–`(88,371)`.
- **Cup (0.451)**: cốc trên **bàn, phía trái-trung tâm**, `(144,268)`–`(173,302)`.
- **Cup (0.382)**: cốc ngay **bên trái cốc trên**, `(120,272)`–`(142,304)`.

 ### 2\. So sánh số prediction ở hai threshold

 Với threshold hiện tại **0.35**:

 - Person: 2
- Bowl: 5
- Oven: 2
- Cup: 2
- **Tổng: 11 prediction**

 Nếu nâng threshold lên **0.50**:

 - Person: 2
- Bowl: 2 (`0.719`, `0.701`; prediction `0.500` không còn nếu dùng điều kiện `score > 0.50`)
- Oven: 2
- Cup: 0
- **Tổng: 6 prediction**

 Nếu quy ước `score >= 0.50` thì bowl có score 0.499948 vẫn không đạt do thực tế nó thấp hơn 0.5.

 ### 3\. Điều gì thay đổi về độ bao phủ và khối lượng reviewer?

 - **Threshold 0.35 → độ bao phủ cao hơn**: giữ lại 11 box, bao gồm các object nhỏ/yếu như cup và bowl.
- Nhưng **reviewer phải xem nhiều box hơn**, đồng thời phải xử lý nhiều false positive hoặc box chất lượng thấp hơn.
- **Threshold 0.50 → ít box hơn**: giảm từ 11 xuống 6, giúp giảm khoảng **45% khối lượng review**.
- Đổi lại, nguy cơ **bỏ sót object thật** tăng lên. Ví dụ hai cup có score chỉ `0.451` và `0.382` sẽ biến mất hoàn toàn ở threshold 0.5.

 =\> Nếu mục tiêu là **recall/độ bao phủ**, 0.35 hợp lý hơn; nếu ưu tiên **precision và giảm reviewer workload**, threshold cao hơn có lợi.

 ### 4\. Đề xuất một quy tắc “box chặt”

 Có thể dùng guideline:

 > **Box phải bao phủ toàn bộ phần object nhìn thấy, sát biên ngoài của object, không bao gồm background dư thừa; không cố “đoán” phần bị che khuất.**

 Cụ thể:

 - Không để box rộng hơn object chỉ vì background có màu/texture tương tự.
- Với object bị che, box bám theo **phần object quan sát được**.
- Với object bị cắt bởi mép ảnh, box được phép **chạm biên ảnh**, không cần kéo box vào trong để tạo một rectangle “đẹp”.
- Giữ nhất quán giữa các annotator về việc có/không annotate object rất nhỏ hoặc chỉ lộ một phần.

 ### 5\. Object bị che khuất/cắt mép: cần guideline hoặc escalation gì?

 Nên có **occlusion/truncation guideline riêng**, vì đây là nguồn bất đồng lớn:

 - **Bị che khuất nhưng nhận diện rõ** → annotate phần nhìn thấy, đánh dấu `occluded`.
- **Chỉ lộ một phần nhưng vẫn đủ bằng chứng object tồn tại** → annotate và đánh dấu `truncated/partial`.
- **Chỉ còn một mảnh quá nhỏ hoặc không chắc object là gì** → quy định minimum-visible-area hoặc đưa vào **review/escalation**.
- **Cắt mép ảnh** → thống nhất rằng box được phép chạm biên; không coi đây là lỗi box.
- Nếu annotator không thống nhất được object thuộc class nào, nên **escalate thay vì tự suy đoán**.

 Đặc biệt trong ảnh này, **person ở mép trái** là ví dụ điển hình cần guideline truncation: box chạm `x=0`, và reviewer cần biết đây là object hợp lệ bị cắt mép chứ không phải box bị đặt sai.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
- Polygon bổ sung chi tiết gì so với box?
- `instance_id` dùng để làm gì và không phải loại ID nào?
- Đề xuất một quy tắc biên mask:
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`
Đúng, dựa vào **3 kết quả bạn vừa chạy** (classification, detection, segmentation), bảng này có thể điền như sau. Đây là cách viết phù hợp để đưa vào **REPORT.md**:

| Tác vụ                    | Đơn vị/định dạng ground truth                                 | Lỗi hoặc điểm mơ hồ quan sát được                                                                                                                                                                   | Annotator làm gì?                                                                                                        | Reviewer xem gì?                                                                                                       |
| ------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **Phân loại ảnh**         | **1 nhãn cho toàn bộ ảnh** (class/category), ví dụ: `traffic` | Ảnh có nhiều loại phương tiện nên có thể khó chọn **một nhãn đại diện**. Model dự đoán `traffic` với độ tin cậy cao hơn các lớp còn lại.                                                            | Xem toàn bộ ảnh và gán **nhãn đúng duy nhất** theo guideline; nếu ảnh không rõ/không thuộc lớp thì đánh dấu cần xem lại. | Kiểm tra nhãn có đúng với nội dung ảnh và đúng guideline không; chú ý các ảnh có nhiều đối tượng hoặc dễ nhầm lớp.     |
| **Phát hiện vật thể**     | **Bounding box + class + confidence** cho từng vật thể        | Có thể **bỏ sót vật thể nhỏ**, box không bao sát vật thể hoặc nhận nhầm class. Trong ảnh có nhiều người/phương tiện nên có khả năng phát hiện trùng hoặc thiếu.                                     | Vẽ **bounding box** quanh từng vật thể cần gán nhãn và chọn class tương ứng. Đảm bảo mỗi vật thể được đánh dấu đầy đủ.   | Kiểm tra **đủ/thiếu object**, box có bao đúng vật thể không, class có chính xác không và có box trùng nhau không.      |
| **Instance segmentation** | **Mask theo từng instance + class** cho mỗi vật thể           | Mask có thể **không sát biên vật thể**, đặc biệt ở vùng phức tạp hoặc vật thể bị che khuất. Trong ảnh, model tạo nhiều mask cho `bowl`, `person`, `oven`, `dining table`… với confidence khác nhau. | Tạo/chỉnh **mask riêng cho từng instance**, đảm bảo mask nằm đúng trên phần thuộc về vật thể và gán đúng class.          | Kiểm tra **biên mask, instance bị thiếu/thừa, class và mức độ chồng lấn** giữa các mask; yêu cầu sửa lại nếu mask sai. |

### Hiểu đơn giản 3 loại này

Bạn có thể nhớ bằng công thức:

**1. Classification → "Ảnh này là gì?"**

```text
Ảnh → traffic
```

Không quan tâm chính xác từng chiếc xe nằm ở đâu.

---

**2. Detection → "Có những vật gì và chúng nằm ở đâu?"**

```text
Ảnh
 ├── person → [x1,y1,x2,y2]
 ├── oven   → [x1,y1,x2,y2]
 └── bowl   → [x1,y1,x2,y2]
```

→ Dùng **hình chữ nhật (bounding box)**.

---

**3. Instance segmentation → "Chính xác từng pixel của từng vật thể ở đâu?"**

```text
Ảnh
 ├── person → mask người
 ├── bowl   → mask bowl 1
 ├── bowl   → mask bowl 2
 └── oven   → mask oven
```

→ Dùng **mask**, nên chi tiết hơn bounding box.

### Liên hệ với quy trình bạn ghi

```text
Ảnh thô
   ↓
Guideline
   ↓
Ground Truth
   ↓
Huấn luyện model
   ↓
Prediction
   ↓
QC / Rework
```

Ví dụ với **segmentation**:

```text
Ảnh người đang đứng trong bếp
          ↓
Annotator
          ↓
Mask người chính xác
          ↓
Ground Truth
          ↓
Model YOLO11-seg
          ↓
Prediction
          ↓
Reviewer
          ↓
Mask đúng → OK
Mask lệch → Rework
```

**Điểm quan trọng nhất cần nhớ:**
`Ground Truth` là **đáp án chuẩn do con người tạo**, còn `Prediction` là **kết quả model dự đoán**. Confidence như `person 0.90` hay `bowl 0.70` **không có nghĩa là ground truth đúng 90%**, mà là mức độ tin cậy của model đối với dự đoán đó.


## 5. An toàn dữ liệu

-Một quy tắc bảo vệ dữ liệu: Không chia sẻ, sao chép hoặc sử dụng dữ liệu/ảnh ngoài phạm vi được cho phép; chỉ sử dụng dữ liệu phục vụ đúng mục đích của dự án.
-Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Reviewer/Quản lý dự án hoặc người phụ trách dữ liệu (Data Owner).

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
