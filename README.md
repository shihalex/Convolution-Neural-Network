# 集成卷積神經網路 | Convolutional Neural Networks

> **程式碼公開說明：** 本 Repository **僅包含 `README.md` 說明文件**，不公開課程 Jupyter Notebook、作業解答、模型程式碼與相關課程素材。為遵守 **Coursera Honor Code（榮譽準則）**及課程學術誠信要求，以下僅以文字整理學習主題、實作方法與既有實驗結果，供課程學習歷程與技術成果展示使用。

本儲存庫整理國立成功大學「**集成卷積神經網路（Convolutional Neural Networks）**」課程的 **8 個實作主題**，內容對應 DeepLearning.AI 在 Coursera 開設的 [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks) 課程練習。主題涵蓋 NumPy 卷積與池化運算、CNN 影像分類、ResNet、MobileNetV2 遷移學習、YOLO 物件偵測、U-Net 語意分割、FaceNet 人臉辨識及神經風格轉換。

本專案定位為**課程學習與模型實作紀錄**。各節所列結果取自課程作業 Notebook 當時保存的執行輸出，**並非重新執行後的獨立驗證結果**；由於不公開程式碼及課程資料，讀者無法僅憑此 Repository 重現實驗。

## 專案資訊

| 項目 | 內容 |
| --- | --- |
| 課程名稱 | 集成卷積神經網路（Convolutional Neural Networks） |
| 開課單位 | 國立成功大學（NCKU）資訊工程學系 |
| 開課學期 | 2025 Fall（114 學年度第 1 學期） |
| 專案性質 | Coursera 課程作業與深度學習實作成果說明 |
| 公開內容 | 僅 `README.md`；不公開 Notebook、作業程式碼及課程提供的資料／模型檔案 |
| 課程大綱 | [NCKU 課程大綱](https://class-qry.acad.ncku.edu.tw/syllabus/online_display.php?syear=0114&sem=1&co_no=F724200&class_code=) |
| 課程網址 | [Coursera — Convolutional Neural Networks（DeepLearning.AI）](https://www.coursera.org/learn/convolutional-neural-networks) |
| GitHub Repository | [shihalex/Convolution-Neural-Network](https://github.com/shihalex/Convolution-Neural-Network) |

## 目錄

- [專案資訊](#專案資訊)
- [各項作業說明](#各項作業說明)
- [課程來源與參考資料](#課程來源與參考資料)

## 各項作業說明

### 1. Convolutional Model — Step by Step

**對應作業：** `Convolution_model_Step_by_Step_v1.ipynb`（僅列出課程作業名稱，不提供檔案）

以 **NumPy 從基礎運算實作 CNN 的前向與反向傳播**，理解卷積與池化的數學運算、維度變化及梯度傳遞，而不是直接以 Keras 的現成卷積層完成作業。

- **卷積前向傳播：** `zero_pad`、`conv_single_step`、`conv_forward`，實作補零、局部卷積核加權計算，並考慮濾波器尺寸、Padding 與 Stride 對輸出張量的影響。
- **池化前向傳播：** `pool_forward`，分別實作 Max Pooling 與 Average Pooling，觀察下採樣如何縮減特徵圖尺寸。
- **卷積反向傳播：** `conv_backward`，練習計算對輸入特徵、卷積權重與偏置的梯度。
- **池化反向傳播：** `create_mask_from_window`、`distribute_value`、`pool_backward`，分別處理最大值位置的梯度回傳與平均分配。

**實作結果與觀察：** Notebook 保存的卷積單步計算、卷積前向傳播及池化前向傳播測試均顯示通過；例如指定測試案例中的單步卷積輸出為 **−6.9991**，卷積前向傳播輸出的平均值約為 **0.5511**。這些數值屬於固定測試輸入下的函式檢查，**不是模型分類準確率**。本作業重點在建立對 CNN 底層計算流程及反向傳播原理的理解。

### 2. Convolutional Model — Application

**對應作業：** `Convolution_model_Application.ipynb`（不公開）

利用 **TensorFlow／Keras** 將 CNN 應用於實際影像分類，比較 Sequential API 與 Functional API 的建模方式，並練習影像資料正規化、訓練及模型評估。

| 任務 | 資料集與模型設計 | 已儲存實驗結果 |
| --- | --- | --- |
| **Happy House：快樂表情二元分類** | 600 張訓練影像、150 張測試影像；輸入為 **64 × 64 × 3**。使用 Sequential API 建立 `ZeroPadding2D → Conv2D → BatchNormalization → ReLU → MaxPooling2D → Flatten → Dense(Sigmoid)`。 | 訓練 **10 Epochs**；測試 **Loss 0.6781、Accuracy 80.00%**。 |
| **SIGNS：手勢六類分類** | 使用手勢影像資料集，以 Functional API 建立兩組 `Conv2D → ReLU → MaxPooling2D`，再接 Flatten 與六類 Softmax 輸出。 | 訓練 **100 Epochs**；最後一輪記錄 `val_loss` 約 **0.4884**、`val_accuracy` **83.33%**。 |

**結果解讀：** Happy House 展示了基本 CNN 在二元表情分類上的應用；SIGNS 則練習透過多層卷積提取特徵並完成多類別分類。須注意 SIGNS 作業將原本的 `test_dataset` 傳入訓練流程的 `validation_data`，因此 **83.33% 是該資料作為驗證集時記錄的數值**，不宜當作完全獨立的最終測試成績。

### 3. Residual Networks

**對應作業：** `Residual_Networks.ipynb`（不公開）

探討深層 CNN 的最佳化困難，使用 **Shortcut／Skip Connection** 建立殘差網路（Residual Network），學習透過殘差映射改善深層模型的梯度傳遞。

- 實作 **Identity Block** 與 **Convolutional Block**：前者在輸入與輸出維度相容時直接相加；後者使用投影捷徑（Projection Shortcut）調整維度。
- 組合殘差區塊建立 **ResNet50**，理解卷積層、Batch Normalization、ReLU 與 Shortcut 加總的網路設計。
- 使用 **SIGNS 六類手勢資料集**進行分類，Notebook 記錄 **1,080 張訓練影像及 120 張測試影像**。

**實作結果與觀察：** 自行訓練的 ResNet50 經 **10 Epochs** 後，測試 **Loss 約 0.5515、Accuracy 80.83%**；另行載入課程提供的 `resnet50.h5` 權重後，測試 **Loss 約 0.1596、Accuracy 95.00%**。兩者來自**不同模型權重來源**，不應描述為同一模型經微調後由 80.83% 提升至 95.00%。此作業主要展現殘差連接的實作原理及不同權重設定下的分類結果。

### 4. Transfer Learning with MobileNetV2

**對應作業：** `Transfer_learning_with_MobileNet_v1.ipynb`（不公開）

以 **ImageNet 預訓練 MobileNetV2** 進行 **Alpaca／非 Alpaca** 二元影像分類，探討**遷移學習（Transfer Learning）**與**微調（Fine-tuning）**在小型資料集上的應用。

- 使用 `image_dataset_from_directory()` 載入 **327 張影像**，其中 **262 張用於訓練、65 張用於驗證**；輸入尺寸 **160 × 160**，Batch Size **32**。
- 使用 Random Flip、Random Rotation 進行資料增強，並套用 MobileNetV2 的輸入前處理。
- **第一階段：** 凍結預訓練 Backbone，透過 Global Average Pooling、Dropout（0.2）及新增的二元分類輸出層訓練分類器。
- **第二階段：** 將 Backbone 自第 **120** 層起的部分層設為可訓練，以較低學習率進行 Fine-tuning；呼叫模型時維持 Batch Normalization 相關的推論設定。

**實驗結果與觀察：** 第一階段訓練 **5 Epochs**，最後一輪驗證 **Loss 約 0.3697、Accuracy 73.85%**；接續微調至訓練紀錄顯示的 **Epoch 10/10**，最後一輪驗證 **Loss 約 0.1079、Accuracy 96.92%**。兩者皆是同一批驗證資料上的紀錄，**沒有額外的獨立測試集指標**。在此實驗設定下，微調後的驗證表現較佳，但仍須考慮資料量小及潛在標註誤差的影響。

### 5. Autonomous Driving — Car Detection

**對應作業：** `Autonomous_driving_application_Car_detection.ipynb`（不公開）

以自動駕駛場景的道路影像為例，使用**課程提供的預訓練 YOLO 模型**進行物件偵測，主要實作重點在**解讀模型輸出、篩選候選框及去除重複偵測**，而非從零訓練 YOLO。

- `yolo_filter_boxes`：依物件信心分數與類別機率篩選候選框。
- `iou`：計算 Intersection over Union（IoU），評估兩個邊界框的重疊程度。
- `yolo_non_max_suppression`：使用 Non-Max Suppression（NMS）處理同類別的重疊框。
- `yolo_eval`：整合候選框座標轉換、信心分數篩選、縮放及 NMS。
- `predict`：使用預訓練模型對圖片進行推論，並繪製物件類別、信心分數與邊界框。

**實作結果與觀察：** 模型輸入設定為 **608 × 608**，搭配 COCO 類別及 Anchor 資訊。Notebook 以 `images/test.jpg` 進行示範，輸出 **10 個偵測框**，包含 **8 個 car、1 個 bus 與 1 個 traffic light**，並顯示各框的預測信心分數。這是**單張影像的偵測展示**，不能當作整體資料集的 mAP、Precision 或 Recall。

### 6. Image Segmentation with U-Net

**對應作業：** `Image_segmentation_Unet_v2.ipynb`（不公開）

利用道路場景影像及其像素標註遮罩，建立 **U-Net 語意分割（Semantic Segmentation）**模型，學習以像素為單位辨識道路場景中的類別。

- 以道路場景的 RGB 影像與對應遮罩作為訓練資料，將影像調整為 **96 × 128 × 3**、遮罩為 **96 × 128 × 1**。
- **Contracting Path：** 使用卷積、下採樣及部分 Dropout 層擷取不同尺度的語意特徵。
- **Expanding Path：** 透過轉置卷積上採樣，搭配 **Skip Connections** 串接編碼器的同尺度特徵，保留定位資訊。
- 模型最後以 **1 × 1 卷積輸出 23 類的像素 logits**，使用 Adam 與 `SparseCategoricalCrossentropy(from_logits=True)` 訓練。

**實作結果與觀察：** Notebook 保存的訓練設定為 **Batch Size 32、5 Epochs**，最後一輪訓練 **Loss 0.4344、像素 Accuracy 87.74%**；並提供影像、標註與預測遮罩的視覺化程式。該流程沒有記錄獨立驗證／測試準確率或 mIoU，因此 **87.74% 僅代表當次訓練資料的像素準確率**，不宜直接推論為模型的實際泛化表現。

### 7. Face Recognition

**對應作業：** `Face_Recognition.ipynb`（不公開）

使用課程提供的**預訓練 FaceNet 模型**，將人臉影像轉換為 **128 維特徵向量（Embedding）**，練習人臉驗證（Verification）與人臉辨識（Recognition）。

- `triplet_loss`：實作 Anchor、Positive、Negative 的距離約束，理解 Triplet Loss 如何讓同一人特徵更接近、不同人特徵更遠；作業並未從零訓練整個 FaceNet。
- `img_to_encoding`：讀取及前處理影像，取得人臉特徵向量。
- `verify`：以輸入影像與指定身分的特徵距離完成 **1:1 驗證**，示範門檻設定為 **0.7**。
- `who_is_it`：將輸入特徵與已建立的人臉資料庫比較，完成 **1:N 辨識**。

**實作結果與觀察：** Notebook 的個別驗證案例中，與已登錄身分相符的照片距離約 **0.5993**，低於門檻而通過驗證；不相符的案例距離約 **1.0259**，因此未通過。`who_is_it` 示範亦能回傳最相近的資料庫身分。上述為**少數示範案例與函式測試**，不是在獨立人臉辨識基準資料集上統計得到的 Accuracy。

### 8. Neural Style Transfer

**對應作業：** `Art_Generation_with_Neural_Style_Transfer.ipynb`（不公開）

利用 **VGG19** 的中間特徵實作**神經風格轉換（Neural Style Transfer, NST）**，讓生成影像保留 Content Image 的主要場景結構，同時呈現 Style Image 的紋理與視覺風格。

- `compute_content_cost`：衡量生成影像與內容影像之間的特徵差異。
- `gram_matrix`、`compute_layer_style_cost`、`compute_style_cost`：使用 Gram Matrix 及多層卷積特徵描述、彙整風格差異。
- `total_cost`：加權整合 Content Loss 與 Style Loss，建立影像最佳化目標。
- `train_step`：使用 `tf.GradientTape` 與 Adam **直接更新生成影像的像素值**，而非重新訓練 VGG19 參數。

**實作結果與觀察：** Notebook 使用羅浮宮影像作為內容圖、繪畫作品作為風格圖，設定生成影像尺寸 **400 × 400**、學習率 **0.01**，執行 **2,501 次最佳化迭代**，並每隔 **250 次**顯示中間成果，觀察風格紋理逐步融入場景的過程。本作業屬於**影像生成的視覺實驗**，未提供分類 Accuracy 或客觀影像品質評估指標。

## 課程來源與參考資料

本頁整理的是課程作業的學習重點及執行紀錄，並非提供 Coursera 作業題目、完整程式碼或標準解答；課程教材及相關內容的權利歸其各自權利人所有。

- [NCKU — 集成卷積神經網路課程大綱（114-1，F724200）](https://class-qry.acad.ncku.edu.tw/syllabus/online_display.php?syear=0114&sem=1&co_no=F724200&class_code=)
- [DeepLearning.AI / Coursera — Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks)
- [Coursera Honor Code](https://www.coursera.org/about/terms)
- [TensorFlow / Keras 官方文件](https://www.tensorflow.org/api_docs)
- [ResNet — Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [MobileNetV2 — Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [YOLO — You Only Look Once](https://arxiv.org/abs/1506.02640)
- [U-Net — Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [FaceNet — A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- [Neural Style Transfer — A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
