# MusicGenre_FCU_Backend

應用網站 : https://musicgenre-fcu.netlify.app/

API文件 : https://musicgenre-fcu.herokuapp.com/docs

前端 : https://github.com/cookieopjax/MusicGenre_FCU

## 關於後端設置
1. 使用FastAPI的uploadfile接收一個檔案
2. 將此檔案存到伺服器中
3. 檢查這個檔案的型態，有需要時進行轉檔
4. 這個route會回傳轉換後的檔案名稱
5. 傳另一個request，進行機器學習，傳回種類


## 關於機器學習
### 如何利用 Dataset 中的 wav 檔去訓練模型 ?
* 利用 mfcc ( 梅爾倒頻譜係數 ) 對 wav檔做特徵的擷取得到特徵向量。
* 利用 numpy 對特徵向量做矩陣轉置，後再取共異變數(相關係數)。
* 將特徵向量取算術平均數
* 將 (算術平均數, 共異變數, 類別) 以 tuple 儲存
* 利用 pickle 將 tuple序列化成二進位的 dat 檔案(模型)

1. 利用KNN演算法去預測一個新資料的音樂類型
2. 將訓練集資料以及要預測的資料視為二維平面上的點
3. 然後先計算新資料與其他各個訓練集資料的距離 (特徵上的差異)
4. 將各個距離小到大排序好後，取前K個(5)
5. K個類別中，佔多數的為預測結果

此部分內容與開發 : [Huaaaa](https://github.com/a47894785), [kingofstar8787](https://github.com/kingofstar8787)
