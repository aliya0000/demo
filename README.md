# demo
作品集嘗試
 (['from sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import classification_report',
  '!pip3 install kaggle kagglehub scikit-learn',
  'import kagglehub\n\n# Download latest version\npath = kagglehub.dataset_download("uciml/iris")\nprint("Path to dataset files:", path)# \n\n# Load the dataset\niris = load_iris()\n\n#呈現資料集\nprint(iris.data)\n\n'],
 ['# DSM 1121\n>主題: 【經典數據分析1: 鳶尾花(iris)資料集】<br>\n#K鄰近分類演算法(KNN)與資料視覺化\n\n本學期DSM主題預告: 鐵達尼號生存預測(11/28)、波士頓房價預測(12/5)、藥物與酒精作用資料集(12/12)',
  "## 1. 資料蒐集(獲取)\n\n### 1.1 Kaggle 資料集\nKaggle是一個數據建模和數據分析競賽平台，同時也是數據愛好者、機器學習工程師和科學家交流和學習的社群，提供了大量高品質的公開數據集，涵蓋廣泛的主題許多數據集供用戶下載。\n\nhttps://www.kaggle.com/\n\n### 1.2 要如何取得 Kaggle 上的資料集呢?\n\n1. 開啟 Kaggle 資料集上的 Notebook，將該 dataset 用 read_csv() 讀取出來，讀取方式輸入以下語法，接著就可以開始對 Dataframe (sales)進行資料分析。\n```python\nimport pandas as pd\nsales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')\n```\n2. 從 Kaggle API 下載資料集，首先要先安裝 Kaggle 套件，接著輸入指令，即可下載資料集。(適用於colab)\n\n3. 下載csv檔案到本地端，首先要先安裝 Kaggle 套件，接著輸入以下語法，即可下載資料集。\n\n\n### 1.3 Kaggle 資料集的對於初學數據科學的一大優勢: 乾淨、高品質\nKaggle 的數據集通常經過標準化處理，資料乾淨且結構明確：\n- 缺失值少或已處理過。\n- 資料格式清晰（如 CSV、JSON、圖像文件）。\n- 附有清楚的數據描述與元數據（Metadata），便於理解。\n\n\nSource: \n- https://nancysw.medium.com/%E5%85%A9%E7%A8%AE%E5%8F%96%E7%94%A8-kaggle-%E8%B3%87%E6%96%99%E9%9B%86%E7%9A%84%E6%96%B9%E6%B3%95-5944a1bcebf3\n- https://ithelp.ithome.com.tw/articles/10306093\n- https://ithelp.ithome.com.tw/articles/10219848\n\n",
  '### a) iris資料集的資訊吧'],
 10,
 14)
