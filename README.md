# Taiwan Property Price Regression
Institute: NYCU BMI 生物醫學資訊研究所｜ Student: 羅子崴 TZU-WEI, LO ｜ Student ID: 315112015

## 1. 專案介紹

本專案使用台灣房地產交易資料預測「總價元」，以平均絕對誤差（MAE）作為主要評估指標。我從面積與時間建立基準模型，再逐步加入地理位置、建物屬性、格局、車位、建案資訊與歷史成交參考，觀察不同資訊對預測的幫助。研究重點除了降低平均誤差，也包含辨識大誤差來源，以及檢查修正是否破壞原本準確的預測。主要閱讀入口為 **[Taiwan_Property_Price_Prediction.ipynb](Taiwan_Property_Price_Prediction.ipynb)**。這份 Notebook 保留資料處理、特徵分析、完整訓練程式、執行紀錄、圖表與實驗討論，原始中文欄位名稱予以保留。

## 2. 資料處理、特徵工程與版本演進

資料處理先統一欄位型別與語意：總價、面積、樓層及格局轉成數值；交易日期依民國年解析；「交易筆棟數」拆為土地、建物與車位數；地址整理成縣市、行政區與道路等位置鍵。非法文字與負面積保留為缺失，不虛構數值。已保存的資料處理結果為 Train 455,782 列、Provided Valid 11,994 列、Test 47,073 列；前處理後分別有 54、54、53 欄，Test 不含目標總價。這是前處理欄數，並非每版模型的特徵數。

| 欄位群組 | 原始資訊與衍生特徵 | 實驗目的 |
| :--- | :--- | :--- |
| 面積與時間 | 土地、建物、車位面積；交易年月、時間序號、月份 sin/cos | 建立交易規模與季節週期基準 |
| 地理位置 | 縣市、行政區、道路、粗地段與地址基地 | 區分不同位置的價格水準 |
| 型態與用途 | 建物型態、主要用途、交易標的、分區、建材 | 區分住宅、店面、辦公與工業交易 |
| 建物與交易結構 | 移轉樓層、總樓層、相對樓層、整棟／多樓層、房廳衛、屋齡、筆棟數、面積比 | 辨識物件配置與交易範圍 |
| 建案與棟別 | 建案名稱、建案行政區、棟別、名稱前綴 | 探索同區位不同產品的差異 |
| 交易語意與缺失 | 管理、隔間、備註衍生的特殊關係、裝潢、毛胚、持分、增建等旗標及缺失狀態 | 辨識可能改變價格解讀的交易條件 |
| 交互特徵 | 行政區×型態、行政區×用途、道路×型態、建案×型態、建案×棟別 | 讓同一屬性在不同情境具有不同定價關係 |
| 歷史成交參考 | 建案、道路與行政區的過往成交價格、支持樣本數與參考月齡 | 補充價格水準及參考可靠性；僅使用允許時間點前的訓練成交 |

月份週期以 `sin(2πm/12)` 與 `cos(2πm/12)` 表示，避免把 12 月和 1 月當成相距最遠的月份。特徵分析包含單欄位預測能力與多欄位交互比較，但單變數關聯弱不代表加入完整模型後沒有價值，重要性也不等同因果影響。

實際凍結清洗將房／廳／衛小於 0 或至少 20 的值轉為未知，這是本次相容規則，不代表商業物件不可能有 20 間房；「每間房至少一坪」並未取代此規則。Train 尚有 8 列缺有效月份、51,124 列缺建案名稱，Provided Valid 缺建案 1,340 列，Test 為 0 列。缺失與市場組成的差異因此列入解讀，而非宣稱補值後問題已消失。個別來源異常總價的排除也僅作來源相容處理，不作為通用房價下限。

實驗以時間切分評估模型對後續交易的預測能力，並分別觀察已見建案、未見建案與大面積交易。比較結果時，同時確認評估期間與樣本是否一致；特徵分析則用來判斷預測價值，不直接解讀為因果關係。

| 版本 | 實驗方向與處理結果 |
| :--- | :--- |
| V1–V6 | 從面積與時間開始，逐步加入位置、建物屬性、格局、車位與建案資訊 |
| V7–V8 | 建立多元件融合系統，加入特定交易的面積與計價單位處理 |
| V9 | 評估候選修正，未達採用條件，未匯出 submission |
| V10–V11 | 建立分位數殘差模型，再沿用模型調整受限修正政策 |
| V12–V13 | 檢查模型恢復流程並評估另一項候選修正；V13 公開分數較差，未採用 |
| V14 | 評估局部可比較案例修正，最終輸出與 V11 相同 |
| V15 | 外部 Test 對應成交總價的後處理紀錄，與獨立模型成果分開解讀 |

權重設計包含單價模型的面積樣本權重、部分元件的時間權重，以及模型之間的融合權重。三者分別處理總價誤差尺度、資料新舊程度與模型互補性，詳細公式及適用條件保留於 Notebook。我同時比較整體 MAE、分群誤差及原本準確樣本的變化，因此即使平均誤差下降，也不一定直接採用該版本。

## 3. 模型用途與權重設計

我以 CatBoostRegressor 作為主要迴歸模型，逐步加入特徵並保持基準參數一致，藉此觀察資訊本身的增益。後續加入 LightGBM 與不同目標的估價元件，檢查模型間是否存在互補誤差。這些元件不代表七種不同演算法，而是以不同資訊與估價途徑建立的模型。

| 模型或元件 | 預測用途 | 設計理由 |
| :--- | :--- | :--- |
| CatBoost 總價模型 | V1–V6 特徵實驗與後續總價元件 | 處理數值、類別與非線性交互作用，以 MAE 對齊總價評估 |
| `structural_total` | 從房產結構與基本屬性估價 | 提供不完全依賴成交參考的估價途徑 |
| `anchored_total` | 結合歷史成交錨點估計總價 | 引入價格水準，補充純屬性模型 |
| `recent_total` | 使用近期資訊的總價估計 | 檢查市場時間變化帶來的偏差 |
| `unit_price` | 預測每平方公尺價格，再乘回面積 | 補充直接總價模型對大面積交易的估計 |
| `comparable_total` | 引入可比較交易資訊 | 協助辨識相似產品的定價差異 |
| `prefix_total` | 加入建案名稱前綴資訊 | 探索產品定位；前綴不直接等同已確認建商 |
| `fine_location_total` | 細地點資訊的總價估計 | 補充行政區內部的空間差異 |
| 比較分支：History／Unit／Matched | 歷史參考、單價與匹配案例的 CatBoost 預測 | 以另一套估價組合提供互補資訊 |
| 比較分支：LightGBM | 與 CatBoost 融合的總價預測 | 檢查不同樹模型的誤差互補性 |
| V10 MultiQuantile 模型 | 估計標準化殘差的 q20、q50、q80 | 提供受限修正方向；V11 沿用此模型，不重新訓練 |

V1–V6 採固定 CatBoost 規格：MAE、2,400 棵樹、深度 8、學習率 0.04、L2 正則化 8、隨機種子 42。總價除以一百萬僅為尺度轉換，並非改用對數價格損失；後續各元件則保留其版本設定，不假設所有模型共用相同參數。

權重分成「訓練樣本權重」與「預測融合權重」。令 $A_i$ 為有效正建物面積、$y_i$ 為總價，單價目標與面積權重為：

$$
u_i=\frac{y_i}{A_i},\qquad w_i=\frac{A_i}{\overline A},\qquad
w_i\lvert u_i-\widehat u_i\rvert=\frac{\lvert y_i-A_i\widehat u_i\rvert}{\overline A}.
$$

因此，在相同有效樣本上，面積加權的單價絕對損失與總價 MAE 僅差固定倍率，避免大交易在總價上的誤差被低估。參考分支的 `anchored_total` 另使用 `max(0.25, 2^(-gap/36))`，再將權重正規化為平均 1；`gap` 為月齡，使近期資料較重要，同時保留舊交易的最低影響。這項時間權重並未套用於全部模型。

| 預測階段 | 固定規則 | 設計考量 |
| :--- | :--- | :--- |
| 參考／比較兩分支 | 原則上各 0.5 | 分散單一估價途徑的偏差；不是七元件各自等權 |
| 缺少參考的回退 | 一般交易的建案、道路、行政區參考皆缺時，採 100% 參考分支 | 避免缺乏支持的比較估價主導結果 |
| 比較分支的大面積單價融合 | `clip(log(A/300)/log(2), 0, 1)` | 300 m² 以下為 0、600 m² 起為 1，中間連續過渡 |
| Matched 修正 | 適用一般且具名未見建案交易，權重 0.1 | 小幅引入相似案例資訊 |
| LightGBM 融合 | `0.5 × 2^(-max(h-1, 0)/6)` | 最多 0.5；預測間隔 h 增加時逐步降低影響 |
| V10 中央修正 | 採用權重 0.25，並限制適用範圍與幅度 | 較大權重雖降 MAE，卻破壞較多原本準確預測 |
| V11 邊界修正 | 向最近 q20／q80 邊界移動，權重 0.5 | 區間內不動，減少向中央值過度修正 |

融合採分階段執行，各規則有適用條件，不能把不同階段的權重直接相加。V7 參考分支內另含結構／錨點、近期校正等混合，完整路徑保留於 Notebook。以上為保存的實驗規格，不代表每個常數均經獨立驗證為全域最佳。

V10 以 `(actual − raw_base)/(A × 1000)` 為殘差目標，再還原為價格分位數。V11 使用最近區間邊界 $b$ 修正基準價格 $p$：

$$
p'=p+0.5\times\mathrm{clip}(b-p,-C,C),\qquad C=\min(3{,}000{,}000,0.2p).
$$

只有超出區間的適用交易才修正。q20–q80 為模型預測分位數，不宣稱具有已校準的區間覆蓋率。

## 4. 採用依據與實驗觀察

我以相同評估樣本比較版本，並同步檢查原本準確群組與新增大誤差。V1–V6 是逐步特徵實驗；V7 同時改變模型組合與參考資訊，不能把其改善全部歸因於新增某一個特徵。

| 實驗證據 | 觀察與決策 |
| :--- | :--- |
| V1 → V2：Check MAE 4,646,840.845 → 2,456,683.887 | 地點資訊帶來明顯增益，支持後續研究位置與用途交互作用 |
| V4 → V5：Check MAE 2,139,550.725 → 2,139,171.758 | 建案資訊在此時間窗僅小幅改善，不能宣稱穩定大幅進步 |
| V5 → V6：Check MAE 升至 2,190,854.759 | 加入更多資訊不一定更好 |
| V6 → V7：Check MAE 降至 1,999,545.981，但 Dev 惡化 | 融合在較晚時間窗有幫助，並非全面勝出 |
| V10 六月選擇：權重 0.25 的準確群誤差變化 −0.960%；權重 0.5 為 +22.405% | 保留較小修正，不只追求最低整體 MAE |
| V10 → V11：相同 Check MAE 1,844,556.249 → 1,804,549.185 | 邊界修正改善約 40,007 元，但新增受損樣本由 5 增至 12 |
| V10 → V11：回報 Public MAE 1,901,809 → 1,901,325 | 公開分數僅改善 484 元，歷史改善不能直接外推 |

「原本準確」由已知評估目標事後定義，只用於診斷，不能作為 Test 推論時可直接辨識的條件。這些結果讓我將降低大誤差與保護小誤差視為需要共同衡量的目標，而不是以較新的版本或較低訓練損失判斷優劣。

## 5. 結果與使用限制

V11 的使用者回報 Kaggle Public MAE 為 1,901,325。本地歷史驗證與公開排行榜分數分開記錄，不假設兩者會以相同比例改善。V15 在後處理階段使用外部提供、與 Test 交易對應的成交總價，替換 946 筆預測。這些總價未用於模型訓練，但已用於最終輸出，因此V15 不列為獨立預測模型的泛化表現。相關來源、修改範圍與輸出檢核保留於匯出紀錄。

## 6. 檔案結構與執行方式

| 路徑 | 內容 |
| :--- | :--- |
| `Taiwan_Property_Price_Prediction.ipynb` | 完整實驗、訓練程式與保存的執行輸出 |
| `configs/` | 執行環境、版本設定與評估流程紀錄 |
| `data/processed/` | 處理後欄位結構等輔助資料 |
| `outputs/` | 各版本的預測、指標、圖表與匯出紀錄 |
| `presentation/` | 簡報與展示素材 |
| `references/` | 參考資料 |
| `src/` | 實驗使用的 Python 模組 |

Notebook 預設為 `REVIEW_ONLY = True`，可展示保存的歷史輸出，不重新訓練或執行模型推論；這是結果閱讀模式，不等於重新計算驗證。請保留 Notebook 的輸出內容，避免清除後儲存。完整訓練程式仍保留於同一份 Notebook，重新訓練需準備原始資料、套件與路徑設定。大型模型權重與快取未納入 Git，因此部分依賴既有模型的流程無法直接載入重現。研究過程使用 ChatGPT／Codex 協助討論特徵、撰寫與除錯程式，以及整理說明。我負責執行實驗、檢查輸出、比較結果與決定後續方向；AI 建議仍需透過資料檢核與實驗證據判斷。

---

## 7. Project Overview

This project predicts total transaction prices from Taiwan property transaction data using Mean Absolute Error (MAE), measured in New Taiwan dollars. I started with area and time features, then progressively introduced location, building attributes, layouts, parking, project information, and historical transaction references. The study examines both average error reduction and the effects of corrections on previously accurate predictions. The main entry point is **[Taiwan_Property_Price_Prediction.ipynb](Taiwan_Property_Price_Prediction.ipynb)**. It contains preprocessing, feature analysis, training code, historical execution records, figures, and discussion. Original Chinese column names are preserved.

## 8. Preprocessing, Features and Experimental Progression

Preprocessing standardizes numeric fields, parses Republic of China calendar dates, splits transaction counts into land/building/parking counts, and derives city, district, and road keys from addresses. Invalid numeric text and negative areas are treated as missing. The recorded processed populations are 455,782 Train rows, 11,994 Provided Valid rows, and 47,073 Test rows, with 54/54/53 preprocessing columns respectively. These counts are distinct from model-specific feature counts.

Feature groups cover area and time; location; property type, use, zoning and material; floor/layout/age/transaction structure; project and block identifiers; transaction semantics and missingness; and historical price references with support counts and age. Explicit interactions include district × type, district × use, road × type, project × type, and project × block. Month sin/cos encoding represents annual cyclicality. Historical references use training transactions available before the permitted reference time. Univariate associations are not treated as causal effects or complete measures of conditional predictive value.

The frozen layout cleaning maps negative or at-least-20 room/hall/bathroom counts to unknown; this is an experiment compatibility rule, not a universal validity claim. The one-ping-per-room hypothesis was not adopted in this frozen workflow. Recorded missing project names differ substantially across Train (51,124), Provided Valid (1,340), and Test (0), and eight Train rows lack valid transaction months. These composition differences remain limitations when interpreting performance.

Temporal splits evaluate predictions for later transactions, with separate analysis of seen projects, unseen projects, and large-area transactions. Comparisons account for differences in evaluation periods and populations. Feature associations are interpreted as predictive evidence rather than causal effects.

| Versions | Experimental focus and outcome |
| :--- | :--- |
| V1–V6 | Progressive feature expansion from area and time to property and project attributes |
| V7–V8 | Multi-component ensembles and transaction-specific area and pricing-unit handling |
| V9 | Candidate correction rejected under its adoption criteria |
| V10–V11 | Quantile residual modeling followed by a revised constrained correction policy |
| V12–V13 | Dependency recovery and another candidate correction; V13 was not adopted |
| V14 | Local comparable-case correction; final output identical to V11 |
| V15 | Test-aligned outcome-assisted postprocessing, evaluated separately from independent models |

Area-based sample weights, time weights, and ensemble weights address different aspects of the prediction problem. Their formulas and application conditions are documented in the notebook. Overall MAE is assessed alongside subgroup performance and changes to previously accurate predictions.

## 9. Models and Weight Design

CatBoostRegressor is the primary regression algorithm. V1–V6 use fixed settings—MAE, 2,400 trees, depth 8, learning rate 0.04, L2 regularization 8, and seed 42—to examine progressive feature additions. Dividing total prices by one million is a scale transformation, not a log-price objective. Later components retain their own recorded settings.

The reference branch contains seven estimation components: structural total price, historically anchored total price, recent total price, unit price, comparable transactions, project-name prefixes, and fine location. These are different estimation routes, not seven distinct algorithms. A comparison branch combines CatBoost History, Unit, and Matched components with LightGBM. V10 adds a MultiQuantile residual model (q20/q50/q80); V11 reuses it with a different correction policy.

For valid positive area, the unit target is `u = y/A` and the sample weight is `A/mean(A)`. Weighted unit absolute error is therefore proportional to total-price absolute error. The reference branch's `anchored_total` additionally uses `max(0.25, 2^(-gap/36))`, normalized to mean one. This recency weighting does not apply to every component.

| Prediction mechanism | Recorded rule | Purpose |
| :--- | :--- | :--- |
| Reference/comparison branches | Normally 0.5 each | Combine complementary estimation routes |
| Missing-reference fallback | 100% reference branch for eligible ordinary transactions missing project, road, and district references | Avoid unsupported comparison estimates |
| Large-area unit blend in the comparison branch | `clip(log(A/300)/log(2), 0, 1)` | Continuous transition between 300 and 600 m² |
| Matched component | 0.1 for eligible named unseen ordinary transactions | Introduce comparable evidence conservatively |
| LightGBM | `0.5 × 2^(-max(h-1, 0)/6)` | Reduce its influence as forecast horizon grows |
| V10 correction | Selected central correction weight 0.25 with eligibility and magnitude limits | Balance average improvement and damage |
| V11 correction | Move halfway toward the nearest q20/q80 boundary, with clipping | Preserve eligible predictions inside the interval |

These are conditional, sequential rules rather than weights to add across all stages. Additional internal reference-branch mixtures are retained in the notebook. Recorded constants are not claimed to be globally optimal. V10 learns `(actual − raw_base)/(A × 1000)`; V11 applies `p + 0.5 × clip(b − p, −C, C)` outside the interval, where `C = min(3,000,000, 0.2p)`. Predicted quantiles are not presented as calibrated coverage guarantees.

## 10. Experimental Evidence and Decisions

Location features reduce Check MAE from 4,646,840.845 in V1 to 2,456,683.887 in V2. Project features yield only a small Check improvement from V4 to V5, and V6 worsens despite more features. V7 improves Check performance but worsens Dev performance, so it is interpreted as a system-level trade-off rather than a universal improvement or a single-feature ablation.

In June, the V10 correction weight of 0.25 changes accurate-group MAE by −0.960%, while 0.5 increases it by 22.405%; the smaller correction is retained despite stronger overall gains at larger weights. V11 reduces same-cohort Check MAE from 1,844,556.249 to 1,804,549.185, while newly damaged rows increase from 5 to 12. Its reported public improvement is only 484, demonstrating limited transfer from historical gains. “Originally accurate” groups require evaluation labels and are diagnostic only, not deployable Test eligibility rules.

## 11. Results and Limitations

The user-reported Kaggle Public MAE for V11 is 1,901,325. Public leaderboard scores are recorded separately from historical validation results.

V15 replaces 946 predictions using externally supplied transaction prices aligned with Test records. These outcomes were not used for model fitting but were used in the final output. Therefore, V15 is excluded from claims about independent forecasting performance. Sources, modification scope, and output checks are retained in the export records.

## 12. Repository and Execution

The notebook contains the complete study. Supporting folders hold configurations (`configs/`), processed schema information (`data/processed/`), version-specific results (`outputs/`), presentation materials (`presentation/`), references (`references/`), and Python modules (`src/`).

The default setting, `REVIEW_ONLY = True`, displays stored historical outputs without model fitting or inference. It does not recompute or independently validate results. Saved notebook outputs must remain intact. Retraining requires the original data, dependencies, and path configuration. Large model weights and caches are excluded from Git, so some saved-model workflows cannot be reproduced directly from the committed files.

ChatGPT/Codex assisted with feature discussions, coding, debugging, and documentation. I executed the experiments, inspected outputs, compared results, and determined subsequent directions. AI-generated suggestions were assessed through data checks and experimental evidence.
