# Hướng dẫn đội code — Thực nghiệm survey v2 (P0, E1–E6, sanity)

Phiên bản 25/10/2026 · Áp dụng cho gói `survey_v2/` + `scripts/run_survey_*.py`, `scripts/run_p0_perm_v3.py`,
`scripts/aggregate_p0_decision.py`, `scripts/run_sanity_auroc.py`, `scripts/make_survey_tables.py`.
Người nhận: đội code (LuanNT). Người viết bài: GVHD. Mọi thắc mắc về *thiết kế* hỏi GVHD trước khi đổi code.

---

## 0. Tóm tắt một trang

**Bài survey làm lại nói điều gì?** Bản v1 bị loại vì phần thực nghiệm không đứng được (S1–S4, mục 1). Bản v2
giữ phần tổng quan, thay toàn bộ thực nghiệm bằng một **bộ đánh giá có kiểm soát** trên bài toán phụ *giai đoạn sớm
(I) vs muộn (II–IV), chỉ mẫu u*, với ba bộ dữ liệu `gse68465_stage_early_late`, `tcga_luad_stage_early_late`,
`tcga_lusc_stage_early_late`. Thông điệp dự kiến (chỉ được viết sau khi E1–E6 xác nhận): *tín hiệu giai đoạn trong
transcriptome khối u yếu; chọn gen chỉ hơn chọn ngẫu nhiên một mức nhỏ và chỉ bền trên GSE68465; khi tín hiệu yếu,
giao thức đánh giá (chia theo bệnh nhân, mốc random-k, kiểm định hoán vị, độ ổn định có CI) quan trọng hơn việc
chọn phương pháp nào.*

**Đội code phải giao gì?** Kết quả của 8 việc, mỗi việc là một thư mục có `config.json` (commit hash, SHA-256 dữ liệu):

| # | Việc | Script | Trả lời | Hạn (Review 24/10, Bảng 17) |
|---|---|---|---|---|
| 1 | Sanity Y10 (đối chứng dương) | `run_sanity_auroc.py` | pipeline có hỏng không | 27/10 |
| 2 | P0 chạy lại có provenance | `run_p0_perm_v3.py` + `aggregate_p0_decision.py` | có tín hiệu / có ích / bền | 27/10 |
| 3 | E1 filter + embedded (3 bộ) | `run_survey_e1.py` | RQ1: hiệu năng theo ngân sách k | 03/11 |
| 4 | E2 chi phí tại một điểm; E3 đường cong theo n, p | `run_survey_e2_e3_timing.py` | RQ2: chi phí, số mũ thực nghiệm | 03/11 (E2), 06/11 (E3) |
| 5 | E1 wrapper (GA, BPSO, GSE68465, 3×5) | `run_survey_e1_wrappers.py` | RQ1 cho wrapper | 08/11 |
| 6 | E4 ổn định theo cỡ mẫu; E5 dư thừa, seed, chồng lấn | `run_survey_e4_e5_stability.py` | RQ3 | 06/11 |
| 7 | Bảng, hình, manifest (E6 ghép E1+E2+E4) | `make_survey_tables.py` | tất cả | sau mỗi đợt |
| 8 | Data card cho 3 bộ giai đoạn + bản chia 5×5 (`cv_splits.csv`) | (E1 tự ghi `cv_splits.csv`) | điều kiện 7 | 27/10 |

**Ba quy tắc không được vi phạm**

1. *Không chạm vào test*: mọi bước (lọc phương sai, chọn gen, chuẩn hóa, chọn C, chọn ngưỡng, fitness của wrapper)
   chỉ dùng phần train của fold. Code đã bố trí như vậy; không "tối ưu" lại.
2. *Không sửa hằng số đặt trước* trong `survey_v2/config.py` (k, seed, M, số hoán vị, lưới C...) nếu chưa được GVHD
   duyệt bằng văn bản. Nếu buộc phải sửa để chạy nổi, ghi lý do vào commit message và báo ngay.
3. *Mọi kết quả phải truy vết*: script chạy phải đã `git commit` + `git push`; `config.json` phải có `commit_hash`
   và `traceable = true`. Kết quả chạy với `KLTN_ALLOW_UNPUSHED=1` chỉ để thử, **không gửi**.

---

## 1. Bối cảnh: vì sao phải làm lại và làm lại cái gì

Review 11/10 (Bảng 4) chỉ ra bốn lỗi của thực nghiệm v1:

- **S1** — đơn vị mẫu là tế bào (single-cell GSE131907), nhãn ở mức bệnh nhân, không chia theo bệnh nhân → mô hình
  học "bệnh nhân/lô"; F1 ≈ 0,85 không phản ánh khả năng phân giai đoạn cho bệnh nhân mới.
- **S2** — lọc MI *theo nhãn* trên toàn bộ dữ liệu trước khi chọn gen (selection bias); độ ổn định 1,0000 ở cỡ
  20.000 gen → vô nghĩa.
- **S3** — một lần chạy, không SD/CI, không kiểm định, nhưng kết luận "mRMR is the most robust".
- **S4** — thời gian đo trên phần cứng không cố định; tiêu đề "RNA-Seq" trong khi GSE68465 là microarray.

Bảy điều kiện để kết quả mới được dùng (Review 11/10 Mục 5, điều chỉnh bởi QĐ6): (1) pipeline bulk chia theo bệnh
nhân, mọi bước trong fold; (2) GSE68465 bỏ 19 mẫu RNA tham chiếu, chỉ mẫu u; (3) P0 quyết định khung bài: ba bộ đạt
theo quy tắc đặt trước, chỉ GSE68465 đạt bền → khung "tín hiệu yếu"; (4) CV lặp theo bệnh nhân 5×5, mean ± SD, CI 95%,
kiểm định cặp Nadeau–Bengio + Holm; (5) thời gian/bộ nhớ trên một máy cố định, lặp ≥ 3, trung vị, đường cong theo n
và p; (6) ổn định bằng Nogueira có CI qua các lần lấy mẫu lại bệnh nhân, so ở cùng k; (7) mọi con số truy về một tệp
và một commit hash.

Từng thí nghiệm E1–E6 là một câu trả lời cho một điều kiện ở trên; mục 6 nêu "để chứng minh điều gì" cho từng cái.

---

## 2. Cài đặt và kiểm tra môi trường

```bash
git clone https://github.com/datachain-uit/RNA-Seq-FS-Survey.git
cd RNA-Seq-FS-Survey
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Thư mục dữ liệu (chứa experiment-data-perm/ và results/), KHÔNG nằm trong kho Git
export KLTN_DATA_ROOT=/path/to/KLTN                       # Windows PowerShell: $env:KLTN_DATA_ROOT = "D:\KLTN"

python -m pytest tests -q -m "not slow"                   # ~10 s: kiểm tra đơn vị
python -m pytest tests -q                                 # ~1 phút: thêm chạy đầu-cuối trên dữ liệu giả
python scripts/smoke_test_survey_v2.py                    # ~5–15 phút: chạy toàn bộ chuỗi trên dữ liệu giả
```

Dữ liệu cần có trong `$KLTN_DATA_ROOT/experiment-data-perm/<bộ>/`: `expression.h5` (key `expression`, mẫu × gen,
log2 gốc), `metadata.csv` (`sample_id`, `patient_id`, `class`, `label` ∈ {0,1}), và cho P0 thêm `perm_folds.csv`,
`folds/fold_<r>/{train,val,test}.h5` (do `scripts/build_perm_test_data.py` tạo). Bộ sanity: `tcga_normal_vs_tumor_sanity`.

Gói mới cần thêm: `skrebate` (ReliefF), `threadpoolctl`, `joblib`, `psutil`, `matplotlib` (đã thêm vào
`requirements.txt`). Nếu `pip install skrebate` lỗi trên Python 3.14, thử `pip install --no-deps skrebate` (skrebate
chỉ cần numpy/scipy/scikit-learn).

Kiểm tra nhanh cấu hình máy chạy E2–E3 (ghi vào biên bản): `python -c "from survey_v2.timing import machine_info; print(machine_info())"`.

---

## 3. Quy tắc truy vết (provenance) — đọc kỹ

Mọi script survey gọi `survey_v2.runlog.start_run(__file__, params)` ở đầu `main()`:

1. `scripts/provenance.py::provenance()` dừng nếu thư mục không phải git clone, hoặc có file **đã theo dõi** bị sửa
   chưa commit (file mới chưa `git add` không tính).
2. `check_traceable()` (đoạn code E2 của Review 24/10, sửa Y13): script đang chạy phải nằm trong Git index và commit
   HEAD phải có trên remote (`git branch -r --contains HEAD`). Nếu không → dừng với mã 2 và in lý do.
3. Mỗi thư mục kết quả có `config.json` với `commit_hash`, `dirty = false`, `traceable = true`, SHA-256 của
   `expression.h5`/`metadata.csv`, phiên bản thư viện, tham số dòng lệnh.

Quy trình đúng mỗi lần chạy thật:

```bash
git status                 # sạch
git pull --rebase
git log -1 --oneline       # ghi lại commit này vào biên bản
python scripts/run_survey_e1.py ...
```

`KLTN_ALLOW_UNPUSHED=1` chỉ dùng khi chạy thử; khi đó `config.json` ghi `traceable = false` và kết quả không được
trích. Vì sao chặt như vậy: kết quả P0 nhận ngày 24/10 ghi `commit_hash = "unknown"` nên **không trích được vào bài**
(nguyên tắc 7), phải chạy lại — đây là việc #2.

---

## 4. Cấu trúc code và luồng dữ liệu

```
survey_v2/
  config.py      hằng số đặt trước (seed 42–46, k ∈ {10,50,100,200,500}, p0 = 5000, M = 30, quần thể 30 × 50 thế hệ...)
  data.py        đọc bộ dữ liệu; chia CV 5×5 theo bệnh nhân; lọc phương sai TRONG fold; lấy mẫu con; bản sao tương quan
  methods.py     10 phương pháp chọn gen + random-k; giao diện chung select(method, X_train, y_train, k_list)
  mrmr.py        mRMR theo Peng 2005 (rời rạc hóa 3 mức, MID/MIQ), vector hóa bằng einsum
  wrappers.py    GA, BPSO mã hóa đúng k; fitness = AUROC CV 3 fold của LR-L2 trong phần train; đếm n_eval
  classify.py    LR-L2 (chính), RF, SVM tuyến tính; C và ngưỡng chọn bằng CV nội bộ; AUROC, BAcc, MCC
  stats.py       t-test Nadeau–Bengio, Holm, Bayesian correlated t-test (ROPE), CI
  stability.py   Nogueira (+ CI), Kuncheva, Jaccard, ma trận chồng lấn giữa phương pháp
  timing.py      đo wall/CPU/bộ nhớ đỉnh; số mũ log–log; thông tin máy
  runlog.py      provenance + check_traceable + thư mục kết quả + config.json
  synthetic.py   dữ liệu giả đúng định dạng (chỉ để test)
scripts/
  run_survey_e1.py, run_survey_e1_wrappers.py, run_survey_e2_e3_timing.py, run_survey_e4_e5_stability.py,
  make_survey_tables.py, run_p0_perm_v3.py, aggregate_p0_decision.py, run_sanity_auroc.py, smoke_test_survey_v2.py
tests/
  test_survey_v2_methods.py, test_survey_v2_stats_stability.py, test_survey_v2_scripts_e2e.py (slow)
```

Luồng của một fold trong E1 (giống nhau cho mọi phương pháp):

```
(train_idx, test_idx) theo bệnh nhân, seed r ∈ {42..46}, fold f ∈ {0..4}
   │
   ├─ B0: LR-L2 trên TOÀN BỘ gen (không lọc) ─────────────────────────────► AUROC_B0(r,f)
   │
   ├─ lọc phương sai KHÔNG nhãn trên train → p0 = 5000 gen (chỉ số cột `pre`)
   │
   ├─ random-k: 100 lần rút k gen trong p0, seed [cohort, r, f, k, draw] → LR-L2 ──► AUROC_rand(r,f,k,draw)
   │
   └─ method ∈ 10 phương pháp: select(method, X_train[:, pre], y_train, k_list)
        ├─ xếp hạng (F-test, MI, mRMR, ReliefF, RF importance, SVM-RFE): một lần, tập k = k gen đầu (lồng nhau)
        ├─ theo k (L1-LR, Elastic Net): đúng k hệ số khác 0 (tinh chỉnh C), bão hòa → bổ sung F-test, ghi n_filled
        └─ wrapper (GA, BPSO): tìm tập đúng k gen, fitness CV nội bộ trên train
        → với từng k: LR-L2 (C bằng CV nội bộ 3 fold, ngưỡng BAcc trên OOF) fit train → đo test ──► AUROC(r,f,k)
          (k ∈ {50, 200} thêm RF, SVM tuyến tính nếu --secondary)
```

Kết quả: `results/survey_v2/E1/<bộ>/e1_per_fold.csv` (một dòng = một (r, f, method, k, classifier)),
`e1_selected_genes.csv` (gen đã chọn, có rank), `e1_random_k.csv`, `e1_scores.csv` (điểm từng mẫu test), `cv_splits.csv`,
`config.json`. `make_survey_tables.py` đọc các tệp này.

---

## 5. Giao thức chung — "vì sao" cho từng lựa chọn

| Lựa chọn | Làm gì trong code | Vì sao (để giải thích khi được hỏi) |
|---|---|---|
| CV 5×5 phân tầng theo **bệnh nhân**, seed 42–46 | `data.repeated_patient_splits` (`StratifiedKFold` khi 1 mẫu/bệnh nhân, `StratifiedGroupKFold` khi nhiều) + assert không trùng bệnh nhân | Sửa S1. Lặp 5 lần để có SD/CI; cùng lần chia cho mọi phương pháp → so sánh cặp. |
| Lọc sơ bộ 5.000 gen **phương sai lớn nhất, trên train, không dùng nhãn** | `data.variance_prefilter` gọi trong fold | Sửa S2: lọc theo nhãn trên toàn bộ dữ liệu = rò rỉ. Lọc không nhãn chỉ giảm chi phí, không "nhìn" test. |
| Mốc **random-k** (100 lần rút/fold) | `methods.random_panel`, seed `[cohort, seed, fold, k, draw]` (Y14) | Câu hỏi thật là "chọn gen có hơn chọn bừa k gen không"; tín hiệu yếu thì mốc này quan trọng hơn mốc "không chọn". |
| Mốc **B0** (toàn bộ gen, không lọc) | `evaluate_panel(X_full)` | Cho biết chọn gen có *mất* thông tin so với dùng tất cả không. |
| Bộ phân loại chính LR-L2, C ∈ {0,01; 0,1; 1; 10} bằng CV nội bộ 3 fold theo AUROC; ngưỡng chọn trên OOF | `classify.evaluate_panel` | Tách bạch "chất lượng tập gen" khỏi "sức mạnh bộ phân loại"; ngưỡng không chọn trên test để BAcc/MCC không lạc quan. |
| RF, SVM tuyến tính chỉ ở k ∈ {50, 200} | `--secondary` | Kiểm tra kết luận có phụ thuộc bộ phân loại không, với chi phí vừa phải. |
| AUROC chính; BAcc, MCC phụ | `classify` | AUROC không phụ thuộc ngưỡng, hợp với lớp lệch; BAcc để so với P0 (dùng BAcc). |
| Chuẩn hóa z-score **fit trên train** trước L1-LR, Elastic Net, SVM-RFE (M7) | `methods.select` | Phương pháp phạt hệ số phụ thuộc thang đo; fit trên test là rò rỉ. |
| Cùng tập gen cho mọi k của phương pháp xếp hạng (tiền tố) | `methods.prefix_panels` (assert lồng nhau) | Đúng định nghĩa "ngân sách k" cho filter; tránh chạy lại không cần thiết. |
| Wrapper chỉ GSE68465, 3 lần lặp đầu (3×5), quần thể 30 × 50 thế hệ | `run_survey_e1_wrappers.py` | Chi phí; GSE68465 là bộ duy nhất được xếp hạng phương pháp (QĐ6). Ghép cặp với các phương pháp khác trên đúng 15 fold đó. |

---

## 6. Mười phương pháp — cài đặt, tham số, điều cần để ý

Ký hiệu: n mẫu train, p gen (sau lọc 5.000), k ngân sách. Hàm trong `survey_v2/methods.py` trừ khi ghi khác.

**Filter**

1. **F-test (ANOVA)** — `rank_f_test`: `sklearn.feature_selection.f_classif`, xếp giảm dần F (NaN xuống cuối).
   O(np). Không có tham số. Là mốc "rẻ nhất" và là thứ tự dự phòng khi L1 bão hòa.
2. **Mutual information** — `rank_mutual_info`: `mutual_info_classif(n_neighbors=3, random_state=seed)` (ước lượng
   KNN của Kraskov/Ross). Chậm hơn F-test đáng kể (vòng lặp theo gen); seed cố định vì ước lượng có nhiễu.
3. **mRMR (Peng, Long & Ding 2005)** — `mrmr.mrmr_ranking`: rời rạc hóa từng gen thành 3 mức theo μ ± σ/2 *tính trên
   train*; MI từ bảng tần suất; chọn tham lam theo **MID** (`I(X;Y) − mean I(X;X_s)`) — tiêu chí MIQ có sẵn
   (`config.MRMR_CRITERION`). Chỉ xếp k_max = 500 gen đầu (đủ cho mọi k); chi phí O(k·n·p) nhờ einsum one-hot.
   *Lưu ý:* Bảng 5 của bài ghi O(p²n) cho mRMR; cài đặt tham lam k gen là O(k n p), nên số mũ theo p đo được kỳ vọng ≈ 1.
   Đây là điểm người viết sẽ bàn ở E3, không phải lỗi.
4. **ReliefF (Kononenko 1994; skrebate)** — `rank_relieff`: `skrebate.ReliefF(n_neighbors=10, n_jobs=1)`, lấy
   `feature_importances_`. O(n²p) — chậm nhất trong nhóm filter; với n ≈ 350, p = 5.000 ước chừng vài phút mỗi fold.

**Embedded**

5. **L1-LR (Lasso logistic)** — `select_sparse_exact_k(kind="l1_lr")`: `LogisticRegression(penalty="l1",
   solver="liblinear", class_weight="balanced")`. Fit đường hệ số trên lưới 26 giá trị C (log 10⁻³…10²), tìm C cho
   **đúng k** hệ số khác 0 (tinh chỉnh nhị phân 6 bước giữa hai điểm lưới; nếu không hit đúng, lấy k hệ số |β| lớn nhất
   của nghiệm có ≥ k hệ số, ghi `exact_k = False`). Khi L1 **bão hòa** (số hệ số khác 0 tối đa < k, thường vì k > n), bổ
   sung theo F-test và ghi `n_filled` (KBTN 5.1). Người viết sẽ báo cáo `n_filled` — không được giấu.
6. **Elastic Net (logistic, l1_ratio = 0,5)** — như trên với `solver="saga"`, `max_iter=3000`, `tol=1e-3`. Chậm hơn L1
   (saga trên 350 × 5.000); nếu quá chậm, báo trước khi đổi `tol`.
7. **RF importance (Breiman 2001)** — `rank_rf_importance`: `RandomForestClassifier(n_estimators=500,
   class_weight="balanced_subsample", random_state=seed)`, xếp theo Gini importance. Ngẫu nhiên theo seed → được đo
   trong E5b.
8. **SVM-RFE (Guyon 2002)** — `rank_svm_rfe`: `LinearSVC(C=1, class_weight="balanced")` trên dữ liệu z-score; mỗi vòng
   loại 10 % gen có w² nhỏ nhất cho đến còn k_min = 10; **thứ tự loại là xếp hạng** (gen bị loại sau = quan trọng hơn),
   nên tập k gen là tiền tố và lồng nhau. Khác với P0 (loại một nửa mỗi vòng, dừng đúng tại từng k): survey cần xếp hạng
   chặt để so cùng k với các phương pháp khác.

**Wrapper** (`survey_v2/wrappers.py`; sửa O4 của Review 04/10)

9. **GA (Siedlecki & Sklansky 1989)** — mã hóa *tập chỉ số đúng k gen*; quần thể 30, 50 thế hệ, tournament 3, giữ 2
   cá thể tốt nhất, lai ghép giữ phần giao của cha mẹ rồi bổ sung ngẫu nhiên từ hiệu đối xứng, đột biến đổi gen với xác
   suất 1/k mỗi gen.
10. **BPSO (Kennedy & Eberhart 1997)** — vận tốc cập nhật theo pbest/gbest (w giảm 0,9 → 0,4; c1 = c2 = 2; |v| ≤ 6),
    x_j = 1 với xác suất sigmoid(v_j), rồi **sửa về đúng k gen** (thừa: bỏ gen có sigmoid nhỏ nhất; thiếu: thêm gen có
    sigmoid lớn nhất).

Fitness của cả hai = AUROC trung bình CV 3 fold phân tầng của LR-L2 (C = 1, có chuẩn hóa) **bên trong phần train**;
có cache theo tập gen; `n_eval` đếm số lần đánh giá thật (báo cáo chi phí không phụ thuộc phần cứng, O7). Phần test
không bao giờ được đọc trong tìm kiếm — test `test_fitness_uses_only_training_rows` kiểm điều này.

**Tham chiếu**: `b0` (toàn bộ gen), `random_k` (k gen ngẫu nhiên trong 5.000 gen đã lọc).

---

## 7. Từng thí nghiệm: mục đích, lệnh, đầu ra, tiêu chí chấp nhận

Mọi lệnh chạy từ thư mục gốc kho, sau khi `export KLTN_DATA_ROOT=...`. Kết quả vào `$KLTN_DATA_ROOT/results/survey_v2/`.
Các script **tự tiếp tục** (resume) khi chạy lại: phần (repeat, fold, method) đã có được bỏ qua.

### 7.1 Sanity Y10 — đối chứng dương (chạy đầu tiên)

- *Chứng minh:* pipeline survey (CV theo bệnh nhân, lọc trong fold, k = 10, LR-L2) phát hiện được tín hiệu mạnh
  normal vs tumor. Nếu không đạt, mọi kết quả yếu ở bài toán giai đoạn đều đáng ngờ.
- *Lệnh:* `python scripts/run_sanity_auroc.py`
- *Đầu ra:* `sanity/sanity_per_fold.csv`, `sanity_summary.csv`, `config.json`; mã thoát 0 nếu đạt.
- *Chấp nhận:* AUROC trung bình ≥ 0,99 cho cả f_test, l1_lr, rf_importance. Nếu < 0,99: KHÔNG sửa ngưỡng; báo GVHD.

### 7.2 P0 — permutation v3 chạy lại từ kho (việc #2)

- *Chứng minh:* ba bộ giai đoạn có tín hiệu thật (p hoán vị) và chọn gen có hơn random-k (gain ≥ 0,02) hay không;
  bộ nào "bền". Kết quả quyết định khung bài (QĐ6) nên phải có commit hash.
- *Lệnh (giữ nguyên thiết kế notebook, chỉ cấu hình chính `balanced`):*

  ```bash
  python scripts/run_p0_perm_v3.py --n-jobs 8            # 3 bộ giai đoạn, 1000 hoán vị, 1000 random-k
  python scripts/aggregate_p0_decision.py --summaries "$KLTN_DATA_ROOT"/results/survey_v2/P0/perm_test_v3/*/summary.csv
  ```

  Thêm `--class-weights balanced none` nếu GVHD yêu cầu lại sensitivity. Script dừng an toàn theo `--time-budget-h`
  và tiếp tục khi chạy lại.
- *Đầu ra:* `P0/perm_test_v3/<bộ>/{summary.csv, pvalues_adjusted.csv, decision.csv, null_scores_per_fold.csv,
  random_k_scores_per_fold.csv, real_*.csv, config.json}`; `P0/{p0_summary_merged.csv, p0_decision.csv, p0_decision.md, config.json}`.
- *Chấp nhận:* `n_perm = n_random = 1000` ở mọi dòng; `family_complete = true`; `commit_hash` khác null. Đối chiếu
  với bảng cũ (Review 20/10, Bảng 4): GSE68465 5/15 có tín hiệu, 5/15 có ích, bền; TCGA-LUAD 7/15, 1/15, không bền;
  TCGA-LUSC 8/15, 1/15, không bền. Vì cùng seed và cùng fold, `bacc_real` phải trùng bản cũ đến sai số làm tròn; nếu
  khác, báo ngay (khác dữ liệu hoặc khác phiên bản sklearn).
- *Thời gian:* notebook cũ cần ~10 giờ Kaggle cho cả hai lượt; trên máy 8 nhân ước lượng 3–6 giờ.

### 7.3 E1 — hiệu năng theo ngân sách k (việc #3, #5)

- *Chứng minh (RQ1):* với tín hiệu yếu, phương pháp chọn gen nào hơn random-k và B0 ở mức k nào; khác biệt giữa các
  phương pháp có ý nghĩa không (chỉ kiểm định trên GSE68465).
- *Lệnh:*

  ```bash
  python scripts/run_survey_e1.py --secondary --n-jobs 5                     # 3 bộ, 8 phương pháp, B0, random-k
  python scripts/run_survey_e1_wrappers.py --n-jobs 5                        # GA + BPSO, GSE68465, repeats 0 1 2
  ```

  Có thể chia nhỏ: `--datasets ...`, `--methods ...`, `--repeats 0 1`. Chạy lại cùng lệnh để tiếp tục.
- *Đầu ra:* `E1/<bộ>/{e1_per_fold.csv, e1_selected_genes.csv, e1_random_k.csv, e1_scores.csv, cv_splits.csv, config.json}`.
- *Chấp nhận:*
  - `e1_per_fold.csv`: mỗi (repeat, fold, method, k) có đúng 1 dòng `lr_l2` (và `rf`, `svm_linear` ở k ∈ {50, 200});
    25 fold cho filter/embedded/B0, 15 fold cho GA/BPSO; `n_selected == k`; `auroc ∈ [0, 1]`.
  - `e1_random_k.csv`: 100 dòng mỗi (repeat, fold, k).
  - `cv_splits.csv`: không bệnh nhân nào vừa train vừa test trong cùng (repeat, fold) — kiểm bằng
    `tests/test_survey_v2_scripts_e2e.py` (đã có) hoặc một dòng pandas.
  - `n_filled > 0` chỉ xuất hiện ở l1_lr/elastic_net khi k lớn hơn số hệ số L1 có thể khác 0; ghi nhận, không xóa.
  - `config.json`: `traceable = true`, `prefilter_p0 = 5000`, `random_k_draws = 100`.
- *Thời gian ước lượng* (một luồng/fold, n ≈ 350, p0 = 5.000): F-test < 1 s; mRMR ~ 1–5 s; MI ~ 1 phút; RF ~ 10–20 s;
  SVM-RFE ~ 1–2 phút; L1 ~ 1–2 phút; Elastic Net ~ 5–10 phút; ReliefF ~ 5–10 phút; random-k 100 × 5 k × LR ≈ 1–2 phút;
  RF phụ 500 cây ~ vài phút. Khoảng 20–30 phút/fold → 75 fold ≈ 25–35 giờ CPU, chia 5 tiến trình ≈ 5–7 giờ.
  Wrapper: ~1.500 lần đánh giá fitness × 3 fit LR mỗi (fold, k) → 5–15 phút/(fold, k) → 15 fold × 5 k × 2 ≈ 15–40 giờ
  CPU, chia 5 tiến trình ≈ 3–8 giờ.

### 7.4 E2 + E3 — chi phí và đường cong độ phức tạp (việc #4)

- *Chứng minh (RQ2):* chi phí thật của từng phương pháp tại một điểm cố định (bảng), và số mũ thực nghiệm theo n và p
  để đối chiếu với bậc lý thuyết của Bảng 5 (v1 chỉ có lý thuyết). Wrapper: chi phí theo số lần đánh giá (10×20 vs 30×50).
- *Lệnh (MỘT máy cố định, không chạy gì khác song song, BLAS 1 luồng — script tự đặt biến môi trường):*

  ```bash
  python scripts/run_survey_e2_e3_timing.py --wrappers
  ```
- *Đầu ra:* `E2_E3/{e2_cost_point.csv, e3_scaling_n.csv, e3_scaling_p.csv, e3_wrapper_evals.csv, e3_exponents.csv,
  machine_config.json, config.json}`.
- *Chấp nhận:* `repeats = 3` (trung vị); `machine_config.json` có tên CPU, RAM, hệ điều hành, `blas_threads_env` = "1";
  `wall_spread_max_rel` (dao động giữa 3 lần) < 0,3 cho các phương pháp > 1 s — nếu lớn hơn, máy không ổn định, chạy lại
  lúc rảnh. Thời gian chỉ gồm bước chọn gen (không gồm bộ phân loại sau chọn).
- *Thời gian:* 8 phương pháp × (1 + 4 + 5) điểm × 3 lần ≈ 1–3 giờ; thêm wrapper (2 × 2 ngân sách × 3 lần) ≈ 1–2 giờ.

### 7.5 E4 + E5 — độ ổn định (việc #6)

- *Chứng minh (RQ3):* (E4) tập gen có lặp lại được khi lấy mẫu lại bệnh nhân không, và phụ thuộc cỡ mẫu thế nào;
  random-k cho sàn ≈ 0. (E5a) dư thừa làm phương pháp nào bất ổn (chọn qua lại giữa bản sao); (E5b) GA/BPSO/RF đổi seed
  thì tập gen và AUROC đổi bao nhiêu; (E5c) các phương pháp có chọn cùng gen không.
- *Lệnh:*

  ```bash
  python scripts/run_survey_e4_e5_stability.py --n-jobs 5                    # E4 + E5a + E5b trên 3 bộ, E5c nếu E1 đã có
  python scripts/run_survey_e4_e5_stability.py --wrappers --n-jobs 5         # (tùy chọn) GA/BPSO trong E4 tại f = 0,7, k = 50
  ```
- *Đầu ra:* `E4_E5/<bộ>/{subsamples.csv, e4_selected_genes.csv, e4_stability.csv, e4_select_times.csv,
  e5a_selected_genes.csv, e5a_redundancy.csv, e5b_seed_runs.csv, e5b_seed_stability.csv, e5c_overlap_k50.csv, config.json}`.
- *Chấp nhận:* `M = 30` ở mọi dòng `e4_stability.csv`; `nogueira` của `random_k` nhỏ và tăng nhẹ theo k (sàn thực
  nghiệm: random-k rút trong 5.000 gen đã lọc nhưng ổn định tính trên toàn bộ gen, nên sàn ≈ k/5.000, tức ≈ 0 ở k = 10
  và ≤ ≈ 0,1 ở k = 500); mọi phương pháp phải cao hơn sàn này rõ rệt (CI không chồng lấn) ở ít nhất một k; `nogueira ∈
  [−1, 1]`; E5a có đủ ρ ∈ {0,5; 0,8; 0,95}; E5b có 5 seed × (rf_importance ở 5 k; ga, bpso ở k = 50).
- *Thời gian:* E4 = 3 tỉ lệ × 30 tập con × 8 phương pháp (giống một fold E1 bỏ phần phân loại) ≈ 90 × 10 phút ≈ 15 giờ
  CPU/bộ → chia 5 tiến trình ≈ 3 giờ/bộ; E5a ≈ E4 ở một tỉ lệ; E5b nhỏ.

### 7.6 Bảng, hình, E6 (việc #7)

```bash
python scripts/make_survey_tables.py
```

Đọc E1, E2_E3, E4_E5 (phần nào có thì làm), ghi `tables/*.csv|*.tex`, `figures/*.pdf|*.png` và
`tables/provenance_manifest.json` (SHA-256 + commit hash của từng tệp đầu vào). E6 = ghép AUROC (E1) – Nogueira (E4 ở
tỉ lệ 0,7) – thời gian (E2, k = 50) và đánh dấu mặt Pareto. Chạy lại sau mỗi đợt kết quả mới; gửi cả thư mục `tables/`
và `figures/`.

### 7.7 Data card và bản chia (việc #8)

Mỗi bộ giai đoạn cần `DATA_CARD.md` (nguồn, lọc, số mẫu/lớp/bệnh nhân, cách gán nhãn sớm/muộn, phiên bản dữ liệu, SHA-256)
theo mẫu của `experiment-data-v2/datasets/*/DATA_CARD.md`; `cv_splits.csv` do E1 sinh là bản chia 5×5 chính thức.

---

## 8. Thống kê dùng trong bài — hiểu để kiểm tra đầu ra

- **p hoán vị** (P0): p = (#{hoán vị có BAcc ≥ thật} + 1)/(B + 1) (Phipson & Smyth 2010). p nhỏ nhất = 1/(B+1).
- **Holm (1979)**: hiệu chỉnh đa kiểm định trong *từng họ*. P0: họ = (bộ, tác vụ, class_weight, loại kiểm định), 15
  kiểm định. E1: họ = một phương pháp × 5 mức k. Cột `*_holm`.
- **t-test hiệu chỉnh Nadeau–Bengio (2003)**: với J = 25 chênh lệch d_j của CV lặp, phương sai của d̄ không phải s²/J
  mà là (1/J + n_test/n_train)·s² vì các tập train chồng lấn. t = d̄ / √((1/J + 1/4)·s²), bậc tự do J − 1. Bảo thủ hơn
  t-test thường — đúng ý đồ. Hàm `stats.nadeau_bengio_ttest`; dùng cho "vs random-k" (một phía) và "vs B0" (hai phía).
- **CI 95%** của mean AUROC cũng dùng phương sai Nadeau–Bengio (`stats.mean_ci`).
- **Bayesian correlated t-test** (Corani & Benavoli 2015; Benavoli et al. 2017): hậu nghiệm Student-t của chênh lệch
  trung bình, báo P(left), P(ROPE), P(right) với ROPE = ±0,01 AUROC — trả lời "hai phương pháp có tương đương về thực
  tiễn không", điều p-value không nói được. Chỉ cho các cặp dẫn đầu trên GSE68465.
- **Nogueira, Sechidis & Brown (2018)**: Φ = 1 − (M/(M−1))·mean_f[p_f(1−p_f)] / ((k̄/d)(1−k̄/d)) với Z là ma trận
  M × d (M lần chọn, d = *toàn bộ* gen của bộ dữ liệu). Φ = 1: luôn chọn cùng tập; Φ ≈ 0: như ngẫu nhiên. CI tiệm cận chỉ
  hợp lệ khi M lần chọn độc lập → E4 dùng M = 30 tập con độc lập (không dùng fold của CV).
- **Kuncheva (2007)**, **Jaccard**: đối chiếu với v1; Kuncheva cần các tập cùng kích thước.
- Người viết chỉ được viết "tốt hơn" khi `p_holm < 0,05`; nếu không, viết "không có bằng chứng khác biệt".

---

## 9. Bàn giao cho người viết

Copy (không đổi tên) các thư mục sau vào `input/results/` của không gian viết bài, kèm `config.json`:

```
input/results/
  sanity/                 sanity_summary.csv, sanity_per_fold.csv, config.json
  P0/                     p0_decision.csv, p0_summary_merged.csv, p0_decision.md, config.json
  P0/perm_test_v3/<bộ>/   summary.csv, decision.csv, pvalues_adjusted.csv, config.json
  E1/<bộ>/                e1_per_fold.csv, e1_selected_genes.csv, e1_random_k.csv, cv_splits.csv, config.json
  E2_E3/                  e2_cost_point.csv, e3_scaling_n.csv, e3_scaling_p.csv, e3_wrapper_evals.csv, e3_exponents.csv, machine_config.json, config.json
  E4_E5/<bộ>/             e4_stability.csv, e5a_redundancy.csv, e5b_seed_stability.csv, e5c_overlap_k50.csv, config.json
  tables/, figures/       toàn bộ (kèm provenance_manifest.json)
  data_cards/             DATA_CARD.md của 3 bộ giai đoạn + sanity
```

Kèm một tệp `HANDOVER.md` ghi: commit hash dùng cho từng thư mục (`git log -1`), ngày chạy, máy chạy (E2–E3), thời gian
thực tế, các cảnh báo in ra khi chạy, và mọi sai lệch so với hướng dẫn này. Không gửi tệp `e1_scores.csv` (lớn) trừ khi được hỏi.

Checklist trước khi gửi:

- [ ] `python -m pytest tests -q` xanh trên máy chạy.
- [ ] Mọi `config.json` có `traceable: true`, `dirty: false`, `commit_hash` khớp `HANDOVER.md`.
- [ ] Sanity đạt (≥ 0,99).
- [ ] P0 khớp Bảng 4 (Review 20/10) về số "có tín hiệu / có ích / bền".
- [ ] E1: 25 fold (15 cho wrapper) × 5 k đủ cho mọi phương pháp và bộ; `cv_splits.csv` không trùng bệnh nhân.
- [ ] E2–E3: `machine_config.json` đầy đủ; dao động giữa 3 lần đo nhỏ.
- [ ] E4: M = 30; random-k ở sàn (≈ k/5.000) và thấp hơn mọi phương pháp.
- [ ] `make_survey_tables.py` chạy không lỗi; `provenance_manifest.json` liệt kê mọi tệp đầu vào.

---

## 10. Lịch gợi ý (theo Review 24/10, Bảng 17)

| Ngày | Việc | Ghi chú |
|---|---|---|
| 25–27/10 | Cài môi trường, `pytest`, smoke test; sanity Y10; chạy lại P0 (#1, #2, #8) | P0 có thể chạy đêm |
| 28/10–03/11 | E1 filter + embedded trên 3 bộ (#3); E2 + E3 trên máy cố định (#4) | E2–E3 chạy khi máy rảnh |
| 04–06/11 | E4 + E5 (#6); `make_survey_tables.py` lần 1 (#7) | gửi bản tạm cho người viết |
| 04–08/11 | E1 wrapper (#5) | chạy song song với việc trên |
| 08–10/11 | `make_survey_tables.py` lần cuối; bàn giao đủ `input/results/` + `HANDOVER.md` | bản đầy đủ bài 13/11 |

---

## 11. Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Xử lý |
|---|---|---|
| `[traceable] DỪNG: script chưa được Git theo dõi / HEAD chưa có trên remote` | chưa commit/push | `git add`, `git commit`, `git push` rồi chạy lại. Không dùng `KLTN_ALLOW_UNPUSHED=1` cho lần chạy thật. |
| `ProvenanceError: Còn thay đổi chưa commit` | sửa file đã theo dõi | commit hoặc `git stash`. |
| `FileNotFoundError .../experiment-data-perm/<bộ>/expression.h5` | `KLTN_DATA_ROOT` sai | kiểm tra biến môi trường; thư mục phải chứa `experiment-data-perm/`. |
| `ValueError: ... không cùng thứ tự sample_id` | expression.h5 và metadata.csv lệch | dựng lại bộ dữ liệu bằng `build_perm_test_data.py`, không sửa tay. |
| `ModuleNotFoundError: skrebate` | chưa cài | `pip install skrebate` (hoặc `--no-deps`). |
| E1 chạy rất chậm | nhiều luồng BLAS tranh nhau khi `--n-jobs > 1` | script đã giới hạn 1 luồng/fold; đặt thêm `OMP_NUM_THREADS=1` ở shell; `--n-jobs` ≤ số nhân vật lý. |
| Elastic Net cảnh báo hội tụ | saga chưa hội tụ ở C lớn | đã tắt cảnh báo; không đổi `tol`/`max_iter` nếu chưa hỏi. |
| `n_filled > 0` ở L1-LR | L1 bão hòa (k > số hệ số khác 0 tối đa ≈ n) | bình thường; người viết sẽ báo cáo. |
| Nogueira của random-k cao hơn sàn k/5.000 rõ rệt (ví dụ > 0,2) hoặc âm mạnh | lỗi dựng ma trận Z hoặc không gian gen | báo GVHD; không sửa số. |
| Thời gian E2 dao động mạnh giữa 3 lần | máy đang bận / tiết kiệm năng lượng | chạy lại khi máy rảnh, cắm điện, tắt turbo nếu được. |

---

## Phụ lục A. Tham số đặt trước (`survey_v2/config.py`) — đối chiếu KBTN v2.1

| Nhóm | Tham số | Giá trị |
|---|---|---|
| Dữ liệu | bộ xếp hạng / mô tả | GSE68465 / TCGA-LUAD, TCGA-LUSC; E2–E3 trên TCGA-LUAD |
| CV | seed lặp; số fold; wrapper | 42, 43, 44, 45, 46; 5; 3 lần lặp đầu |
| Ngân sách | k | 10, 50, 100, 200, 500 |
| Lọc sơ bộ | p0 | 5.000 gen phương sai lớn nhất trên train |
| Random-k | số lần rút | 100 mỗi (fold, k); seed `[cohort, seed, fold, k, draw]` |
| Bộ phân loại | LR-L2 C; CV nội bộ; RF cây; phụ ở k | {0,01; 0,1; 1; 10}; 3 fold; 500; {50, 200} |
| mRMR | tiêu chí; rời rạc hóa | MID; 3 mức μ ± σ/2 |
| ReliefF | lân cận | 10 |
| Wrapper | quần thể; thế hệ; fitness; GA; BPSO | 30; 50; AUROC CV 3 fold LR-L2 C = 1; tournament 3, elite 2, đột biến 1/k; w 0,9→0,4, c1 = c2 = 2, v_max 6 |
| Ổn định | M; tỉ lệ; ρ; bản sao; seed E5b; k chồng lấn | 30; 0,5/0,7/0,9; 0,5/0,8/0,95; 50; 0–4; 50 |
| Thống kê | α; ROPE; tỉ lệ n_test/n_train | 0,05; ±0,01 AUROC; 1/4 |
| Chi phí | lần đo; k; lưới n; lưới p; ngân sách wrapper | 3; 50; 100/200/300/toàn bộ; 1.000/2.000/5.000/10.000/toàn bộ; 10×20, 30×50 |
| Sanity | bộ; k; ngưỡng AUROC | tcga_normal_vs_tumor_sanity; 10; 0,99 |

## Phụ lục B. Tài liệu gốc để đọc thêm

Đã có trong danh mục tham khảo của bản v1 (người viết giữ): Guyon & Elisseeff 2003 (tổng quan FS); Saeys 2007;
Peng 2005 (mRMR); Urbanowicz 2018 (ReliefF, skrebate); Tibshirani 1996 và Zou & Hastie 2005 (L1, Elastic Net);
Breiman 2001 (RF); Guyon 2002 (SVM-RFE); Siedlecki & Sklansky 1989 (GA); Kennedy & Eberhart 1995 (PSO);
Nogueira 2018; Kuncheva 2007.

Người viết đang bổ sung (`input/references/new_references_required.md`): Nadeau & Bengio 2003; Holm 1979;
Benavoli et al. 2017 (Bayesian correlated t-test); Ambroise & McLachlan 2002 (selection bias); Kennedy & Eberhart 1997
(BPSO); Ojala & Garriga 2010 và Phipson & Smyth 2010 (kiểm định hoán vị). Đội code không cần trích dẫn; chỉ cần
biết nguồn gốc của từng công thức trong mục 8 khi được hỏi.
