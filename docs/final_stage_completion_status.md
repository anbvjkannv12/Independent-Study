# 最後階段完成狀態紀錄

> 日期：2026-09-30  
> 依據：`docs/final_stage_work_plan.md`

## 已完成

- 工作 1（部分）：已將 `feature-ab-test-v2/logs/ab_v2` 收回 main 的 `logs/ab_v2/`；已產生 `backup_manifest.json`（不進 git）；已抽查 3 個檔案 SHA-256，均一致。
- 工作 2：已同步總報告第六、八節，並把 `docs/next_work_plan.md` 標為歷史文件。
- 工作 3：已新增 `scripts/make_report_figures.py`，產生 F1～F10 與 `docs/figures/figure_data.json`。
- 工作 4：已撰寫繳交報告初稿 `docs/final_project_report.md`。
- 工作 5：已撰寫口試簡報大綱與 Q&A `docs/final_presentation_outline.md`。
- 工作 6：已產生 `requirements-lock.txt`；完整測試 107 項通過；`verify_multi_horizon_outputs.py` V1～V8 PASS；三段式已由逐列預測重算。
- 工作 9：已另開波動大小預測延伸研究預先登記、實作、測試並執行；四幣種皆通過預定判準，結果見 `docs/volatility_target_extension_results.md`。

## 依使用者指示不做

- 工作 7：即時預測展示不做。

## 仍需人工確認或外部執行

- 第零節外部條件（繳交期限、格式、頁數、口試、是否需系統展示、是否需繳交程式碼或資料）仍需向老師或系上確認。
- 遠端 private repo 尚未設定；需要使用者決定 GitHub/GitLab/其他位置後再 push。
- `logs/`、`models/`、`data/features/` 已列入 manifest，但實際複製到兩個雲端/外接備份位置需由使用者在本機外部儲存完成。
- 工作 8 須等 2027-01-25 之後才能開始。

## 檢查紀錄

- 完整測試：`logs/final_full_test_stdout.log`、`logs/final_full_test_stderr.log`，107 tests OK。
- 多視窗驗證：`logs/final_verify_multi_horizon_stdout.log`，V1～V8 PASS。
- 三段式重算：`logs/final_three_class_recompute_stdout.log`、`logs/final_three_class_recompute_stderr.log`。
- 波動延伸研究重跑：`logs/volatility_target_rerun_stdout.log`、`logs/volatility_target_rerun_stderr.log`。
