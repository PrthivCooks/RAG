import pytest
from core.evaluator import ResearchEvaluator
from finetune.prepare_finetuning import FinancialFineTuningPreparer

def test_mrr_and_ndcg():
    evaluator = ResearchEvaluator()
    retrieved = ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"]
    relevant = ["chunk-2", "chunk-5"]

    # chunk-2 is at rank 2 -> MRR = 1/2 = 0.5
    mrr = evaluator.compute_mrr(retrieved, relevant)
    assert mrr == 0.5

    # NDCG@5
    ndcg5 = evaluator.compute_ndcg(retrieved, relevant, k=5)
    assert 0.0 < ndcg5 <= 1.0

    # Precision & Recall
    prec, rec = evaluator.compute_precision_recall_at_k(retrieved, relevant, k=5)
    assert prec == 2 / 5
    assert rec == 2 / 2

def test_finetuning_dataset_export(tmp_path):
    preparer = FinancialFineTuningPreparer(output_dir=tmp_path)
    sample_chunks = [
        {
            "text": "Apple designs smartphones and expanded its AI technologies under Apple Intelligence.",
            "metadata": {
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "section": "Item 1",
                "section_title": "Business"
            }
        }
    ]

    out_alpaca = preparer.generate_instruction_dataset(sample_chunks, format_type="alpaca")
    import os, json
    assert os.path.exists(out_alpaca)
    with open(out_alpaca, "r") as f:
        data = json.load(f)
        assert len(data) == 1
        assert "AAPL" in data[0]["instruction"]

    script_path = preparer.generate_unsloth_training_script()
    assert os.path.exists(script_path)
