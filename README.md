# KNN-Instruct

Repo for [EMNLP 2024] KNN-Instruct: Automatic Instruction Construction with K Nearest Neighbor Deduction.

## Steps to Reproduce KNN-Instruct

Step1. Create a virtual env and activate it.

```shell
conda create -n knn-inst python=3.12 && conda activate knn-inst
```

Step2. Install all requirements.

```shell
pip install -r requirements.
```

Step3. Prepare your OpenAI API.

```
export OPENAI_API_BASE=https://xxx.xxxxxx.xxx/v1/
export OPENAI_API_KEY=sk-ssssssssssssssssssssssss
```

Step4. Prepare the seed dataset `seeds-pool.jsonl` in advance.

```shell
python code/prepare-seed.py
```

Step5. Run the `knn-inst.py` to double the seed dataset.

```shell
python code/knn-inst.py
```

The 3k synthesized samples would be added to the `seeds-pool.jsonl`. Re-run `knn-inst.py` and you would get `KNN-Inst-12k` (`gpt-4o-mini` as teacher).
