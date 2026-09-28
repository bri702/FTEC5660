# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution: 
> to students: please fill your solution description here.

Visualization：

  A[Receipt images] --> B[image_data_url: encode images]
  B --> C

  subgraph Batch["chain.batch(inputs): process each receipt independently"]
        subgraph Extract["parse_chain: extract structured data"]
            C[receipt_prompt] --> D[DeepSeek model]
            D --> E[JsonOutputParser]
        end
        E --> F["RunnableLambda(compute_receipt)"]
        F --> G[Calculate per-receipt totals using Decimal]
    end

  G --> H[Aggregate results across receipts]
  H --> I[Return dictionary for the two queries]
  I --> J[Provided runner writes results.csv]


Description：
The solution encodes receipt images with image_data_url() and processes them independently using chain.batch(). The extraction chain, receipt_prompt | llm | JsonOutputParser(), combines instructions and an image, calls deepseek-v4-flash-vision-exp, and parses its response into structured data. RunnableLambda(compute_receipt) then performs monetary calculations using Python Decimal. Finally, the per-receipt results are aggregated and returned as a dictionary for the two required queries, which the provided runner saves to results.csv.
