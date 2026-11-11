# Data

Put your downloaded files here, exactly like this (names matter):

```
data/raw/
├── trials/
│   ├── ClinicalTrials.2021-04-27.part1.zip   (any file name ending in .zip works;
│   ├── ClinicalTrials.2021-04-27.part2.zip    no need to unzip)
│   ├── ClinicalTrials.2021-04-27.part3.zip
│   ├── ClinicalTrials.2021-04-27.part4.zip
│   └── ClinicalTrials.2021-04-27.part5.zip
├── topics2021.xml
├── qrels2021.txt
├── topics2022.xml
└── qrels2022.txt
```

The same 5 trial zips are used for both 2021 and 2022; only topics and qrels differ by year.

Then, from the project folder, convert them once (free, runs on your computer):

```
python -m trialmatch_agent prepare --trials data/raw/trials --topics data/raw/topics2021.xml data/raw/topics2022.xml --qrels data/raw/qrels2021.txt data/raw/qrels2022.txt --judged-only
```

This creates `data/processed/trials.jsonl`, `topics2021.jsonl`, `topics2022.jsonl`,
`qrels2021.txt`, and `qrels2022.txt`, which `configs/trec2021.json` and `configs/trec2022.json` use.
Both folders are ignored by git (the data is large and publicly downloadable).
