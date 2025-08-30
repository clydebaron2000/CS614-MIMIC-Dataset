# QLoRA Validation on MIMIC-IV

**Authors**: David Everly, Francis Poeske, Clyde Rapinan, Gellknight Arulmani  
**Language**: Python  
**Version**: 1.0  

---

## Description
This project implements fine-tuning of the **Llama 3–8B** model on the **MIMIC-IV v3.1** dataset using **QLoRA (Quantized Low-Rank Adaptation)**.  
It provides training and validation scripts to compare model outputs **with and without the QLoRA adapter** against a set of curated prompts and ideal response patterns validated by a medical professional. Outputs are logged, stored, and exported for further quantitative and qualitative evaluation.

---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Examples](#examples)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Installation
### Dependencies
- Python ≥ 3.10  
- Access to **Meta Llama 3** models (via HuggingFace). Access keys can be requested via the HuggingFace model card.
- MIMIC-IV Dataset (via https://physionet.org/content/mimiciv/3.1/). Credentialed Access Required

Install Python requirements:
```bash
pip install -r requirements.txt
```

---

## Usage
Program is intended to be run using a Unix-like terminal such as Linux, macOS Terminal (untested), or MINGW64 (Git Bash) on Windows.

### Steps

1. Consolidate the MIMIC-IV relational tables into a subject-keyed JSONL file
   - `python convert_to_jsonl.py`
2. Summary Stats
   - python summarize_jsonl.py
3. Gradient Boost to determine least important features
   - `python feature_select_xgb_columns.py`
4. Prompt Engineering
   - `python make_symptom_prompts_ctx_dxpx.py`
5. Train Model
   - `tmux new -s qlora_train
conda activate qlora
python train_qlora.py | tee /storage/cache/outputs/train.log`
6. Patient-pipleline check
   - `python qa_by_subject.py`
7. Run the validation script by navigating to the `run_files` directory and executing:
   - ```bash
  cd run_files
./run.sh [n]
```
  - Where `n` = number of trials to run.

Model outputs are saved in `/tmp` as CSV files and previewed on the command line. After the batch finishes, copy CSV results to your local machine, for example:
```bash
scp user@remote:/tmp/output.csv ./results/
```

---

## Features
- **End-to-end pipeline:** raw CSV -> JSONL -> prompts -> fine-tuned model
- **EHR system pipeline test**
- **Automated validation pipeline**: Sends predefined prompts to Llama 3–8B before and after QLoRA adaptation.
- **Side-by-side evaluation**: Saves model responses alongside ideal response patterns for downstream analysis.
- **Lightweight logging**: Input/output previewed in terminal and exported to CSV in `/tmp`.
- **Portable results**: `scp` command suggestion shown at the end for secure export.

---

## Examples
### Example Command
```bash
./run.sh 50
```
Runs 50 validation trials and exports results to `/tmp`.

### Example Output (preview)
```text
Prompt: "Summarize the key risk factors for sepsis."
Response (Base):   "Sepsis occurs due to..."
Response (QLoRA):  "Sepsis is caused by infection..."
Ideal Response:    "Key risk factors for sepsis include..."
```

---

## References
[1] 2023. MIMIC-IV, version 3.1. https://physionet.org/content/mimiciv/3.1/.  
[2] 2024. Meta Llama 3 announcement. https://ai.meta.com/blog/meta-llama-3/.  
[3] Brian G. Arndt, John W. Beasley, Michelle D. Watkinson, Jonathan L. Temte,
Wen-Jan Tuan, Christine A. Sinsky, and Valerie J. Gilchrist. 2017. Tethered to the
EHR: Primary Care Physician Workload Assessment Using EHR Event Log Data
and Time-Motion Observations. Annals of Family Medicine 15, 5 (2017), 419–426.
https://doi.org/10.1370/afm.2121  
[4] clydebaron2000. 2025. CS614-MIMIC-Dataset. GitHub. https://github.com/
clydebaron2000/CS614-MIMIC-Dataset  
[5] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023.
QLoRA: Efficient Finetuning of Quantized LLMs. arXiv preprint (2023).
arXiv:2305.14314 https://arxiv.org/abs/2305.14314  
[6] Daniel Fraile Navarro, Enrico Coiera, Tom W. Hambly, Zachary Triplett, Nabeel
Asif, Agus Susanto, Anik Chowdhury, Antonio Azcoaga Lorenzo, Mark Dras, and
Shlomo Berkovsky. 2025. Expert evaluation of large language models for clinical
dialogue summarization. Scientific Reports 15 (2025), 1195. https://doi.org/10.
1038/s41598-024-84850-x  
[7] Rebekah L. Gardner, Emily Cooper, Jacqueline Haskell, Daniel A. Harris, Sara
Poplau, Philip J. Kroth, and Mark Linzer. 2019. Physician stress and burnout:
the impact of health information technology. Journal of the American Medical
Informatics Association 26, 2 (2019), 106–114. https://doi.org/10.1093/jamia/
ocy145  
[8] Tian Han, Lisa C. Adams, John-Michael Papaioannou, Philipp Grundmann, T.
Oberhauser, A. Figueroa, Alexander Löser, Daniel Truhn, and Keno K. Bressem.
2025. MedAlpaca — an open-source collection of medical conversational AI
models and training data. arXiv preprint (2025). https://doi.org/10.48550/arXiv.
2304.08247 arXiv:2304.08247  
[9] Alistair Johnson et al. 2023. MIMIC-IV (version 3.1). Scientific Data (2023).
https://pmc.ncbi.nlm.nih.gov/articles/PMC9810617/  
[10] Kenton Li. 2023. ChatDoctor. https://github.com/Kent0n-Li/ChatDoctor. GitHub
repository, accessed August 3, 2025.  
[11] Siru Liu. 2025. Detecting emergencies in patient portal messages using large
language models and a knowledge graph. Journal of the American Medical
Informatics Association 32, 6 (2025), 1032–1039. https://doi.org/10.1093/jamia/
ocaf059  
[12] Qingyu Lu, Deqing Dou, and Tuyen H. Nguyen. 2022. ClinicalT5: A generative
language model for clinical text. In Findings of the Association for Computational
Linguistics: EMNLP 2022. 5436–5443. https://aclanthology.org/2022.findingsemnlp.398  
[13] Conor Senecal, Madeline Mahowald, Lilach Lerman, Francisco Lopes-Jimenez,
and Amir Lerman. 2021. Increasing utility of Google Trends in monitoring
cardiovascular disease. Digital Health 7 (2021), 20552076211033420.
https://doi.org/10.1177/20552076211033420  
[14] Adrian Toma, Paul R. Lawler, Jimmy Ba, Ramesh G. Krishnan, Bruce B. Rubin,
and Bo Wang. 2023. Clinical Camel: An open expert-level medical language
model with dialogue-based knowledge encoding. arXiv preprint (2023).
https://doi.org/10.48550/arXiv.2305.12031 arXiv:2305.12031  
[15] Daan Van Veen, Charlotte Van Uden, Louis Blankemeier, Jean-Benoit Delbrouck,
A. Aali, Christoph Bluethgens, A. Pareek, M. Polacin, E. P. Reis, Anna Seehofnerová, N. Rohatgi, P. Hosamani, W. Collins, N. Ahuja, Curtis P. Langlotz, J.
Hom, S. Gatidis, John Pauly, and Akshay S. Chaudhari. 2024. Adapted large language models can outperform medical experts in clinical text summarization. Nature Medicine 30, 4 (2024), 1134–1142. https://doi.org/10.1038/s41591-024-02855-5  
[16] Ingo C. Wiest, Daniel Ferber, Jing Zhu, Martin van Treeck, Sebastian K. Meyer,
R. Juglan, Z. I. Carrero, Daniel Paech, Jens Kleesiek, Maximilian P. Ebert, Daniel
Truhn, and Jakob N. Kather. 2024. Privacy-preserving large language models for
structured medical information retrieval. NPJ Digital Medicine 7, 1 (2024), 257.
https://doi.org/10.1038/s41746-024-01233-2  
[17] Chao Wu, Weizhe Lin, Xi Zhang, Yuxia Zhang, Wenzhu Xie, and Yanshan Wang.
2024. PMC-LLaMA: Toward building open-source language models for medicine.
Journal of the American Medical Informatics Association 31, 9 (2024), 1833–1843.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11639126/  
[18] Xiaoqian Yang, Weizhe Lin, Yuxia Zhang, Xi Zhang, Yanshan Wang, and Wenzhu
Xie. 2022. GatorTron: A large language model for clinical natural language
processing. medRxiv (2022). https://doi.org/10.1101/2022.02.27.22271257  
[19] Xin Zhao, Siru Liu, Shi-Yuan Yang, and Chunyan Miao. 2025. MedRAG: Enhancing
retrieval-augmented generation with knowledge graph-elicited reasoning for
healthcare copilot. arXiv preprint (2025). https://doi.org/10.48550/arXiv.2502.
04413 arXiv:2502.04413

---

## Contributing
Program made possible by hardware provided by **Drexel University**. Contributions and improvements are welcome—please open an issue or submit a pull request.

---

## License
This project follows the **MIMIC-IV Data Use Agreement (DUA)**. See PhysioNet’s policies for terms of use.

---

## Disclaimer
This project was developed independently on personal time and is **not affiliated with or endorsed by any employer or healthcare organization**.  
All data used is publicly available and non-identifiable.  
All opinions and methods reflect personal research and experimentation.



