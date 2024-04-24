# Zero-Shot Stance Detection Enhanced with Background Knowledge

This repository contains the implementation of a zero-shot stance detection model enhanced with background knowledge, as described in the paper "Zero-Shot Stance Detection Enhanced with Augmented Background Knowledge" by Jacky Jiang, Jerry Wei, and Lize Shao from Rice University.

## Introduction

The goal of stance detection is to identify a text’s stance towards a particular topic without having seen examples of the stance during training. Our approach improves zero-shot stance detection by incorporating additional background knowledge, categorized into topic-related and expression-related knowledge. Using retrieval-enhanced generation techniques, we collect topic-related knowledge from Wikipedia and provide expression-related knowledge specific to jargon used in online communication.

## Requirements

- Python 3.x
- PyTorch
- Transformers by Hugging Face
- tqdm
- scikit-learn
- faiss
- tensorboardX

You can install the required packages using `pip`:
```bash
pip install torch transformers tqdm scikit-learn faiss-cpu tensorboardX
```

## Usage
### Dataset Preparation
Ensure your dataset is formatted according to the expectations of the script. Example datasets can be found in the ./datasets folder with the appropriate structure.

### Training and Testing the Model
To run the training and testing of the model, use the following command:

```bash
python main.py --model_name bert-scl --dataset covid --target "stay at home orders" --output_par_dir test_outputs
```

You can customize the training by specifying additional command line arguments. Here are some key arguments you might consider:

- `--model_name`: Model to use (default: bert-scl).
- `--dataset`: Dataset to use (choices: semeval16, covid).
- `--target`: Specific target within the dataset for stance detection.
- `--output_par_dir`: Parent directory to save outputs.

### Custom Configuration
Modify the script's argument parser setup in `main.py` to include any additional or custom configurations specific to your needs.

## Output
The output includes the trained model, logs, and performance metrics which are saved in the specified output directory.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

This README file is formatted in Markdown and ready to be placed directly into your GitHub repository to provide users with all necessary information about how to set up and run your stance detection model.
