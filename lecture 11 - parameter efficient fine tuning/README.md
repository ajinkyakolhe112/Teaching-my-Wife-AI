
# Lecture 11: Simple PEFT Fine-Tuning

This directory contains the materials for Lecture 11, focusing on Parameter-Efficient Fine-Tuning (PEFT).

## Files

*   `lecture_11.md`: A detailed markdown file explaining the concepts of PEFT and LoRA, with a step-by-step guide to fine-tuning.
*   `peft_fine_tuning.py`: The Python script for fine-tuning a model using PEFT and LoRA.
*   `requirements.txt`: A list of the necessary Python libraries to run the script.
*   `pride_and_prejudice_instructions.json`: The instruction dataset (you will need to create this file based on the instructions in Lecture 10).

## Instructions

1.  **Install Dependencies:**
    Before running the script, make sure you have all the required libraries installed. You can do this by running the following command in your terminal:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare the Dataset:**
    This lecture assumes you have already created the `pride_and_prejudice_instructions.json` file as described in Lecture 10. Make sure this file is in the same directory as the `peft_fine_tuning.py` script.

3.  **Run the Fine-Tuning Script:**
    To start the fine-tuning process, simply run the `peft_fine_tuning.py` script from your terminal:
    ```bash
    python peft_fine_tuning.py
    ```
    The script will load the dataset, configure the model with LoRA, and begin training. Once training is complete, it will save the fine-tuned model to a new directory named `pride_prejudice_peft_model`.

4.  **Test the Model:**
    The script will automatically test the model with a sample question after training. You can modify the `peft_fine_tuning.py` script to ask different questions and see how the model responds.
