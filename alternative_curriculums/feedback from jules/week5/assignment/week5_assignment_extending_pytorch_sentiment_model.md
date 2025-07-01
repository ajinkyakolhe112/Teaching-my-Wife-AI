# Assignment 5: Extending the PyTorch Sentiment Model or Applying Pipelines

**Objective:** To either modify and retrain the custom PyTorch sentiment analysis model or to apply Hugging Face pipelines to a new dataset and analyze the results.

**Prerequisites:**
*   Understanding of concepts covered in Week 5 Lectures.
*   Python 3.x installed.
*   Libraries: `transformers`, `torch`, `datasets`. `wandb` is required if choosing Option 1.
    ```bash
    pip install transformers torch datasets wandb
    # Add scikit-learn if not already installed for other general ML tasks
    # pip install scikit-learn 
    ```
*   For Option 1:
    *   Access to `simple_sentiment_model_pytorch.py` and `train_sentiment_model_pytorch.py` from Week 5 lectures.
    *   Ability to run Python scripts and train a PyTorch model (potentially on a GPU for faster training, though CPU is possible).
*   For Option 2:
    *   Internet access to browse Hugging Face Hub and potentially download small datasets.

---

## Instructions:

**Choose ONE of the following two options for your assignment.**

---

### Option 1: Modify and Retrain the PyTorch Model

This option focuses on experimenting with the custom PyTorch model developed in the lectures.

1.  **Choose a Modification:**
    Select **one** modification to the `SimpleSentimentModel` (defined in `simple_sentiment_model_pytorch.py`) or the training process (in `train_sentiment_model_pytorch.py`). Here are some ideas:
    *   **Model Architecture Changes:**
        *   Add another linear layer (e.g., `self.fc_intermediate = nn.Linear(32, 16)` and then `self.fc2 = nn.Linear(16, 1)`). Don't forget to update the `forward` method!
        *   Change the `embedding_dim` (e.g., from 64 to 128, or to 32).
        *   Change the `hidden_dim` of `fc1` (e.g., from 32 to 64, or to 16).
        *   (Advanced) Add a `nn.Dropout` layer after the ReLU activation or before the final `fc2` layer to potentially reduce overfitting. Remember to set `model.train()` and `model.eval()` appropriately if you use dropout.
    *   **Training Process Changes:**
        *   Experiment with a different optimizer (e.g., `torch.optim.SGD` with momentum instead of `torch.optim.Adam`).
        *   Adjust the `learning_rate` (e.g., make it 10x smaller or larger).
        *   Change the `num_epochs` for training (e.g., increase to 10 or decrease to 3).
        *   Modify the `batch_size` in `load_and_prepare_data` (e.g., 16 or 64).

2.  **Implement the Modification:**
    *   Make the chosen modification in your local copies of `simple_sentiment_model_pytorch.py` and/or `train_sentiment_model_pytorch.py`.
    *   Clearly comment your changes.

3.  **Retrain the Model:**
    *   Run your modified `train_sentiment_model_pytorch.py` script.
    *   Ensure `wandb` (Weights & Biases) is active. If you haven't used it before, you might need to sign up for a free account at [wandb.ai](https://wandb.ai) and run `wandb login` in your terminal once. The script will guide you.
    *   Let the model train. Note down the `wandb` run name or URL.

4.  **Report Results:**
    *   **A. Describe Your Modification:**
        *   What specific change did you make?
        *   Why did you choose this modification? What effect did you hypothesize it would have?
    *   **B. Present Training Logs:**
        *   Provide screenshots of your `wandb` training logs, specifically showing the loss and accuracy curves over epochs. (If you cannot use wandb, record the epoch-wise loss and accuracy printed to the console).
        *   Note the final average loss and accuracy achieved by your modified model.
    *   **C. Compare and Analyze:**
        *   Compare the performance of your modified model to the original model (the lecture version typically achieves around 80-85% accuracy on IMDB after 5 epochs, depending on setup).
        *   Did your change improve, degrade, or have little effect on performance (accuracy and loss)?
        *   Discuss your findings. Did the results match your hypothesis? Why or why not?
    *   **D. Challenges:**
        *   Briefly discuss any challenges you encountered during this process (e.g., debugging, understanding concepts, resource limitations).

---

### Option 2: Apply Hugging Face Pipeline to a New Dataset

This option focuses on applying a pre-trained Hugging Face sentiment analysis pipeline to a dataset you find or create.

1.  **Find or Create a Small Text Dataset:**
    *   Find a small dataset of texts (e.g., 20-50 texts) suitable for sentiment analysis. This could be:
        *   Product reviews for a specific category from an e-commerce site.
        *   A collection of tweets on a particular current event or topic (ensure they are in a language your chosen model supports).
        *   Comments from a specific blog post or YouTube video.
        *   A few paragraphs from different news articles or editorials.
    *   **Important:**
        *   Provide a link to your data source if it's public.
        *   If you manually collected the data, briefly describe your collection method and include the texts directly in your report.
        *   Ensure the data is in a format that you can easily read into a Python list of strings.

2.  **Choose a Hugging Face Sentiment Analysis Pipeline:**
    *   Select a specific pre-trained sentiment analysis model from the [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads&filter=sentiment-analysis) to use via the `pipeline` function.
    *   **Examples:**
        *   `distilbert-base-uncased-finetuned-sst-2-english` (general, good baseline)
        *   `cardiffnlp/twitter-roberta-base-sentiment-latest` (fine-tuned on tweets)
        *   `nlptown/bert-base-multilingual-uncased-sentiment` (outputs 1-5 stars, multilingual)
        *   `finiteautomata/bertweet-base-sentiment-analysis` (another Twitter-focused model)
    *   **Justify your choice briefly (1-2 sentences):** Why is this model suitable for your chosen dataset or task?

3.  **Perform Sentiment Analysis:**
    *   Write a Python script or use a Jupyter Notebook to:
        *   Load your dataset into a list of strings.
        *   Initialize the chosen Hugging Face `pipeline` for "sentiment-analysis" with your selected model.
        *   Run the pipeline on your dataset to get sentiment predictions for each text.

4.  **Analyze and Report Results:**
    *   **A. Sentiment Distribution:**
        *   Present a summary of the sentiment distribution found in your dataset (e.g., X% POSITIVE, Y% NEGATIVE, Z% NEUTRAL, or distribution of star ratings if using a model like `nlptown/bert-base-multilingual-uncased-sentiment`).
    *   **B. Example Predictions:**
        *   Show 5-10 examples of texts from your dataset along with their predicted sentiments and scores.
    *   **C. Discussion:**
        *   Discuss any interesting findings. Were the model's predictions generally accurate in your opinion?
        *   Were there any specific types of statements or nuances in your dataset that the model struggled with or classified surprisingly? Provide examples if so.
        *   What are the potential limitations of using this pre-trained model for your specific dataset?
    *   **D. Real-World Application:**
        *   Briefly describe how the sentiment analysis results for your chosen dataset might be useful in a real-world scenario. (e.g., "Understanding customer feedback for product X," "Gauging public opinion on topic Y," "Moderating comments on platform Z").

---

## Submission Guidelines:

*   Clearly indicate which **Option (1 or 2)** you chose.
*   Compile your responses, code (for Option 2, or modified code snippets for Option 1), results, and analysis into a **single document**.
*   You can use Markdown (preferred, save as a `.md` file) or create a PDF.
*   If you chose Option 1, include relevant screenshots from `wandb` or your console output for training logs.
*   If you chose Option 2, ensure your dataset (or a link to it) and the texts used are clearly presented.
*   Name your file clearly, e.g., `week5_assignment_yourname.md`.

Good luck! This assignment is designed to give you practical experience in either refining a custom model or applying existing powerful models to new data.
