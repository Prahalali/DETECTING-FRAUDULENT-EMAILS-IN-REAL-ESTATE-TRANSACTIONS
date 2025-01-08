## DETECTING-FRAUDULENT-EMAILS-IN-REAL-ESTATE-TRANSACTIONS
In today’s digital era, the proliferation of cyberattacks, particularly Business Email Compromise (BEC),
poses significant threats to financial transactions, with real estate being a prime target. Attackers manipulate
email communication, employing sophisticated tactics to exploit vulnerabilities and mislead individuals into
fraudulent financial transfers. This project aims to combat such risks through a robust Natural Language
Processing (NLP)-driven system that identifies and flags potentially fraudulent emails in real estate
transactions.
The proposed system utilizes advanced NLP technologies, with Bidirectional Encoder Representations from
Transformers (BERT) at its core. By analyzing email content, timestamps, and structural patterns, the model
discerns markers indicative of fraud, such as urgency in tone, anomalies in timestamps, and requests for
sensitive information. Complementing this is an extensive feature engineering process that incorporates
temporal, structural, and content-based insights, all optimized using Python libraries such as NumPy, Pandas,
and SpaCy.
The project follows a structured pipeline, beginning with data preprocessing techniques like tokenization, stop
word removal, and standardization, ensuring clean and actionable data. Exploratory Data Analysis (EDA)
reveals critical insights into spam versus legitimate email characteristics, employing heatmaps, scatter plots,
and bar graphs to highlight correlations and patterns. Model training incorporates a fine-tuned BERT
classifier, leveraging both pre-trained embeddings and additional custom features for enhanced accuracy.
Evaluation of the system, split into an 80:20 training-to-testing ratio, demonstrated high performance in
distinguishing fraudulent emails from legitimate ones. Batch processing and cross-entropy loss functions
further ensured model robustness and generalization to diverse datasets.
The final deliverable includes a user-friendly interface for real-time fraud detection, emphasizing email security
and mitigating financial risks. This project not only advances NLP applications but also sets a precedent for
deploying machine learning solutions to enhance cybersecurity in critical sectors like real estate transactions.
By integrating contextual understanding with technical precision, this system signifies a
proactive step toward safer digital financial practices.


# 1.Data Collection and Dataset Preparation:
Data cleaning is crucial for preparing the dataset for analysis and model training. Below are the steps taken in
detail:
a. Handling Missing Values:
b. Removing Duplicates:
c. Label Encoding: Labels like ham and spam are non-numeric. Encoded labels into numeric format (ham
0, spam → 1) using map() for easier model interpretation.
d. Renaming Columns:
label → Category
message → Email
Text
e. Unnecessary Columns: Some columns like sender, timestamp, or recipient may not be directly useful for
spam classification. Dropped irrelevant columns after determining they didn't add value to the model.
Retained only necessary columns (Email_Text and Category) for initial exploration.

# 2.Exploratory Data Analysis (EDA):
1. Importing Libraries
2.  Loading the Dataset
3.  Understanding the target Column
4.  Exploring Email Characteristics
   
# 3.Data Processing:
Data preprocessing is a crucial step in any data science or machine learning pipeline. It involves cleaning,
transforming, and preparing raw data for analysis or modeling. In this project, we focus on preprocessing text
data for spam classification. Let me walk you through the details of this process, including the libraries used,
the logic implemented, and how visualizations help us understand the dataset.
Libraries Used: nltk (Natural Language Toolkit)
Used for text preprocessing. Functions like tokenization, stopword removal, and stemming come from this
library.
Example: nltk.word
_
tokenize splits text into individual tokens (words or punctuation).We also used the
Porter Stemmer from nltk to reduce words to their root forms.
string: Provides a list of punctuation characters. Helps us filter out punctuations from text.
matplotlib.pyplot and seaborn: Used for visualizations.
Word clouds help visualize the frequency of words in spam and ham messages.
Bar plots display the most common words in each category.
collections.Counter: Counts the frequency of words in spam and ham messages.
Text Transformation:
Transform
_
text, is responsible for cleaning and standardizing the text.
Convert text to lowercase (text.lower()), ensuring consistency in comparison.
Tokenize the text using nltk.word
_
tokenize, breaking it into words or punctuation marks.
Remove non-alphanumeric characters using isalpha().
Eliminate stopwords (common words like "the"
, "and") using stopwords.words('english').
Stem the words to their root forms with the Porter Stemmer (e.g.,
"running" becomes "run").
Apply Transformation to Data: The transform
_
text function is applied to the text column of the DataFrame
to generate a new column called transformed_text. This column contains the cleaned and processed text.

# 4.Model Building:
Detecting fraudulent emails poses a distinct challenge in text classification due to the need for a deep
understanding of contextual meaning and relationships within the text. While traditional machine learning
algorithms like Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Random Forest, and similar
approaches excel in handling structured data, they often fall short when dealing with the complexities of
natural language. The nuanced semantics and variability inherent in text require sophisticated processing that
these models struggle to achieve. Although these algorithms may demonstrate strong accuracy on training
datasets, their performance typically declines significantly when applied to new, unseen data, highlighting
their limitations in eﬀectively generalizing to unstructured text data.
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model renowned
for its remarkable ability to grasp linguistic context. Trained on extensive text datasets, BERT employs a
bidirectional approach, analyzing both preceding and succeeding words to understand the meaning of text
within its full context. This contextual awareness makes BERT exceptionally well-suited for tasks such as
fraudulent email detection, where subtle diﬀerences in phrasing and tone can carry significant meaning.
Evaluation of Alternative Algorithms
Before selecting BERT, examined several traditional and machine learning algorithms to assess their
applicability:
SVC (Support Vector Classifier with Sigmoid Kernel): While SVC is eﬀective for classification problems, it
lacks the capability to interpret the semantic intricacies of text, making it less suitable for tasks involving
nuanced language.
KNN (K-Nearest Neighbors): Although KNN performs well with numerical data, it is not designed to
leverage language patterns or detect deeper text-based relationships.
Multinomial Naive Bayes (MultinomialNB): This model is often used for basic text classification and can
handle categorical data reasonably well. However, it struggles with capturing complex sentence structures
and inter-word dependencies.
Decision Trees, Random Forest, AdaBoost, XGBoost: These ensemble and tree-based models are highly
eﬀective with structured datasets and benefit from rigorous feature engineering. Nonetheless, they are
not inherently designed to process raw text data. Unlike BERT, these models require extensive
preprocessing to transform text into numerical features, which can result in a loss of context and
meaning.
To address this, BERT (Bidirectional Encoder Representations from Transformers) has been chosen for its
state-of-the-art natural language processing (NLP) capabilities, which enable the model to comprehend
complex linguistic relationships and context. This report details the steps involved in leveraging BERT for
identifying fraudulent emails, emphasizing the use of tokenization, feature engineering, and model
optimization.
1. Tokenization with BertTokenizer: The first crucial step in processing email content is tokenization using
BertTokenizer. Tokenization breaks down raw email text into smaller, manageable units called tokens. Each
token represents a meaningful chunk of text, such as words or subwords, and is assigned a unique identifier.
This transformation ensures that the model can process the text in a manner compatible with its architecture,
preserving the context and semantics of the content. The result is a set of tokens that serves as input to
BERT, enabling it to understand and analyze the text eﬃciently.
2. Attention Masks for Input Management: Given the varying lengths of email content, attention masks are
used to distinguish between meaningful text and padding. Padding tokens are introduced to standardize the
input length for batch processing. Attention masks ensure that the model focuses only on the actual content
of the email, while ignoring padding tokens. This approach enhances computational eﬃciency and prevents
unnecessary calculations, making the training process faster and more eﬀective.
3. Input Features and Labels: To improve the predictive performance of the model, we integrate additional
input features alongside the tokenized text. These features provide a deeper understanding of the email's
content and structure:
Number of Characters: Indicates the overall length of the email, providing insight into its verbosity.
Number of Words: Reflects the richness of the email’s content.
Number of Sentences: Oﬀers a measure of the structural complexity of the email.
These features are concatenated with the tokenized data, providing the model with more context to
distinguish between legitimate and fraudulent emails. Corresponding labels (fraudulent or non-fraudulent)
are assigned to each email, forming the basis for supervised learning.
4. Model Selection: BertForSequenceClassification: For this task, the BertForSequenceClassification
architecture is employed. This pre-trained BERT model has been fine-tuned for binary classification tasks,
making it well-suited for distinguishing between fraudulent and non-fraudulent emails. The use of a pre-
trained model ensures the system leverages BERT’s deep understanding of language semantics, requiring
minimal task-specific adjustments. The model outputs predictions across two classes: fraudulent and non-
fraudulent.
5. Optimizer: AdamW: The AdamW optimizer is chosen for fine-tuning BERT. It is well-suited for BERT-
based architectures due to its ability to eﬃciently handle large models and its implementation of weight
decay, which improves generalization. AdamW dynamically adjusts the learning rate during training, helping
the model converge faster while avoiding overfitting. This optimization strategy ensures that the model
performs optimally across diverse email datasets.
6. Learning Rate Scheduler: A learning rate scheduler is employed to regulate the learning rate throughout the
training process. The scheduler helps prevent the model from converging too quickly, which might cause
instability, or too slowly, which could hinder progress. By adjusting the learning rate gradually, the scheduler
ensures stable and consistent model performance, leading to improved results over time.
7. Feature Engineering: In addition to tokenization and basic features, several advanced feature engineering
techniques are employed to enhance the model’s ability to identify fraudulent emails. These features include:
Content Features: These features focus on the linguistic properties of the email, such as keyword
frequency, sentence structure, and tone. By analyzing these aspects, the model can detect subtle patterns
indicative of fraudulent intent.
Structural Features: These capture the overall structure of the email, such as its length in characters,
words, and sentences. Diﬀerences in structure, such as overly long or excessively short emails, may signal
fraudulent activity.
Temporal Features: Time-based patterns, such as the timestamp of the email and peak sending times
(e.g., spam emails often sent during specific hours), oﬀer valuable clues for detecting fraudulent behavior.
Metadata Features: Additional metadata, including sender information, domain analysis, and "reply-to"
addresses, are analyzed to identify any discrepancies or irregularities that might suggest fraud.
8. Training and Optimization: To ensure eﬃcient training, the following steps are followed:
Training Pipeline: The dataset is split into 80% training and 20% testing sets. Stratified sampling is used
to balance the classes (fraudulent vs. non-fraudulent).
Fine-tuning: BERT is fine-tuned for binary classification using cross-entropy loss, a standard loss function
for classification tasks.
Optimization Techniques: The AdamW optimizer is employed for weight adjustments, and a learning rate
scheduler is used to dynamically adjust the learning rate during training. Data is processed in batches to
handle large datasets eﬃciently.

# 5. Validation and Testing:
Test Dataset: A separate, unseen segment of the dataset will be designated for testing, representing 20% of the
total data. The test cases will cover a variety of email types, including:
Simple spam emails containing clear and easily identifiable keywords.
Sophisticated phishing attempts that employ subtle language and tactics.
Legitimate emails with complex formatting and structure.
Validation Metrics:
Accuracy: The overall percentage of correct predictions made by the model, indicating how well it
classifies emails.
Precision: The ratio of true fraud predictions to the total number of predicted fraud cases, highlighting
the model’s eﬀectiveness in identifying fraud.
Recall: The percentage of actual fraudulent emails correctly identified by the model, reflecting its ability
to detect fraud.
5. Validation and Testing:
F1-Score: A balanced metric that combines both precision and recall, oﬀering a comprehensive view of
the model's performance by addressing false positives and false negatives.
Stress Testing:
Evaluate the model's performance on highly imbalanced datasets, where one category (e.g., ham or spam)
is overwhelmingly prevalent.
Introduce adversarial examples to test the robustness and adaptability of the model against unexpected
or manipulated inputs.
Real-World Simulation:
Simulate a real-world email environment by deploying the model on a mock email server that processes
both legitimate and spam emails.
Monitor the model’s performance, focusing on its ability to maintain accuracy and eﬃciency in real-time,
adapting to shifting email traﬃc patterns.
Error Analysis:
Conduct a thorough review of misclassified emails to uncover patterns, misunderstandings, or
weaknesses in the model’s decision-making process.
Use the findings from error analysis to refine feature selection, improve the model's architecture, and
enhance performance in future iterations.
Training and Evaluation:
After the model setup, the next step is the training process.
Training Loop:
The model is trained over several epochs, where each epoch involves passing the training data through the
model. Based on the accuracy of its predictions, the model’s parameters (weights) are adjusted using
backpropagation to minimize error (loss).
Evaluation:
Once the training is complete, the model is evaluated on a separate validation dataset. This step checks how
well the model has learned by comparing its predictions to the true labels.
Metrics:
Accuracy: Measures the proportion of correct predictions made by the model.
Precision: Indicates how many of the emails predicted as fraudulent are actually fraudulent.
Recall: Reflects the number of actual fraudulent emails that the model correctly identified.
F1-Score: A balanced metric combining precision and recall, oﬀering an overall assessment of the
model’s performance.
Confusion Matrix: A table displaying the number of correct and incorrect predictions, broken down by
class (fraudulent or non-fraudulent), providing insights into how well the model distinguishes between
diﬀerent categories.

# 6. Deployment:
For real-world applicability, a user-friendly interface is developed to allow real-time fraud detection. The
model is integrated with a backend system that processes incoming emails and returns predictions. The
deployment system ensures seamless interaction with end-users, providing quick and accurate fraud detection
in email communications.
Conclusion: By leveraging BERT’s advanced NLP capabilities, coupled with thoughtful feature engineering
and optimization strategies, this approach oﬀers a robust solution for detecting fraudulent emails. The
integration of tokenization, attention masks, and additional features ensures that the model can process and
analyze email content eﬀectively. This system, once deployed, has the potential to significantly reduce the
impact of fraudulent emails, enhancing both security and user trust in email communication.
