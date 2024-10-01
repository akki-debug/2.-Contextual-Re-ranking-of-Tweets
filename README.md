### **2. Contextual Re-ranking of Tweets**

#### **Problem:**
Generating high-quality tweets that effectively capture business goals, engage the audience, and align with a given prompt is a challenging task. Generative models like GPT-2 often produce content that may be irrelevant, have a negative tone, or fail to meet engagement objectives. Without a robust evaluation framework, choosing the best output becomes tedious, resulting in ineffective social media content.

#### **Solution:**
This project addresses the issue by generating multiple tweet candidates using GPT-2 and implementing a re-ranking mechanism using BERT for sentiment analysis and Sentence-BERT for relevance scoring. Sentiment analysis helps prioritize tweets with a positive tone, while relevance scoring ensures alignment with the initial prompt. The combined ranking, which averages sentiment and relevance scores, identifies tweets that are both positive and contextually relevant. This structured approach helps optimize tweet selection, leading to better-quality social media posts that align with business objectives.
