import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# Load GPT-2 model and tokenizer for tweet generation
gpt2_model_name = "gpt2"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Load BERT model and tokenizer for re-ranking (sentiment analysis)
bert_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)
bert_model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Load SentenceTransformer for relevance scoring
st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to generate multiple tweet candidates with diverse outputs
def generate_tweet_candidates(prompt, num_candidates=5, max_length=50, temperature=1.2, top_p=0.9):
    gpt2_model.eval()
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')

    candidates = set()  # Using a set to avoid duplicates
    attempts = num_candidates * 3  # Make extra attempts to avoid repetition

    for _ in range(attempts):
        output = gpt2_model.generate(input_ids,
                                     max_length=max_length,
                                     num_return_sequences=1,
                                     no_repeat_ngram_size=2,
                                     temperature=temperature,  # Introduces randomness
                                     top_p=top_p,  # Nucleus sampling
                                     do_sample=True)  # Enables random sampling
        tweet = gpt2_tokenizer.decode(output[0], skip_special_tokens=True).strip()
        candidates.add(tweet)
        if len(candidates) >= num_candidates:
            break

    return list(candidates)

# Function to rank tweets based on sentiment using BERT
def rank_tweets_by_sentiment(tweets):
    scores = []
    for tweet in tweets:
        inputs = bert_tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=512).to('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = bert_model(**inputs)
        logits = outputs.logits
        sentiment_score = torch.softmax(logits, dim=1).cpu().detach().numpy()[0]
        positive_score = sentiment_score[-1]
        scores.append(positive_score)
    
    ranked_tweets = sorted(zip(tweets, scores), key=lambda x: x[1], reverse=True)
    return ranked_tweets

# Function to rank tweets based on relevance using Sentence-BERT
def rank_tweets_by_relevance(prompt, tweets):
    scores = []
    prompt_embedding = st_model.encode(prompt, convert_to_tensor=True)
    
    for tweet in tweets:
        tweet_embedding = st_model.encode(tweet, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(prompt_embedding, tweet_embedding).item()
        scores.append(similarity_score)
    
    ranked_tweets = sorted(zip(tweets, scores), key=lambda x: x[1], reverse=True)
    return ranked_tweets

# Streamlit App
st.title("Contextual Re-ranking of Tweets")

st.write("Generate and rank tweets based on sentiment and relevance to the given prompt.")

# Input prompt
prompt = st.text_input("Enter a prompt for tweet generation:", "Advice for entrepreneurs")

# Number of tweet candidates
num_candidates = st.slider("Number of tweet candidates to generate:", 3, 10, 5)

if st.button("Generate Tweets"):
    with st.spinner("Generating tweets..."):
        tweet_candidates = generate_tweet_candidates(prompt, num_candidates=num_candidates)
        st.subheader("Generated Tweet Candidates:")
        for i, tweet in enumerate(tweet_candidates, 1):
            st.write(f"{i}: {tweet}")

    if tweet_candidates:
        with st.spinner("Ranking by sentiment..."):
            ranked_by_sentiment = rank_tweets_by_sentiment(tweet_candidates)
            st.subheader("Tweets Ranked by Sentiment:")
            for tweet, score in ranked_by_sentiment:
                st.write(f"Tweet: {tweet}, Sentiment Score: {score}")

        with st.spinner("Ranking by relevance..."):
            ranked_by_relevance = rank_tweets_by_relevance(prompt, tweet_candidates)
            st.subheader("Tweets Ranked by Relevance:")
            for tweet, score in ranked_by_relevance:
                st.write(f"Tweet: {tweet}, Relevance Score: {score}")

        with st.spinner("Calculating combined ranking..."):
            combined_scores = [(tweet, (sentiment + relevance) / 2) for (tweet, sentiment), (_, relevance) in zip(ranked_by_sentiment, ranked_by_relevance)]
            ranked_by_combined = sorted(combined_scores, key=lambda x: x[1], reverse=True)
            
            st.subheader("Final Ranked Tweets by Combined Sentiment & Relevance:")
            for tweet, score in ranked_by_combined:
                st.write(f"Tweet: {tweet}, Combined Score: {score}")
