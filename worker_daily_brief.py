# worker_daily_brief.py (Debug & Robustness Enhanced Version)

import os
import sys # Import sys to use flush
import json
import time
from datetime import datetime
from typing import List, Dict, Any

from groq import Groq
import logging
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore

# --- Initialization with Immediate Feedback ---
print("--- Script Starting ---", flush=True) # Checkpoint 1: Script has started

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Firebase Initialization with Detailed Logging ---
try:
    print("Initializing Firebase...", flush=True) # Checkpoint 2: Attempting Firebase init
    
    service_account_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    groq_api_key_str = os.getenv('GROQ_API_KEY')

    # Detailed check for environment variables
    if not service_account_json_str:
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not found or is empty.", flush=True)
        sys.exit(1) # Exit with an error code
    
    if not groq_api_key_str:
        print("ERROR: GROQ_API_KEY environment variable not found or is empty.", flush=True)
        sys.exit(1)

    print("Successfully retrieved environment variables.", flush=True)

    service_account_info = json.loads(service_account_json_str)
    cred = credentials.Certificate(service_account_info)
    
    print("Firebase credential object created.", flush=True) # Checkpoint 3: Cred object created

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
        print("Firebase app initialized for the first time.", flush=True)
    else:
        print("Firebase app was already initialized.", flush=True)

    db_firestore = firestore.client()
    analyzed_news_collection = db_firestore.collection('analyzed_news')
    daily_briefs_collection = db_firestore.collection('daily_briefs')
    logger.info("BRIEFING WORKER: Firebase setup complete.")
    print("BRIEFING WORKER: Firebase setup complete.", flush=True) # Checkpoint 4: Firebase is ready

except json.JSONDecodeError as e:
    logger.error(f"FATAL: Failed to parse GOOGLE_APPLICATION_CREDENTIALS_JSON. It might be malformed. Error: {e}", exc_info=True)
    print(f"FATAL: Failed to parse GOOGLE_APPLICATION_CREDENTIALS_JSON. Error: {e}", flush=True)
    sys.exit(1)
except Exception as e:
    logger.error(f"FATAL: Failed to initialize Firebase: {e}", exc_info=True)
    print(f"FATAL: Failed to initialize Firebase: {e}", flush=True)
    sys.exit(1)

# ... (The rest of your code: BriefingAIProcessor, main function) ...
# ... (Please make sure the rest of the code is unchanged from the previous version) ...

class BriefingAIProcessor:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        # No need to check for key again, we did it at the start
        self.client = Groq(api_key=self.api_key)
        self.model = "llama3-70b-8192"

    def generate_market_briefing(self, news_summaries: List[str]) -> Dict[str, Any]:
        if not news_summaries:
            logger.warning("BRIEFING WORKER: No news summaries provided to generate briefing.")
            return {}
        context_str = "\n\n---\n\n".join(news_summaries)
        prompt = f"""
        You are an expert financial market analyst AI for the "FinanceFlow" app. Your task is to analyze the following collection of recent news summaries and generate a structured "Daily Market Briefing".

        CONTEXT - RECENT NEWS SUMMARIES:
        ---
        {context_str}
        ---

        Based *only* on the context provided, generate the briefing with the following strict JSON structure. Do not include any text, reasoning, or markdown formatting outside of the JSON object itself. The language must be professional but accessible.

        {{
          "market_headline": "A concise, impactful headline summarizing the overall market sentiment for the day, in English. (e.g., 'Tech Stocks Surge on Positive Inflation Data, Eyes on Fed Meeting')",
          "market_overview": "A single, easy-to-understand paragraph in English summarizing the key events and themes from all news provided. Explain the general market mood and what's driving it.",
          "key_drivers_and_outlook": [
            "A bullet point interpreting the most significant news and its potential impact on the market or specific sectors.",
            "Another bullet point highlighting a different key factor or trend visible from the news.",
            "A forward-looking statement on what investors should watch out for next (e.g., upcoming reports, economic data)."
          ],
          "movers_and_shakers": ["A list of up to 5 most relevant stock ticker symbols (e.g., 'AAPL', 'PTT') that are central to today's news context."]
        }}
        """
        
        try:
            logger.info("BRIEFING WORKER: Sending request to Groq for market briefing...")
            start_time = time.time()
            
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            end_time = time.time()
            logger.info(f"BRIEFING WORKER: Received response from Groq in {end_time - start_time:.2f} seconds.")
            
            response_content = chat_completion.choices[0].message.content
            return json.loads(response_content)

        except Exception as e:
            logger.error(f"BRIEFING WORKER: Groq briefing generation failed: {e}", exc_info=True)
            return {}

def main():
    logger.info("--- Starting FinanceFlow Daily Briefing Worker ---")
    today_id = datetime.utcnow().strftime('%Y-%m-%d')
    logger.info(f"BRIEFING WORKER: Generating brief for date (UTC): {today_id}")

    brief_doc_ref = daily_briefs_collection.document(today_id)
    if brief_doc_ref.get().exists:
        logger.info(f"BRIEFING WORKER: A daily brief for {today_id} already exists. Exiting.")
        return

    try:
        logger.info("BRIEFING WORKER: Fetching latest analyzed news from Firestore...")
        query = analyzed_news_collection.order_by("published", direction=firestore.Query.DESCENDING).limit(15)
        docs = query.stream()
        
        news_summaries_for_context = [
            doc.to_dict().get('analysis', {}).get('summary_en', '')
            for doc in docs
            if doc.to_dict().get('analysis', {}).get('summary_en')
        ]
        
        if not news_summaries_for_context:
            logger.warning("BRIEFING WORKER: No news with summaries found in Firestore to generate a briefing. Exiting.")
            return
            
        logger.info(f"BRIEFING WORKER: Found {len(news_summaries_for_context)} news summaries to use as context.")

    except Exception as e:
        logger.error(f"BRIEFING WORKER: Failed to fetch news from Firestore: {e}", exc_info=True)
        return

    ai_processor = BriefingAIProcessor()
    briefing_data = ai_processor.generate_market_briefing(news_summaries_for_context)
    
    if briefing_data and 'market_headline' in briefing_data:
        try:
            briefing_data['generated_at_utc'] = firestore.SERVER_TIMESTAMP
            briefing_data['source_news_count'] = len(news_summaries_for_context)
            
            brief_doc_ref.set(briefing_data)
            logger.info(f"BRIEFING WORKER: Successfully generated and saved daily brief for {today_id}.")
        except Exception as e:
            logger.error(f"BRIEFING WORKER: Failed to save the briefing to Firestore: {e}", exc_info=True)
    else:
        logger.error("BRIEFING WORKER: AI did not return valid briefing data. Nothing was saved.")

    logger.info("--- FinanceFlow Daily Briefing Worker Finished ---")


if __name__ == "__main__":
    main()