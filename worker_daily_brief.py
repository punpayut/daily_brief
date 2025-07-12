# worker_daily_brief.py (Dual-Period Version)

import os
import sys # Import sys to get command-line arguments
import json
import time
from datetime import datetime
from typing import List, Dict, Any

from groq import Groq
import logging
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore

# --- Initialization (No changes here) ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    service_account_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if not service_account_json_str:
        logger.error("FATAL: GOOGLE_APPLICATION_CREDENTIALS_JSON not found.")
        sys.exit(1)
    
    service_account_info = json.loads(service_account_json_str)
    cred = credentials.Certificate(service_account_info)
    
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    
    db_firestore = firestore.client()
    analyzed_news_collection = db_firestore.collection('analyzed_news')
    daily_briefs_collection = db_firestore.collection('daily_briefs')
    logger.info("BRIEFING WORKER: Firebase initialized successfully.")
except Exception as e:
    logger.error(f"BRIEFING WORKER: Failed to initialize Firebase: {e}", exc_info=True)
    sys.exit(1)

# --- AI Processor Class (with Dynamic Prompt) ---
class BriefingAIProcessor:
    def __init__(self, period: str):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.error("FATAL: GROQ_API_KEY not found.")
            sys.exit(1)
        self.client = Groq(api_key=self.api_key)
        self.model = "llama3-70b-8192"
        self.period = period # Store the period (AM/PM)

    def _get_prompt_for_period(self, context_str: str) -> str:
        # Select prompt instructions based on the period
        if self.period == "AM":
            task_instruction = 'Your task is to create a "Morning Market Briefing" that sets the stage for the upcoming trading day.'
            headline_example = "e.g., 'Markets Poised for Volatility as Inflation Data Looms'"
        else: # PM
            task_instruction = 'Your task is to create a "Mid-day Market Update" that summarizes the key movements and news that have occurred so far today.'
            headline_example = "e.g., 'Tech Stocks Lead Market Rebound; Oil Prices Slip'"
            
        return f"""
        You are an expert financial market analyst AI for the "FinanceFlow" app.
        {task_instruction} Analyze the provided collection of recent news summaries.

        CONTEXT - RECENT NEWS SUMMARIES:
        ---
        {context_str}
        ---

        Based *only* on the context provided, generate the briefing with the following strict JSON structure. Do not include any text outside of the JSON object.

        {{
          "market_headline": "A concise, impactful headline in English, suitable for a {self.period} report. {headline_example}",
          "market_overview": "A single paragraph in English summarizing the key events and market mood, tailored for a {self.period} perspective.",
          "key_drivers_and_outlook": [
            "A bullet point interpreting the most significant news and its impact.",
            "Another bullet point highlighting a different key factor or trend.",
            "A forward-looking statement on what to watch for next."
          ],
          "movers_and_shakers": ["A list of up to 5 most relevant stock ticker symbols central to the news context."]
        }}
        """

    def generate_market_briefing(self, news_summaries: List[str]) -> Dict[str, Any]:
        if not news_summaries:
            logger.warning("No news summaries provided.")
            return {}
        
        context_str = "\n\n---\n\n".join(news_summaries)
        prompt = self._get_prompt_for_period(context_str)
        
        try:
            logger.info(f"Sending request to Groq for [{self.period}] market briefing...")
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            response_content = chat_completion.choices[0].message.content
            return json.loads(response_content)
        except Exception as e:
            logger.error(f"Groq briefing generation failed for period {self.period}: {e}", exc_info=True)
            return {}

# --- Main Execution Logic ---
def main(period: str):
    logger.info(f"--- Starting FinanceFlow Daily Briefing Worker for Period: [{period}] ---")
    
    # Generate a period-specific document ID
    today_date_str = datetime.utcnow().strftime('%Y-%m-%d')
    document_id = f"{today_date_str}_{period}"
    
    logger.info(f"Generating brief for document ID: {document_id}")

    brief_doc_ref = daily_briefs_collection.document(document_id)
    if brief_doc_ref.get().exists:
        logger.info(f"Briefing for {document_id} already exists. Exiting.")
        return

    try:
        logger.info("Fetching latest 15 analyzed news from Firestore...")
        query = analyzed_news_collection.order_by("published", direction=firestore.Query.DESCENDING).limit(30)
        docs = query.stream()
        
        news_summaries = [
            doc.to_dict().get('analysis', {}).get('summary_en', '')
            for doc in docs if doc.to_dict().get('analysis', {}).get('summary_en')
        ]
        
        if not news_summaries:
            logger.warning("No news with summaries found. Exiting.")
            return
            
        logger.info(f"Found {len(news_summaries)} summaries to use as context.")

    except Exception as e:
        logger.error(f"Failed to fetch news from Firestore: {e}", exc_info=True)
        return

    ai_processor = BriefingAIProcessor(period=period)
    briefing_data = ai_processor.generate_market_briefing(news_summaries)
    
    if briefing_data and 'market_headline' in briefing_data:
        try:
            briefing_data['generated_at_utc'] = firestore.SERVER_TIMESTAMP
            briefing_data['period'] = period # Add period to the data itself
            briefing_data['source_news_count'] = len(news_summaries)
            
            brief_doc_ref.set(briefing_data)
            logger.info(f"Successfully generated and saved brief for {document_id}.")
        except Exception as e:
            logger.error(f"Failed to save briefing to Firestore: {e}", exc_info=True)
    else:
        logger.error("AI did not return valid briefing data. Nothing was saved.")

    logger.info(f"--- Worker finished for Period: [{period}] ---")

if __name__ == "__main__":
    # Get the period ('AM' or 'PM') from the command-line arguments
    # sys.argv[0] is the script name, sys.argv[1] is the first argument
    if len(sys.argv) > 1 and sys.argv[1].upper() in ["AM", "PM"]:
        run_period = sys.argv[1].upper()
        main(run_period)
    else:
        error_msg = "FATAL: Missing or invalid period argument. Please run as 'python worker_daily_brief.py AM' or 'python worker_daily_brief.py PM'."
        logger.error(error_msg)
        print(error_msg, file=sys.stderr)
        sys.exit(1)