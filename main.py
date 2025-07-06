import argparse
import os
import base64
import json
import io
import psycopg2
import requests
from PIL import Image
from dotenv import load_dotenv
import openai
from openai.types.chat import ChatCompletionUserMessageParam # Explicit import for type hinting

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI client
# Using base_url and api_key as specified for LM Studio or custom API
client = openai.OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
                       api_key=os.getenv("OPENAI_API_KEY", "lm-studio")) # Default to lm-studio if not in .env

# LLM Prompt - defined globally as it's static
LLM_INSTRUCTION_PROMPT = """
Extract the following information from this image of a receipt or transaction:
- date_of_transaction (format:YYYY-MM-DD)
- vendor
- category (choose from: toiletry, housewares, fast food, groceries, clothing, gas, car maintenance, utilities, gardening, home repair, tools, renovations, other, entertainment, travel, electronics, furniture, healthcare, education, subscriptions, donations)
- brief_description_of_purchased_items

Provide the output as a JSON object. Example:
{
    "date_of_transaction": "2025-01-26",
    "vendor": "Walmart",
    "category": "groceries",
    "brief_description_of_purchased_items": "Milk, eggs, bread, apples"
}
If a field cannot be extracted, set its value to null. Ensure the category is one of the specified options.

Other context:
    some of the gas stations in canada: Canadian Tire, Chevron, Domo Gasoline, Esso, Federated Co-operatives, Fifth Wheel Truck Stops, FJ Management, Gulf Canada, Husky Energy, Imperial Oil, Irving Oil, Joy Gas Stations, Little Chief Service Station, Mohawk Oil, Nuns' Island gas station, OLCO Petroleum Group, Parkland Corporation, Petro-Canada, Pilot Flying J, Pioneer Energy, Shell Canada, Suncor Energy, Supertest Petroleum, Ultramar, Wilson Fuel

tips for brief_description_of_purchased_items:
    IF category is 'gas', then attempt to identify the type of gas purchased (e.g., regular, premium, diesel) and the qty purchased.
"""

# Valid categories for validation
VALID_CATEGORIES = {'toiletry', 'housewares', 'fast food', 'groceries', 'clothing', 'gas', 'car maintenance', 'utilities', 'gardening', 'home repair', 'tools', 'renovations', 'other', 'entertainment', 'travel', 'electronics', 'furniture', 'healthcare', 'education', 'subscriptions', 'donations'}


def encode_image(image_path):
    """Encodes an image to a base64 string suitable for OpenAI's Vision API."""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def process_llm_extraction(image_base64_data_url, filename):
    """
    Sends an image to the LLM for data extraction and returns the parsed JSON.
    """
    llm_extracted_data = None
    try:
        messages = [
            ChatCompletionUserMessageParam(
                role="user",
                content=[
                    {"type": "text", "text": LLM_INSTRUCTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_base64_data_url}
                    }
                ],
            )
        ]

        # Use the model as specified (e.g., "google/gemma-3-27b" for LM Studio)
        openai_response = client.chat.completions.create(
            model="google/gemma-3-27b",
            messages=messages,
            # For LM Studio with some local models, response_format might not be fully supported
            # or the model might not adhere perfectly. We'll strip markdown tags if present.
            # response_format={"type": "json_object"}, # Uncomment if your specific LM Studio model supports it well
            temperature=0.1,
            max_tokens=1000
        )

        if openai_response.choices and openai_response.choices[0].message.content:
            raw_llm_response_content = openai_response.choices[0].message.content

            # Remove ```json and ``` from string if present, as LM Studio models might output them
            raw_llm_response_content = raw_llm_response_content.replace("```json", "").replace("```", "").strip()

            try:
                llm_extracted_data = json.loads(raw_llm_response_content)

                category_from_llm = llm_extracted_data.get("category")
                if category_from_llm and category_from_llm.lower() not in VALID_CATEGORIES:
                    print(f"Warning: Invalid category '{category_from_llm}' for {filename}. Setting to 'other'.")
                    llm_extracted_data["category"] = 'other'
                elif not category_from_llm:
                    llm_extracted_data["category"] = 'other'

                return llm_extracted_data

            except json.JSONDecodeError as e:
                print(f"Error parsing LLM response for {filename}: {e}")
                print(f"Raw LLM response: {raw_llm_response_content}")
                return None
        else:
            print(f"No valid response content from LLM for {filename}.")
            return None

    except openai.APIStatusError as e:
        print(f"OpenAI API Error for {filename}: Status {e.status_code}, Type: {e.type}, Message: {e.message}")
        return None
    except openai.APIConnectionError as e:
        print(f"OpenAI Connection Error for {filename}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred with LLM for {filename}: {e}")
        return None

def upload_image_to_nocodb_storage(image_path, nocodb_api_base_url, nocodb_api_token):
    """
    Uploads an image file to NocoDB's storage API.

    Args:
        image_path (str): The local path to the image file.
        nocodb_api_base_url (str): The base URL for NocoDB API (e.g., "https://app.nocodb.com/").
        nocodb_api_token (str): Your NocoDB API token.

    Returns:
        dict or None: JSON object to be used in insert record API for attachment field, None on failure.
    """
    # Note: Using the hardcoded one directly as per your working code
    storage_upload_url = f"{nocodb_api_base_url}/api/v2/storage/upload" 
    
    headers = {
        "xc-token": nocodb_api_token,
    }

    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(storage_upload_url, headers=headers, files=files)
            response.raise_for_status()
            print(f"Successfully uploaded image {os.path.basename(image_path)} to NocoDB storage.")

            if not response.json() or not isinstance(response.json(), list) or len(response.json()) == 0:
                print(f"Unexpected response format from NocoDB storage upload for {os.path.basename(image_path)}.")
                return None
            return response.json()[0] # Returns the first (and only) JSON object for the attachment
            
    except requests.exceptions.HTTPError as e:
        print(f"Error uploading image to NocoDB storage for {os.path.basename(image_path)}: HTTP {e.response.status_code}")
        try:
            error_details = e.response.json()
            print(f"NocoDB Error Message: {error_details.get('msg', e.response.text)}")
        except json.JSONDecodeError:
            print(f"NocoDB Error Response (non-JSON): {e.response.text}")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Error connecting to NocoDB storage: {e}")
        return None
    except requests.exceptions.Timeout:
        print(f"NocoDB storage upload timed out for {os.path.basename(image_path)}.")
        return None
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during NocoDB image upload for {os.path.basename(image_path)}: {e}")
        return None

def send_results_to_supabase(results_to_upload, conn):
    """
    Sends a list of extracted LLM data to Supabase.
    """
    if not os.getenv("SUPABASE_ENABLED", "False").lower() == "true":
        print("Supabase upload is disabled by SUPABASE_ENABLED environment variable.")
        return

    cur = conn.cursor()
    supabase_processed_count = 0
    supabase_skipped_count = 0

    print("\n--- Starting Supabase Uploads ---")
    for data, filename, _ in results_to_upload: # _ ignores image_path as it's not needed for Supabase here
        try:
            insert_sql = """
            INSERT INTO public.transactions (date_of_transaction, vendor, category, description, image_filename)
            VALUES (%s, %s, %s, %s, %s);
            """
            cur.execute(insert_sql, (
                data.get("date_of_transaction"),
                data.get("vendor"),
                data.get("category"),
                data.get("brief_description_of_purchased_items"),
                filename
            ))
            conn.commit() # Commit after each successful insert
            print(f"Successfully inserted {filename} into Supabase.")
            supabase_processed_count += 1
        except psycopg2.Error as e:
            print(f"Database error inserting data for {filename} into Supabase: {e}")
            conn.rollback()
            supabase_skipped_count += 1
        except Exception as e:
            print(f"An unexpected error occurred during Supabase insertion for {filename}: {e}")
            supabase_skipped_count += 1
    print(f"Supabase Summary: Processed {supabase_processed_count}, Skipped {supabase_skipped_count}")
    cur.close() # Close cursor after batch operations

def send_results_to_nocodb(results_to_upload, nocodb_api_base_url=None, nocodb_table_id=None):
    """
    Uploads images and then sends extracted data records to NocoDB.
    Args:
        results_to_upload (list): List of tuples containing (llm_data, original_filename, image_path).
        nocodb_api_base_url (str): Base URL for NocoDB API.
        nocodb_table_id (str): NocoDB table ID to insert records into.
    """
    if not os.getenv("NOCODB_ENABLED", "False").lower() == "true":
        print("NocoDB upload is disabled by NOCODB_ENABLED environment variable.")
        return

    nocodb_api_url = f"{nocodb_api_base_url}/api/v2/tables/{nocodb_table_id}/records" if nocodb_api_base_url and nocodb_table_id else None
    nocodb_api_token = os.getenv("NOCODB_API_TOKEN")

    if not nocodb_api_url or not nocodb_api_token:
        print("NocoDB API URL or Token not set in .env. Skipping NocoDB batch insertion.")
        return

    nocodb_processed_count = 0
    nocodb_skipped_count = 0

    print("\n--- Starting NocoDB Uploads ---")
    for llm_data, original_filename, image_path in results_to_upload:
        # Step 1: Upload the image to NocoDB storage
        uploaded_image_info = upload_image_to_nocodb_storage(image_path, nocodb_api_base_url, nocodb_api_token)
        
        if not uploaded_image_info:
            print(f"Failed to upload image for {original_filename} to NocoDB storage. Skipping record insertion.")
            nocodb_skipped_count += 1
            continue

        headers = {
            "xc-token": nocodb_api_token,
            "Content-Type": "application/json",
        }

        # Step 2: Use the returned information in the payload for the record insertion
        payload = {
            "Date of Transaction": llm_data.get("date_of_transaction"),
            "Purchase Method": "credit",
            "Vendor": llm_data.get("vendor"),
            "Category": llm_data.get("category"),
            "Description": llm_data.get("brief_description_of_purchased_items"),
            "Image Attachment": [uploaded_image_info]
        }

        try:
            response = requests.post(nocodb_api_url, headers=headers, json=payload)
            response.raise_for_status()

            print(f"Successfully sent data and image attachment for {original_filename} to NocoDB.")
            nocodb_processed_count += 1

        except requests.exceptions.HTTPError as e:
            print(f"Error sending data to NocoDB for {original_filename}: HTTP {e.response.status_code}")
            try:
                error_details = e.response.json()
                if "msg" in error_details:
                    print(f"NocoDB Error Message: {error_details['msg']}")
            except json.JSONDecodeError:
                print(f"NocoDB Error Response (non-JSON): {e.response.text}")
            nocodb_skipped_count += 1
        except requests.exceptions.ConnectionError as e:
            print(f"Error connecting to NocoDB: {e}")
            nocodb_skipped_count += 1
        except requests.exceptions.Timeout:
            print(f"NocoDB request timed out for {original_filename}.")
            nocodb_skipped_count += 1
        except Exception as e:
            print(f"An unexpected error occurred while sending to NocoDB for {original_filename}: {e}")
            nocodb_skipped_count += 1
    print(f"NocoDB Summary: Processed {nocodb_processed_count}, Skipped {nocodb_skipped_count}")


def main():
    parser = argparse.ArgumentParser(description="Process images with OpenAI Vision API and send data to Supabase and NocoDB.")
    parser.add_argument("folder_path", type=str, help="Path to the directory containing images.")
    args = parser.parse_args()

    # Supabase PostgreSQL connection details from environment variables
    DB_HOST = os.getenv("SUPABASE_HOST")
    DB_DATABASE = os.getenv("SUPABASE_DATABASE")
    DB_USER = os.getenv("SUPABASE_USER")
    DB_PASSWORD = os.getenv("SUPABASE_PASSWORD")
    DB_PORT = os.getenv("SUPABASE_PORT", "5432")

    # Validate Supabase environment variables if Supabase is enabled
    if os.getenv("SUPABASE_ENABLED", "False").lower() == "true":
        if not all([DB_HOST, DB_DATABASE, DB_USER, DB_PASSWORD]):
            print("Error: Supabase database credentials must be set in your .env file if SUPABASE_ENABLED is True.")
            return

    # Validate OpenAI API Key
    # Using local LM Studio, so API key might be "lm-studio" or not needed for base_url
    if not os.getenv("OPENAI_API_KEY"):
         print("Warning: OPENAI_API_KEY not set in .env file. Using default 'lm-studio'.")
         # If you intend to use OpenAI's cloud API, this should be an error
    
    conn = None
    try:
        if os.getenv("SUPABASE_ENABLED", "False").lower() == "true":
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_DATABASE,
                user=DB_USER,
                password=DB_PASSWORD,
                port=DB_PORT
            )
            print(f"Successfully connected to Supabase PostgreSQL pooler at {DB_HOST}:{DB_PORT}.")

        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
        
        # List to store successfully extracted data for batch uploads
        results_for_upload = [] 

        total_files = 0
        total_skipped_llm = 0

        for filename in os.listdir(args.folder_path):
            if filename.lower().endswith(image_extensions):
                total_files += 1
                image_path = os.path.join(args.folder_path, filename)
                print(f"\nProcessing image: {image_path}")

                image_base64_data_url = encode_image(image_path)
                if not image_base64_data_url:
                    total_skipped_llm += 1
                    continue

                llm_extracted_data = process_llm_extraction(image_base64_data_url, filename)
                
                if llm_extracted_data:
                    # Collect results for batch upload
                    results_for_upload.append((llm_extracted_data, filename, image_path))
                else:
                    total_skipped_llm += 1
            else:
                print(f"Skipping non-image file: {filename}")
                # Don't increment total_files or skipped_llm for non-image files

        # --- Perform Batch Uploads after processing all images ---
        if results_for_upload:
            if os.getenv("SUPABASE_ENABLED", "False").lower() == "true" and conn:
                send_results_to_supabase(results_for_upload, conn)
            else:
                print("Supabase upload skipped (disabled or no connection).")

            if os.getenv("NOCODB_ENABLED", "False").lower() == "true":
                send_results_to_nocodb(results_for_upload, os.getenv("NOCODB_API_BASE_URL"), os.getenv("NOCODB_TABLE_ID"))
            else:
                print("NocoDB upload skipped (disabled).")
        else:
            print("No data extracted successfully from any image. Skipping database uploads.")

    except psycopg2.Error as e:
        print(f"Could not connect to Supabase PostgreSQL pooler: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")
        print(f"\n--- Overall Summary ---")
        print(f"Total image files found: {total_files}")
        print(f"Images where LLM extraction failed or were not images: {total_skipped_llm}")
        print(f"Images ready for upload to DBs (LLM extracted successfully): {len(results_for_upload)}")


if __name__ == "__main__":
    main()