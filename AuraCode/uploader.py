import requests
import json
import os
import argparse
import getpass
from pathlib import Path
from io import BytesIO

# --- Configuration ---
# You can change the default API URL here if needed.
DEFAULT_API_URL = "http://localhost:8000"


def login(api_url: str, username: str, password: str) -> str | None:
    """
    Logs into the Aura API and returns an authentication token.
    Returns None if login fails.
    """
    login_endpoint = f"{api_url}/auth/login"
    login_payload = {
        "username": username,
        "password": password
    }
    headers = {"Content-Type": "application/json"}

    print(f"Attempting to log in as '{username}'...")
    try:
        response = requests.post(login_endpoint, json=login_payload, headers=headers)

        if response.status_code == 200:
            token = response.json().get("access_token")
            print("✅ Login successful!")
            return token
        else:
            print(f"❌ Login failed. Status code: {response.status_code}")
            print(f"   Reason: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ A network error occurred while trying to log in: {e}")
        return None


def upload_agent_from_biography(api_url: str, auth_token: str, file_path: Path) -> bool:
    """
    Reads a biography JSON file, sets it to public, and uploads it to create a new agent.
    Returns True on success, False on failure.
    """
    upload_endpoint = f"{api_url}/agents/create/from-biography"
    headers = {"Authorization": f"Bearer {auth_token}"}

    print(f"\nProcessing file: {file_path.name}")

    try:
        # 1. Read the JSON file into a Python dictionary
        with open(file_path, 'r', encoding='utf-8') as f:
            biography_data = json.load(f)

        # 2. **CRITICAL STEP**: Modify the data in memory to make the agent public
        # This ensures it's public regardless of the file's original setting.
        if 'config' in biography_data:
            biography_data['config']['is_public'] = True
            print("   - Set 'is_public' flag to True.")
        else:
            print("   - ⚠️ Warning: 'config' section not found. Cannot set public flag.")
            # Optionally, you could skip the file here if you want to be strict.
            # return False

        # 3. Convert the modified dictionary back to JSON bytes
        modified_json_bytes = json.dumps(biography_data).encode('utf-8')

        # 4. Prepare the file for multipart/form-data upload
        files_payload = {
            'file': (file_path.name, BytesIO(modified_json_bytes), 'application/json')
        }

        # 5. Make the request
        print(f"   - Uploading to create new public agent...")
        response = requests.post(upload_endpoint, headers=headers, files=files_payload)

        if response.status_code == 200:
            agent_name = response.json().get("name", "Unknown Agent")
            agent_id = response.json().get("agent_id")
            print(f"✅ Success! Created public agent '{agent_name}' (ID: {agent_id}).")
            return True
        else:
            print(f"❌ Upload failed for {file_path.name}. Status code: {response.status_code}")
            print(f"   Reason: {response.text}")
            return False

    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return False
    except json.JSONDecodeError:
        print(f"❌ Error: Could not parse JSON from {file_path.name}. Please check its format.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ A network error occurred during upload: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred with {file_path.name}: {e}")
        return False
def add_biography_to_agent(api_url: str, auth_token: str, agent_id: str, file_path: Path) -> bool:
    """
    Reads a JSON file with a 'biography' list and adds those memories to an existing agent.
    Returns True on success, False on failure.
    """
    update_endpoint = f"{api_url}/agents/{agent_id}/biography/add"
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

    print(f"\nProcessing file to update agent: {agent_id}")
    print(f"   - Source file: {file_path.name}")

    try:
        # 1. Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. Validate that the file has a "biography" key which is a list
        if 'biography' not in data or not isinstance(data['biography'], list):
            print(f"❌ Error: The file '{file_path.name}' must contain a top-level key 'biography' with a list of memories.")
            return False

        # 3. Prepare the payload
        payload = {
            "biography": data['biography']
        }

        # 4. Make the request
        print(f"   - Adding {len(payload['biography'])} new biographical memories...")
        response = requests.post(update_endpoint, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Added {result.get('memories_added', 0)} memories to agent {agent_id}.")
            if result.get('errors'):
                print(f"   - Warnings/Errors: {result['errors']}")
            return True
        else:
            print(f"❌ Update failed for agent {agent_id}. Status code: {response.status_code}")
            print(f"   Reason: {response.text}")
            return False

    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return False
    except json.JSONDecodeError:
        print(f"❌ Error: Could not parse JSON from {file_path.name}. Please check its format.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ A network error occurred during update: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred with {file_path.name}: {e}")
        return False


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Bulk upload or update agent biographies on the Aura AI platform."
    )
    # --- Group for mutually exclusive actions: create vs update ---
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--create",
        dest="directory",
        type=str,
        help="Path to the directory with .json files to CREATE new agents."
    )
    action_group.add_argument(
        "--update",
        dest="agent_id",
        type=str,
        help="The ID of an EXISTING agent to add memories to."
    )

    parser.add_argument(
        "--file",
        type=str,
        help="The single .json file to use for the --update action."
    )
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Your Aura AI username."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"The base URL of the Aura AI API (default: {DEFAULT_API_URL})."
    )
    args = parser.parse_args()

    # --- Validate arguments for --update action ---
    if args.agent_id and not args.file:
        parser.error("--update requires --file.")

    # Securely get the password
    password = getpass.getpass(f"Enter password for user '{args.username}': ")

    # 1. Log in to get the token
    token = login(args.url, args.username, password)
    if not token:
        return  # Exit if login fails

    # --- Execute the chosen action ---
    if args.directory:
        # --- CREATE MODE ---
        source_directory = Path(args.directory)
        if not source_directory.is_dir():
            print(f"❌ Error: The specified path '{args.directory}' is not a valid directory.")
            return

        json_files = list(source_directory.glob("*.json"))
        if not json_files:
            print(f"ℹ️ No .json files found in '{source_directory}'. Nothing to do.")
            return

        print(f"\nFound {len(json_files)} JSON file(s) to process for CREATION.")
        success_count = 0
        failure_count = 0
        for file_path in json_files:
            if upload_agent_from_biography(args.url, token, file_path):
                success_count += 1
            else:
                failure_count += 1

        print("\n" + "=" * 30)
        print("      UPLOAD SUMMARY (CREATE)")
        print("=" * 30)
        print(f"Successful uploads: {success_count}")
        print(f"Failed uploads:     {failure_count}")
        print("=" * 30)

    elif args.agent_id:
        # --- UPDATE MODE ---
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"❌ Error: The specified file '{args.file}' does not exist.")
            return

        add_biography_to_agent(args.url, token, args.agent_id, file_path)


if __name__ == "__main__":
    main()