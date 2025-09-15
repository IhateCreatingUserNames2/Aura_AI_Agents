import sqlite3
from pathlib import Path

# --- CONFIGURATION ---
DB_FILENAME = "aura_agents.db"
DB_PATH = Path(__file__).resolve().parent / DB_FILENAME


def check_agent_status():
    """Reads the agents table and prints their public status."""
    if not DB_PATH.exists():
        print(f"❌ Database file not found at: {DB_PATH}")
        return

    print(f"🔍 Checking agents in database: {DB_PATH}")
    print("-" * 80)
    print(f"{'Agent Name':<30} | {'Agent ID':<38} | {'is_public_template'}")
    print("-" * 80)

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Select the columns we care about
        cursor.execute("SELECT name, id, is_public_template FROM agents")
        agents = cursor.fetchall()

        if not agents:
            print("No agents found in the database.")
        else:
            for agent in agents:
                name, agent_id, is_template = agent
                status_text = f"✅ YES (Marketplace)" if is_template == 1 else "❌ NO (Private)"
                print(f"{name:<30} | {agent_id:<38} | {status_text}")

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

    print("-" * 80)
    print("\nℹ️  For an agent to appear in the marketplace, 'is_public_template' must be '✅ YES'.")
    print("    Use the 'Publish to Marketplace' button in the UI to create a public template.")


if __name__ == "__main__":
    check_agent_status()