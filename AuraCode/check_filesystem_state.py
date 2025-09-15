# Quick check of current file system state
import json
from pathlib import Path


def check_current_state():
    """Check the current state of agent configurations in file system"""

    # Values from your database
    EXPECTED_USER_ID = "40c8d42f-858d-4060-b5be-8cef6480e9a3"  # xupeta
    TARGET_AGENT_ID = "4974f67b-6a43-4eee-ba2b-2f9511fe6260"  # Aura

    agent_storage_path = Path("agent_storage")

    print("=== Current File System State ===")
    print(f"Agent storage path: {agent_storage_path.absolute()}")
    print(f"Expected user_id: {EXPECTED_USER_ID}")
    print(f"Target agent_id: {TARGET_AGENT_ID}")

    if not agent_storage_path.exists():
        print("❌ Agent storage directory does not exist!")
        return

    print(f"\nDirectory structure:")
    for item in agent_storage_path.iterdir():
        if item.is_dir():
            print(f"📁 {item.name}/")

            # Check if this is the expected user directory
            is_expected_user = item.name == EXPECTED_USER_ID
            if is_expected_user:
                print(f"   ✅ This is xupeta's user directory")

            # List JSON files
            json_files = list(item.glob("*.json"))
            for json_file in json_files:
                if json_file.name == "memory_blossom.json":
                    print(f"   📄 {json_file.name} (memory file)")
                    continue

                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    agent_id = data.get('agent_id', 'Unknown')
                    user_id = data.get('user_id', 'Unknown')
                    name = data.get('name', 'Unknown')

                    print(f"   📄 {json_file.name}")
                    print(f"      Agent: {name}")
                    print(f"      agent_id: {agent_id}")
                    print(f"      user_id: {user_id}")

                    # Check if this is our target agent
                    if agent_id == TARGET_AGENT_ID:
                        print(f"      🎯 This is the target agent!")
                        if user_id == EXPECTED_USER_ID:
                            print(f"      ✅ user_id matches database")
                        else:
                            print(f"      ❌ user_id MISMATCH! Expected: {EXPECTED_USER_ID}")

                except Exception as e:
                    print(f"   ❌ Error reading {json_file.name}: {e}")
        else:
            print(f"📄 {item.name}")

    # Check if expected user directory exists
    expected_user_dir = agent_storage_path / EXPECTED_USER_ID
    if expected_user_dir.exists():
        print(f"\n✅ Expected user directory exists: {expected_user_dir}")

        # Check if target agent config exists in correct location
        target_config = expected_user_dir / f"{TARGET_AGENT_ID}.json"
        if target_config.exists():
            print(f"✅ Target agent config exists in correct location: {target_config}")
        else:
            print(f"❌ Target agent config NOT found at: {target_config}")
    else:
        print(f"\n❌ Expected user directory does NOT exist: {expected_user_dir}")


if __name__ == "__main__":
    check_current_state()