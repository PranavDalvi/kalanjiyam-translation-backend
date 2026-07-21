import argparse
import sys
from app.api_key import generate_api_key, list_api_keys, revoke_api_key, init_db


def main():
    parser = argparse.ArgumentParser(description="Manage API Keys for Kalanjiyam Translation Backend")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: create
    create_parser = subparsers.add_parser("create", help="Generate a new API key")
    create_parser.add_argument("--name", "-n", required=True, help="App/Client name for the API key owner")

    # Command: list
    subparsers.add_parser("list", help="List all generated API keys")

    # Command: revoke
    revoke_parser = subparsers.add_parser("revoke", help="Revoke an API key by ID or prefix")
    revoke_parser.add_argument("target", help="Key Prefix (e.g. kt_a1b2) or numeric Key ID to revoke")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    init_db()

    if args.command == "create":
        raw_key, prefix = generate_api_key(args.name)
        print("\n========================================================")
        print("                NEW API KEY GENERATED                   ")
        print("========================================================")
        print(f" Owner/App Name : {args.name}")
        print(f" Key Prefix     : {prefix}")
        print(f" API Key        : {raw_key}")
        print("--------------------------------------------------------")
        print(" IMPORTANT: Save this key now! It will NOT be shown again.")
        print("========================================================\n")

    elif args.command == "list":
        keys = list_api_keys()
        if not keys:
            print("No API keys found.")
            return

        print(f"\n{'ID':<5} | {'Prefix':<10} | {'Status':<8} | {'Created At':<25} | {'Name'}")
        print("-" * 75)
        for k in keys:
            status_str = "ACTIVE" if k["is_active"] else "REVOKED"
            print(f"{k['id']:<5} | {k['key_prefix']:<10} | {status_str:<8} | {k['created_at']:<25} | {k['name']}")
        print()

    elif args.command == "revoke":
        success = revoke_api_key(args.target)
        if success:
            print(f"Successfully revoked API key matching '{args.target}'.")
        else:
            print(f"Error: No key matching '{args.target}' was found.")


if __name__ == "__main__":
    main()
