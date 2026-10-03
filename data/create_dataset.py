"""
Script to create the specialized DevOps Copilot training dataset.
Generates a structured .jsonl file with instruction-response pairs formatted for Qwen ChatML.
"""
import json
import os

DATASET = [
    # Process Management
    {
        "instruction": "Find all processes running on port 8080 and show their PIDs.",
        "response": json.dumps({
            "command": "lsof -i :8080 -t",
            "explanation": "Queries open files/sockets listening on TCP/UDP port 8080 and outputs only their process IDs.",
            "safety_level": "SAFE",
            "caution": "None. Read-only inspection."
        }, indent=2)
    },
    {
        "instruction": "Find and kill all zombie or defunct processes gracefully.",
        "response": json.dumps({
            "command": "ps -ef | grep defunct | grep -v grep | awk '{print $3}' | xargs -r kill -s SIGCHLD",
            "explanation": "Identifies parent PIDs of defunct (zombie) processes and sends SIGCHLD to make parents reap them.",
            "safety_level": "MEDIUM",
            "caution": "Never kill PID 1 (systemd/init). Only signals parent processes."
        }, indent=2)
    },
    {
        "instruction": "Show the top 5 memory consuming processes currently active.",
        "response": json.dumps({
            "command": "ps aux --sort=-%mem | head -n 6",
            "explanation": "Lists all running processes sorted in descending order of memory usage and prints the top 5 plus header.",
            "safety_level": "SAFE",
            "caution": "None. Read-only command."
        }, indent=2)
    },
    {
        "instruction": "Terminate all hanging Python scripts immediately.",
        "response": json.dumps({
            "command": "pkill -9 -f python",
            "explanation": "Sends SIGKILL (signal 9) to all processes matching the pattern 'python' in their command line.",
            "safety_level": "HIGH",
            "caution": "Forces immediate termination. Unsaved state or database connections will not exit cleanly."
        }, indent=2)
    },
    
    # Disk & Storage
    {
        "instruction": "Find the top 10 largest files or directories taking up space in /var/log.",
        "response": json.dumps({
            "command": "du -ah /var/log | sort -rh | head -n 10",
            "explanation": "Recursively calculates human-readable disk usage for all files in /var/log and displays the 10 largest.",
            "safety_level": "SAFE",
            "caution": "Requires sudo privileges if log files have restricted permissions."
        }, indent=2)
    },
    {
        "instruction": "Find and delete all log files older than 30 days in /var/log/custom.",
        "response": json.dumps({
            "command": "find /var/log/custom -type f -name '*.log' -mtime +30 -exec rm -f {} +",
            "explanation": "Searches for files ending with .log with modification timestamps greater than 30 days and removes them in batches.",
            "safety_level": "HIGH",
            "caution": "Permanently deletes files. Verify with a dry run ('find ... -name \"*.log\" -mtime +30') before execution."
        }, indent=2)
    },
    {
        "instruction": "Check disk space usage across all mounted filesystems in human-readable format.",
        "response": json.dumps({
            "command": "df -hT",
            "explanation": "Displays total, used, and available disk space along with the filesystem type for all mounted drives.",
            "safety_level": "SAFE",
            "caution": "None."
        }, indent=2)
    },
    {
        "instruction": "Find all files in /home modified in the last 24 hours.",
        "response": json.dumps({
            "command": "find /home -type f -mtime -1",
            "explanation": "Searches for regular files under /home whose modification time is less than 1 day (24 hours).",
            "safety_level": "SAFE",
            "caution": "May take several seconds if /home contains large directory structures."
        }, indent=2)
    },

    # Networking & Security
    {
        "instruction": "List all active TCP listening sockets and the process names attached to them.",
        "response": json.dumps({
            "command": "ss -tulpn | grep LISTEN",
            "explanation": "Inspects socket statistics for TCP listeners, displaying process name and PID associated with each socket.",
            "safety_level": "SAFE",
            "caution": "Requires sudo/root privileges to view process names of daemons owned by other users."
        }, indent=2)
    },
    {
        "instruction": "Check if port 443 is open and responding on remote host example.com.",
        "response": json.dumps({
            "command": "nc -zv -w 5 example.com 443",
            "explanation": "Uses netcat in zero-I/O mode with a 5-second timeout to verify TCP connectivity to port 443.",
            "safety_level": "SAFE",
            "caution": "May be blocked by intermediate firewall or rate-limiting rules."
        }, indent=2)
    },
    {
        "instruction": "Block all incoming traffic from malicious IP 198.51.100.25 using UFW firewall.",
        "response": json.dumps({
            "command": "sudo ufw deny from 198.51.100.25 to any",
            "explanation": "Inserts a firewall rule into Ubuntu Uncomplicated Firewall to drop all packets originating from 198.51.100.25.",
            "safety_level": "MEDIUM",
            "caution": "Ensure you are not blocking your own management or bastion IP address."
        }, indent=2)
    },
    {
        "instruction": "Generate a modern, secure ED25519 SSH keypair with a comment.",
        "response": json.dumps({
            "command": "ssh-keygen -t ed25519 -C 'devops@company.internal' -f ~/.ssh/id_ed25519",
            "explanation": "Generates a high-security elliptic curve ED25519 SSH private and public key pair.",
            "safety_level": "SAFE",
            "caution": "Do not overwrite existing keys if ~/.ssh/id_ed25519 already exists."
        }, indent=2)
    },

    # Docker & Containers
    {
        "instruction": "Remove all stopped containers, unused networks, and dangling images to free up space.",
        "response": json.dumps({
            "command": "docker system prune -f",
            "explanation": "Reclaims disk space by purging stopped containers, unreferenced networks, and untagged images without prompting.",
            "safety_level": "MEDIUM",
            "caution": "Will remove stopped development containers. Use without -f to review items first."
        }, indent=2)
    },
    {
        "instruction": "View live streaming logs of container 'api-gateway' starting from the last 100 lines.",
        "response": json.dumps({
            "command": "docker logs -f --tail 100 api-gateway",
            "explanation": "Attaches to the stdout/stderr stream of the container, outputting the most recent 100 entries and tailing new logs.",
            "safety_level": "SAFE",
            "caution": "Press Ctrl+C to disconnect from log stream without stopping the container."
        }, indent=2)
    },
    {
        "instruction": "Open an interactive bash shell inside a running container named 'postgres-db'.",
        "response": json.dumps({
            "command": "docker exec -it postgres-db /bin/bash",
            "explanation": "Allocates a pseudo-TTY and keeps stdin open inside the running container executing the bash shell.",
            "safety_level": "SAFE",
            "caution": "Commands executed inside the container will directly affect the container runtime."
        }, indent=2)
    },
    {
        "instruction": "Inspect live CPU and memory utilization of all running Docker containers.",
        "response": json.dumps({
            "command": "docker stats --no-stream",
            "explanation": "Captures a single snapshot of resource utilization (CPU %, Memory, Network I/O) for all running containers.",
            "safety_level": "SAFE",
            "caution": "None."
        }, indent=2)
    },

    # Git Operations
    {
        "instruction": "Undo the most recent git commit but keep all modified code staged.",
        "response": json.dumps({
            "command": "git reset --soft HEAD~1",
            "explanation": "Moves HEAD back by one commit while keeping all changes in the staging index (ready to recommit).",
            "safety_level": "SAFE",
            "caution": "Do not use if the commit has already been pushed to a protected shared branch."
        }, indent=2)
    },
    {
        "instruction": "Discard all local uncommitted changes and restore working directory to clean state.",
        "response": json.dumps({
            "command": "git restore . && git clean -fd",
            "explanation": "Restores all tracked files to match HEAD and removes all untracked files and directories.",
            "safety_level": "HIGH",
            "caution": "Permanently deletes all uncommitted and untracked work. This cannot be undone."
        }, indent=2)
    },
    {
        "instruction": "Stash only unstaged changes while keeping staged changes intact.",
        "response": json.dumps({
            "command": "git stash push --keep-index -m 'temporary unstaged work'",
            "explanation": "Saves unstaged modifications to the git stash stack while leaving indexed/staged files in the working tree.",
            "safety_level": "SAFE",
            "caution": "Can be retrieved later with 'git stash pop'."
        }, indent=2)
    },
    {
        "instruction": "Find which commit introduced a bug using git binary search.",
        "response": json.dumps({
            "command": "git bisect start && git bisect bad && git bisect good v1.2.0",
            "explanation": "Initializes git bisect wizard marking current commit as broken and version 1.2.0 as known good.",
            "safety_level": "SAFE",
            "caution": "Run 'git bisect reset' when finished to return to normal branch state."
        }, indent=2)
    },

    # System Logs & Systemd
    {
        "instruction": "Inspect systemd service logs for nginx failed within the past 1 hour.",
        "response": json.dumps({
            "command": "journalctl -u nginx --since '1 hour ago' -p err..emerg --no-pager",
            "explanation": "Queries journald for units named nginx filtering by log priority error through emergency in the last hour.",
            "safety_level": "SAFE",
            "caution": "Requires sudo or membership in the systemd-journal group."
        }, indent=2)
    },
    {
        "instruction": "Restart the docker service and verify its current running status.",
        "response": json.dumps({
            "command": "sudo systemctl restart docker && systemctl status docker --no-pager",
            "explanation": "Restarts the Docker daemon via systemd and prints the immediate operational status.",
            "safety_level": "MEDIUM",
            "caution": "Will temporarily disconnect active container bridge networks."
        }, indent=2)
    },
    {
        "instruction": "Enable a systemd service called 'backend-api' so it automatically starts on boot.",
        "response": json.dumps({
            "command": "sudo systemctl enable --now backend-api.service",
            "explanation": "Creates symlinks in systemd targets for autostart on boot and immediately starts the service.",
            "safety_level": "SAFE",
            "caution": "Ensure the service configuration file in /etc/systemd/system/ is valid before enabling."
        }, indent=2)
    },
    {
        "instruction": "Monitor kernel ring buffer messages in real-time for hardware or driver events.",
        "response": json.dumps({
            "command": "sudo dmesg -wH",
            "explanation": "Follows kernel log events in real-time with human-readable timestamps and colorized output.",
            "safety_level": "SAFE",
            "caution": "Press Ctrl+C to terminate the monitoring stream."
        }, indent=2)
    },

    # File Parsing & Text Processing
    {
        "instruction": "Count unique IP addresses accessing an nginx access.log file.",
        "response": json.dumps({
            "command": "awk '{print $1}' /var/log/nginx/access.log | sort | uniq -c | sort -nr | head -n 20",
            "explanation": "Extracts the client IP column, sorts values, calculates frequency count, and outputs the top 20 active IPs.",
            "safety_level": "SAFE",
            "caution": "Ensure the log format uses column 1 for remote host address."
        }, indent=2)
    },
    {
        "instruction": "Replace all occurrences of 'http://old-api.com' with 'https://new-api.com' in all .env files recursively.",
        "response": json.dumps({
            "command": "find . -type f -name '*.env' -exec sed -i 's|http://old-api.com|https://new-api.com|g' {} +",
            "explanation": "Finds all .env files and performs in-place stream editing to replace the target URL string.",
            "safety_level": "HIGH",
            "caution": "Performs in-place file modifications. Commit or back up files before executing."
        }, indent=2)
    },
    {
        "instruction": "Extract the third column of a comma-separated CSV file named data.csv.",
        "response": json.dumps({
            "command": "cut -d',' -f3 data.csv",
            "explanation": "Parses data.csv using comma as the field delimiter and prints only field 3.",
            "safety_level": "SAFE",
            "caution": "Does not handle escaped commas inside quoted strings; use python/csvkit for complex CSVs."
        }, indent=2)
    },
    {
        "instruction": "Compress a directory named 'project' into a tar.gz archive preserving file permissions.",
        "response": json.dumps({
            "command": "tar -czpf project_backup.tar.gz ./project",
            "explanation": "Creates a gzip-compressed archive preserving absolute permissions and ownership flags.",
            "safety_level": "SAFE",
            "caution": "Ensure sufficient free disk space exists to write the compressed archive."
        }, indent=2)
    },
    {
        "instruction": "Monitor live network bandwidth by interface in the terminal.",
        "response": json.dumps({
            "command": "sar -n DEV 1 5",
            "explanation": "Collects network interface statistics once every second for 5 iterations reporting Rx/Tx kB/s.",
            "safety_level": "SAFE",
            "caution": "Requires the 'sysstat' package installed."
        }, indent=2)
    },
    {
        "instruction": "Check SSL/TLS certificate expiration date for website https://antigravity.internal.",
        "response": json.dumps({
            "command": "echo | openssl s_client -servername antigravity.internal -connect antigravity.internal:443 2>/dev/null | openssl x509 -noout -dates",
            "explanation": "Initiates an SSL handshake, extracts the server certificate, and parses the notBefore and notAfter expiration dates.",
            "safety_level": "SAFE",
            "caution": "None."
        }, indent=2)
    }
]


def generate_jsonl(output_path: str = "./data/devops_dataset.jsonl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in DATASET:
            f.write(json.dumps(item) + "\n")
    print(f"Successfully generated {len(DATASET)} training samples at {output_path}")


if __name__ == "__main__":
    generate_jsonl()
