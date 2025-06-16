#!/usr/bin/env python3
import asyncio
import logging
import signal
import subprocess
import sys
import time

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# Global flag for graceful shutdown
running = True
processes = []

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    logging.info(f"Received signal {signum}, initiating graceful shutdown...")
    running = False
    # Terminate all child processes
    for p in processes:
        try:
            p.terminate()
        except:
            pass

def run_script(script_name):
    """Run a Python script and monitor its output"""
    while running:
        try:
            logging.info(f"Starting {script_name}...")
            process = subprocess.Popen(
                [sys.executable, script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            processes.append(process)
            
            # Stream output
            for line in process.stdout:
                if not running:
                    break
                print(f"[{script_name}] {line.strip()}")
            
            # Wait for process to complete
            return_code = process.wait()
            if return_code != 0:
                logging.error(f"{script_name} exited with code {return_code}")
            
            if not running:
                break
                
            logging.warning(f"{script_name} stopped unexpectedly, restarting in 5 seconds...")
            time.sleep(5)
            
        except Exception as e:
            logging.error(f"Error running {script_name}: {e}")
            if not running:
                break
            time.sleep(5)

async def main():
    """Main function to run both spam detection tasks"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Create tasks for both scripts
        message_task = asyncio.create_task(
            asyncio.to_thread(run_script, "infer_msgs.py")
        )
        profile_task = asyncio.create_task(
            asyncio.to_thread(run_script, "infer_profile.py")
        )

        # Wait for both tasks to complete
        await asyncio.gather(message_task, profile_task)

    except Exception as e:
        logging.error(f"Error in main task: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutting down by user request...")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1) 