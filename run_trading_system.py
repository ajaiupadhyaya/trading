import subprocess
import sys
import os
from dotenv import load_dotenv

def check_requirements():
    """Check if all required environment variables are set"""
    load_dotenv()
    required_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables in your .env file")
        sys.exit(1)

def start_services():
    """Start all required services"""
    # Start the API server
    api_process = subprocess.Popen(
        ["uvicorn", "app.api.trading_api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Start the dashboard
    dashboard_process = subprocess.Popen(
        ["python", "app/dashboard.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Start the trading system
    trading_process = subprocess.Popen(
        ["python", "-m", "app.trading_system"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    return api_process, dashboard_process, trading_process

def main():
    """Main function to run the trading system"""
    print("Starting Trading System...")
    
    # Check requirements
    check_requirements()
    
    # Start services
    api_process, dashboard_process, trading_process = start_services()
    
    print("\nTrading System is running!")
    print("Dashboard: http://localhost:8050")
    print("API Documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop all services")
    
    try:
        # Keep the script running
        while True:
            # Check if any process has terminated
            if api_process.poll() is not None:
                print("API server has stopped")
                break
            if dashboard_process.poll() is not None:
                print("Dashboard has stopped")
                break
            if trading_process.poll() is not None:
                print("Trading system has stopped")
                break
            
            # Sleep for a bit
            import time
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping all services...")
        
    finally:
        # Terminate all processes
        for process in [api_process, dashboard_process, trading_process]:
            if process.poll() is None:
                process.terminate()
                process.wait()
        
        print("All services stopped")

if __name__ == "__main__":
    main() 