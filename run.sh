#!/bin/bash

# 2ex.ch Service Launcher

echo "2ex.ch - Frictionless Cryptocurrency Exchange"
echo "============================================="
echo ""

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container..."
else
    echo "Running locally..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python 3 is not installed"
        exit 1
    fi
    
    # Check/Install dependencies
    echo "Checking dependencies..."
    pip3 install -q requests flask flask-cors 2>/dev/null || {
        echo "Installing dependencies..."
        pip3 install requests flask flask-cors
    }
fi

# Check for .env file
if [ ! -f .env ]; then
    echo "Warning: .env file not found"
    echo "Create .env with:"
    echo '  EXOLIX_SECRET="your_exolix_api_key"'
    echo '  FIXEDFLOAT_KEY="your_fixedfloat_key"'
    echo '  FIXEDFLOAT_SECRET="your_fixedfloat_secret"'
    echo ""
fi

# Start service
echo "Starting 2ex.ch service on port 5000..."
echo "Access at: http://localhost:5000"
echo ""
echo "Example usage:"
echo "  curl localhost:5000/ltc2xmr4YOUR_MONERO_ADDRESS"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the service
python3 twoex_service.py