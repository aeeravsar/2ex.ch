# 2ex.ch

Zero-friction cryptocurrency exchange aggregator. No registration, no API keys, just send crypto.

## Quick Start

```bash
# Exchange 0.5 LTC for XMR
curl 2ex.ch/ltc2xmr:YourMoneroAddress:0.5

# Returns deposit address - send LTC there, receive XMR
```

## How It Works

1. Request exchange with destination address and amount to send
2. Get deposit address
3. Send exact amount to deposit address
4. Receive exchanged currency at your address

Automatically selects provider with lowest fees from Exolix, FixedFloat, and Alfacash. (There will be more soon)

## API

### Create Exchange

```bash
# Basic exchange (amount = what you SEND)
GET /{from}2{to}:{address}:{amount}

# Minimum amount exchange
GET /{from}2{to}:{address}

# With network specification (destination only)
GET /{from}2{to}:{network}:{address}:{amount}
```

**Examples:**
```bash
curl 2ex.ch/btc2eth:0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb7:0.1
curl 2ex.ch/btc2usdt:trc:TAddress:1
curl 2ex.ch/eth2btc:bc1qAddress:1
```

### Status & Management

```bash
GET /s/{id}    # Check transaction status
GET /d/{id}    # Delete transaction from database
```

### Rates & Information

```bash
GET /i/{from}2{to}      # Detailed pair info with fees
GET /c/{from}2{to}      # Compare all provider rates
GET /p                  # List supported pairs
GET /pn                 # List pairs with network variants
GET /pr                 # Provider status
GET /h                  # Help (also /)
```

## Network Specification

For multi-chain tokens like USDT, specify network after currency pair:

```bash
btc2usdt:eth:0xAddr:1    # USDT on Ethereum
btc2usdt:trc:TAddr:1     # USDT on Tron
btc2usdt:bsc:0xAddr:1    # USDT on BSC
```

Networks: eth, trc, bsc, matic, avax, sol, trx

## Self-Hosting

```bash
# Clone
git clone https://github.com/aeeravsar/2ex.ch
cd 2ex.ch

# Configure .env
cp .env.example .env
nano .env

# Run
python3 twoex_service.py -p 5000
```

Service runs on http://localhost:5000

## Features

- **Best rate selection** - Compares effective fees across providers
- **Fee transparency** - Shows fees vs market rates (CoinGecko)
- **Rate limiting** - 100 requests/minute per IP
- **Transaction monitoring** - Auto-updates status
- **No registration** - Completely anonymous

## Response Format

```
Exchange Created: f3605c72

SEND:
  Amount: 0.1 BTC
  To Address: bc1qAddress

RECEIVE:
  Amount: 2.5 ETH
  Your Address: 0xYourAddress

DETAILS:
  Provider: exolix
  Status Check: /s/f3605c72
  Expires: 2025-01-01 12:00:00 UTC

FEE INFO:
  Exchange Rate: 1 BTC = 25.0 ETH
  Effective Fee: 1.89% (Standard fee)
  Market Rate: 1 BTC = 25.5 ETH
```

## Important Notes

- Amount specified is what you **send**, not what you receive
- Send exact amount shown or transaction fails
- Transactions expire after 1 hour
- Minimum/maximum limits enforced by providers
- No KYC for supported amounts

## Supported Currencies

Major: BTC, ETH, LTC, XMR, USDT, USDC, BNB, SOL, DOGE, TRX

Run `/p` for full list, `/pn` for network variants.

## License

[GPL 2.0](LICENSE)
