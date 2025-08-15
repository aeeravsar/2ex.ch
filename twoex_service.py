#!/usr/bin/env python3
"""
2ex.ch - Frictionless cryptocurrency exchange service
A multithreaded Flask API for instant crypto swaps
"""

import os
import json
import uuid
import time
import logging
import threading
import argparse
from datetime import datetime, timedelta, timezone
import pytz
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import sqlite3

# Import our exchange providers
from exolix import ExolixClient, RateType as ExolixRateType, ExolixError
from fixedfloat import FixedFloatClient, OrderType as FFOrderType, FixedFloatError
from alfacash import AlfacashClient, AlfacashError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('2ex.ch')


class RateLimiter:
    """Per-IP rate limiter with sliding window"""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests = defaultdict(deque)  # IP -> deque of timestamps
        self.lock = threading.Lock()
    
    def is_allowed(self, ip: str) -> bool:
        """Check if request from IP is allowed"""
        with self.lock:
            now = time.time()
            ip_requests = self.requests[ip]
            
            # Remove old requests outside the window
            while ip_requests and ip_requests[0] <= now - self.window_seconds:
                ip_requests.popleft()
            
            # Check if under limit
            if len(ip_requests) >= self.max_requests:
                return False
            
            # Add current request
            ip_requests.append(now)
            return True
    
    def get_retry_after(self, ip: str) -> int:
        """Get seconds until next request is allowed"""
        with self.lock:
            ip_requests = self.requests.get(ip, deque())
            if not ip_requests:
                return 0
            
            oldest_request = ip_requests[0]
            return max(0, int(oldest_request + self.window_seconds - time.time()))


# Global rate limiter
rate_limiter = RateLimiter(max_requests=100, window_minutes=1)


def check_rate_limit():
    """Check rate limit for current request"""
    # Get real IP address (handle proxy headers)
    ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', '127.0.0.1'))
    if ',' in ip:
        ip = ip.split(',')[0].strip()  # Take first IP if comma-separated
    
    if not rate_limiter.is_allowed(ip):
        retry_after = rate_limiter.get_retry_after(ip)
        logger.warning(f"Rate limit exceeded for IP {ip}")
        return f"Rate limit exceeded. Try again in {retry_after} seconds.\n", 429, {
            'Content-Type': 'text/plain',
            'Retry-After': str(retry_after)
        }
    return None


class Provider(Enum):
    """Supported exchange providers"""
    EXOLIX = "exolix"
    FIXEDFLOAT = "fixedfloat"
    ALFACASH = "alfacash"


class TransactionStatus(Enum):
    """Transaction status states"""
    CREATED = "created"
    WAITING_DEPOSIT = "waiting_deposit"
    CONFIRMING = "confirming"
    EXCHANGING = "exchanging"
    SENDING = "sending"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    REFUNDED = "refunded"


@dataclass
class Transaction:
    """Transaction data model"""
    id: str
    provider: str
    provider_tx_id: str
    from_currency: str
    to_currency: str
    to_address: str
    deposit_address: str
    deposit_amount: float
    expected_amount: float
    actual_amount: Optional[float]
    status: str
    created_at: str
    updated_at: str
    expires_at: str
    metadata: Dict[str, Any]


class TransactionManager:
    """Manages transactions with SQLite database"""
    
    def __init__(self, db_path: str = "twoex.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    provider_tx_id TEXT NOT NULL,
                    from_currency TEXT NOT NULL,
                    to_currency TEXT NOT NULL,
                    to_address TEXT NOT NULL,
                    deposit_address TEXT NOT NULL,
                    deposit_amount REAL NOT NULL,
                    expected_amount REAL NOT NULL,
                    actual_amount REAL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON transactions(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_provider_tx ON transactions(provider_tx_id)')
            conn.commit()
    
    def create(self, transaction: Transaction) -> Transaction:
        """Create new transaction"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transaction.id,
                    transaction.provider,
                    transaction.provider_tx_id,
                    transaction.from_currency,
                    transaction.to_currency,
                    transaction.to_address,
                    transaction.deposit_address,
                    transaction.deposit_amount,
                    transaction.expected_amount,
                    transaction.actual_amount,
                    transaction.status,
                    transaction.created_at,
                    transaction.updated_at,
                    transaction.expires_at,
                    json.dumps(transaction.metadata)
                ))
                conn.commit()
        return transaction
    
    def get(self, tx_id: str) -> Optional[Transaction]:
        """Get transaction by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute('SELECT * FROM transactions WHERE id = ?', (tx_id,)).fetchone()
            if row:
                return Transaction(
                    id=row['id'],
                    provider=row['provider'],
                    provider_tx_id=row['provider_tx_id'],
                    from_currency=row['from_currency'],
                    to_currency=row['to_currency'],
                    to_address=row['to_address'],
                    deposit_address=row['deposit_address'],
                    deposit_amount=row['deposit_amount'],
                    expected_amount=row['expected_amount'],
                    actual_amount=row['actual_amount'],
                    status=row['status'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    expires_at=row['expires_at'],
                    metadata=json.loads(row['metadata'])
                )
        return None
    
    def update_status(self, tx_id: str, status: str, actual_amount: Optional[float] = None):
        """Update transaction status"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                if actual_amount is not None:
                    conn.execute('''
                        UPDATE transactions 
                        SET status = ?, actual_amount = ?, updated_at = ?
                        WHERE id = ?
                    ''', (status, actual_amount, datetime.now(timezone.utc).isoformat(), tx_id))
                else:
                    conn.execute('''
                        UPDATE transactions 
                        SET status = ?, updated_at = ?
                        WHERE id = ?
                    ''', (status, datetime.now(timezone.utc).isoformat(), tx_id))
                conn.commit()
    
    def delete(self, tx_id: str) -> bool:
        """Delete transaction by ID"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('DELETE FROM transactions WHERE id = ?', (tx_id,))
                conn.commit()
                return cursor.rowcount > 0
    
    def get_active_transactions(self) -> List[Transaction]:
        """Get all active transactions for monitoring"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM transactions 
                WHERE status IN (?, ?, ?, ?)
            ''', (
                TransactionStatus.WAITING_DEPOSIT.value,
                TransactionStatus.CONFIRMING.value,
                TransactionStatus.EXCHANGING.value,
                TransactionStatus.SENDING.value
            )).fetchall()
            
            return [Transaction(
                id=row['id'],
                provider=row['provider'],
                provider_tx_id=row['provider_tx_id'],
                from_currency=row['from_currency'],
                to_currency=row['to_currency'],
                to_address=row['to_address'],
                deposit_address=row['deposit_address'],
                deposit_amount=row['deposit_amount'],
                expected_amount=row['expected_amount'],
                actual_amount=row['actual_amount'],
                status=row['status'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                expires_at=row['expires_at'],
                metadata=json.loads(row['metadata'])
            ) for row in rows]


class ExchangeService:
    """Main exchange service handling multiple providers"""
    
    def __init__(self):
        self.tx_manager = TransactionManager()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.monitoring_active = True
        
        # Load API credentials
        self._load_credentials()
        
        # Provider activation status
        self.provider_status = {
            Provider.EXOLIX: True,  # Active by default
            Provider.FIXEDFLOAT: False,  # Deactivated as requested
            Provider.ALFACASH: True  # Active by default
        }
        
        # Caching system
        self.cache_file = "pairs_cache.json"
        self.cache_duration = 24 * 60 * 60  # 24 hours in seconds
        
        # Initialize provider clients
        self.providers = {}
        if self.exolix_key:
            self.providers[Provider.EXOLIX] = ExolixClient(self.exolix_key)
            logger.info("Exolix provider initialized")
        
        if self.ff_key and self.ff_secret:
            self.providers[Provider.FIXEDFLOAT] = FixedFloatClient(self.ff_key, self.ff_secret)
            logger.info("FixedFloat provider initialized")
        
        # Alfacash doesn't need API keys
        self.providers[Provider.ALFACASH] = AlfacashClient()
        logger.info("Alfacash provider initialized")
        
        if not self.providers:
            logger.warning("No exchange providers configured!")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_transactions, daemon=True)
        self.monitor_thread.start()
    
    def _load_credentials(self):
        """Load API credentials from environment"""
        # Try to load from .env file if exists
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value.strip('"')
        
        self.exolix_key = os.getenv('EXOLIX_SECRET')
        self.ff_key = os.getenv('FIXEDFLOAT_KEY')
        self.ff_secret = os.getenv('FIXEDFLOAT_SECRET')
        self.alfacash_email = os.getenv('ALFACASH_EMAIL', 'noreply@2ex.ch')
    
    def parse_request(self, request_string: str) -> Dict[str, Any]:
        """Parse 2ex.ch style request: ltc2xmr:address, ltc2xmr:address:amount, or xmr2usdt:eth:address:amount"""
        # Split by colons to get parts
        parts = request_string.split(':')
        
        if len(parts) < 2:
            raise ValueError("Invalid format - use: {from}2{to}:{address} or {from}2{to}:{network}:{address} or {from}2{to}:{address}:{amount}")
        
        # First part is currency pair
        pair_part = parts[0]
        
        # Parse currency pair first
        currency_sep = pair_part.find('2')
        if currency_sep == -1:
            raise ValueError("Invalid format - use: {from}2{to}:{address} or {from}2{to}:{network}:{address}")
        
        from_currency_raw = pair_part[:currency_sep].upper()
        to_currency_raw = pair_part[currency_sep + 1:].upper()
        
        if not from_currency_raw or not to_currency_raw:
            raise ValueError("Invalid currency codes")
        
        # Parse from_currency and from_network using the strict parser
        from_currency, from_network = self._parse_currency_network(from_currency_raw)
        
        # Parse to_currency and to_network using the strict parser
        to_currency, to_network = self._parse_currency_network(to_currency_raw)
        
        # Now determine what the remaining parts are
        # Possible formats:
        # 1. {from}2{to}:{address}
        # 2. {from}2{to}:{address}:{amount}
        # 3. {from}2{to:network}:{address}
        # 4. {from}2{to:network}:{address}:{amount}
        # 5. {from:network}2{to}:{address}
        # 6. {from:network}2{to}:{address}:{amount}
        # 7. {from:network}2{to:network}:{address}
        # 8. {from:network}2{to:network}:{address}:{amount}
        
        # If we have network in currency pair, adjust parts indexing
        if ':' in to_currency_raw:
            # Format: {from}2{to:network}:{address}[:{amount}]
            if len(parts) < 3:
                raise ValueError("Invalid format - network specified but no address provided")
            address = parts[2]
            amount_part_idx = 3
        else:
            # Format: {from}2{to}:{address}[:{amount}] or {from}2{to}:{network}:{address}[:{amount}]
            address_or_network = parts[1]
            
            # Check if parts[1] looks like a network (common networks: eth, trc, bsc, etc.)
            common_networks = {'eth', 'trc', 'bsc', 'matic', 'avax', 'sol', 'trx', 'bitcoin', 'litecoin', 'monero'}
            if (len(parts) >= 3 and address_or_network.lower() in common_networks and 
                to_currency.upper() in ['USDT', 'USDC', 'BTC', 'ETH']):
                # Format: {from}2{to}:{network}:{address}[:{amount}]
                to_network = address_or_network.upper()
                address = parts[2]
                amount_part_idx = 3
            else:
                # Format: {from}2{to}:{address}[:{amount}]
                address = address_or_network
                amount_part_idx = 2
        
        if not address:
            raise ValueError("No destination address provided")
        
        # Basic address validation
        address = address.strip()
        if len(address) < 10 or len(address) > 200:
            raise ValueError("Invalid address length")
        
        # Check for common invalid characters
        if any(char in address for char in [' ', '\n', '\t', '\r']):
            raise ValueError("Address contains invalid characters")
        
        # Parse amount if provided
        amount = None
        if len(parts) > amount_part_idx:
            try:
                amount = float(parts[amount_part_idx])
                # Validate amount is a real number
                import math
                if math.isnan(amount) or math.isinf(amount):
                    raise ValueError("Amount must be a valid number")
                if amount <= 0:
                    raise ValueError("Amount must be positive")
                if amount > 1e9:  # Sanity check - no one is exchanging 1 billion of anything
                    raise ValueError("Amount too large")
            except ValueError as e:
                if "could not convert" in str(e):
                    raise ValueError("Invalid amount specified")
                raise e
        
        return {
            'from_currency': from_currency,
            'to_currency': to_currency,
            'from_network': from_network,
            'to_network': to_network,
            'to_address': address,
            'from_amount': amount  # This is the amount sender wants to send
        }
    
    def create_exchange(self, request_string: str, provider: Optional[Provider] = None) -> Dict[str, Any]:
        """Create new exchange transaction"""
        try:
            # Parse request
            parsed = self.parse_request(request_string)
            
            # Validate currencies
            all_currencies = set()
            for p in self.providers.keys():
                if self.provider_status.get(p, False):
                    provider_currencies = self._get_provider_currencies(p)
                    all_currencies.update(provider_currencies)
            
            from crypto_mapper import crypto_mapper
            all_currencies.update(crypto_mapper.get_all_symbols())
            
            if parsed['from_currency'].upper() not in all_currencies:
                raise ValueError(f"Invalid source currency: {parsed['from_currency']}. Use /p to see supported currencies.")
            
            if parsed['to_currency'].upper() not in all_currencies:
                raise ValueError(f"Invalid destination currency: {parsed['to_currency']}. Use /p to see supported currencies.")
            
            # Validate networks if specified
            if parsed.get('from_network') and not self._validate_network(parsed['from_currency'], parsed['from_network']):
                raise ValueError(f"Invalid network '{parsed['from_network']}' for {parsed['from_currency']}")
            
            if parsed.get('to_network') and not self._validate_network(parsed['to_currency'], parsed['to_network']):
                raise ValueError(f"Invalid network '{parsed['to_network']}' for {parsed['to_currency']}")
            
            # Select provider (use best rate from active providers if not specified)
            if provider is None:
                provider = self._select_best_provider(parsed['from_currency'], parsed['to_currency'])
                if provider is None:
                    raise Exception("No active exchange providers available")
            
            # Get fee information for transparency
            comparison = self.get_provider_comparison(parsed['from_currency'], parsed['to_currency'], 1.0)
            selected_provider_data = comparison['providers'].get(provider.value, {})
            
            if provider not in self.providers:
                raise Exception(f"Provider {provider.value} not configured")
            
            if not self.provider_status.get(provider, False):
                raise Exception(f"Provider {provider.value} is deactivated")
            
            # Create exchange with selected provider
            if provider == Provider.EXOLIX:
                result = self._create_exolix_exchange(parsed)
            elif provider == Provider.FIXEDFLOAT:
                result = self._create_fixedfloat_exchange(parsed)
            elif provider == Provider.ALFACASH:
                result = self._create_alfacash_exchange(parsed)
            else:
                raise Exception(f"Unsupported provider: {provider.value}")
            
            # Store transaction
            tx_id = str(uuid.uuid4())[:8]  # Short ID for easy use
            transaction = Transaction(
                id=tx_id,
                provider=provider.value,
                provider_tx_id=result['provider_tx_id'],
                from_currency=parsed['from_currency'],
                to_currency=parsed['to_currency'],
                to_address=parsed['to_address'],
                deposit_address=result['deposit_address'],
                deposit_amount=result['deposit_amount'],
                expected_amount=result['expected_amount'],
                actual_amount=None,
                status=TransactionStatus.WAITING_DEPOSIT.value,
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                metadata=result.get('metadata', {})
            )
            
            self.tx_manager.create(transaction)
            
            # Add fee information to response
            fee_info = {}
            if 'effective_fee_percent' in selected_provider_data:
                fee_info = {
                    'effective_fee_percent': selected_provider_data['effective_fee_percent'],
                    'fee_note': selected_provider_data['fee_note'],
                    'market_rate': selected_provider_data.get('market_rate')
                }
            
            return {
                'success': True,
                'id': tx_id,
                'deposit_address': result['deposit_address'],
                'deposit_amount': result['deposit_amount'],
                'deposit_currency': parsed['from_currency'],
                'expected_amount': result['expected_amount'],
                'destination_currency': parsed['to_currency'],
                'destination_address': parsed['to_address'],
                'from_network': parsed.get('from_network'),
                'to_network': parsed.get('to_network'),
                'provider': provider.value,
                'fee_info': fee_info,
                'expires_at': self._format_time(transaction.expires_at),
                'status_url': f'/s/{tx_id}'
            }
            
        except Exception as e:
            logger.error(f"Failed to create exchange: {e}")
            # Sanitize error messages - don't expose internal details
            error_msg = str(e)
            if "API Error:" in error_msg:
                # Keep API errors as they're already sanitized
                pass
            elif "Invalid" in error_msg or "not found" in error_msg.lower():
                # Keep validation errors
                pass
            else:
                # Generic error for unexpected issues
                error_msg = "Exchange creation failed. Please try again."
            
            return {
                'success': False,
                'error': error_msg
            }
    
    def _create_exolix_exchange(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Create exchange using Exolix"""
        client = self.providers[Provider.EXOLIX]
        
        # Create transaction based on whether user specified from_amount
        if parsed.get('from_amount'):
            # Check minimum first when user specifies an amount
            try:
                rate = client.get_rate(
                    parsed['from_currency'],
                    parsed['to_currency'],
                    1.0,
                    rate_type=ExolixRateType.FIXED
                )
                min_to_amount = float(rate.get('minToAmount', rate.get('minAmount', 0)))
                max_to_amount = float(rate.get('maxToAmount', rate.get('maxAmount', 999999)))
                
                min_from_amount = float(rate.get('minAmount', 0))
                max_from_amount = float(rate.get('maxAmount', 999999))
                
                if parsed['from_amount'] < min_from_amount:
                    raise ValueError(f"Amount too small. Minimum: {min_from_amount} {parsed['from_currency']}")
                if parsed['from_amount'] > max_from_amount:
                    raise ValueError(f"Amount too large. Maximum: {max_from_amount} {parsed['from_currency']}")
                    
            except ExolixError:
                pass  # If rate check fails, let the transaction attempt proceed
            
            # User specified the amount they want to send
            # Use amount parameter
            tx = client.create_transaction(
                coin_from=parsed['from_currency'],
                coin_to=parsed['to_currency'],
                network_from=parsed.get('from_network'),
                network_to=parsed.get('to_network'),
                withdrawal_address=parsed['to_address'],
                amount=parsed['from_amount'],  # Amount receiver wants
                rate_type=ExolixRateType.FIXED
            )
            logger.info(f"Created tx with amount={parsed['from_amount']}, response: {tx}")
        else:
            # No amount specified, use minimum
            rate = client.get_rate(
                parsed['from_currency'],
                parsed['to_currency'],
                1.0,
                rate_type=ExolixRateType.FIXED
            )
            min_amount = float(rate.get('minAmount', 0.001))
            
            tx = client.create_transaction(
                coin_from=parsed['from_currency'],
                coin_to=parsed['to_currency'],
                network_from=parsed.get('from_network'),
                network_to=parsed.get('to_network'),
                withdrawal_address=parsed['to_address'],
                amount=min_amount,  # Minimum amount to send
                rate_type=ExolixRateType.FIXED
            )
        
        # Handle the response - check all possible field names
        withdrawal_amt = tx.get('withdrawalAmount') or tx.get('amountTo') or tx.get('toAmount') or 0
        deposit_amt = tx.get('amount') or tx.get('amountFrom') or tx.get('fromAmount', 0)
        
        return {
            'provider_tx_id': tx.get('id', ''),
            'deposit_address': tx.get('depositAddress', ''),
            'deposit_amount': float(deposit_amt),
            'expected_amount': float(withdrawal_amt),
            'metadata': {
                'network_from': tx.get('networkFrom', ''),
                'network_to': tx.get('networkTo', ''),
                'raw_response': str(tx)  # For debugging
            }
        }
    
    def _create_fixedfloat_exchange(self, parsed: Dict[str, str]) -> Dict[str, Any]:
        """Create exchange using FixedFloat"""
        client = self.providers[Provider.FIXEDFLOAT]
        
        # Get price first
        price = client.get_price(
            parsed['from_currency'],
            parsed['to_currency'],
            1.0,
            direction="from",
            order_type=FFOrderType.FLOAT
        )
        
        # Use specified amount or minimum
        amount = parsed.get('from_amount', float(price.get('min', 0.001)))
        
        # Create order
        order = client.create_order(
            from_currency=parsed['from_currency'],
            to_currency=parsed['to_currency'],
            to_address=parsed['to_address'],
            amount=amount,
            order_type=FFOrderType.FLOAT
        )
        
        return {
            'provider_tx_id': order.get('id', ''),
            'deposit_address': order.get('from', {}).get('address', ''),
            'deposit_amount': amount,
            'expected_amount': float(order.get('to', {}).get('amount', 0)),
            'metadata': {
                'token': order.get('token', ''),
                'rate': price.get('rate', 0)
            }
        }
    
    def get_status(self, tx_id: str) -> Dict[str, Any]:
        """Get transaction status"""
        tx = self.tx_manager.get(tx_id)
        if not tx:
            return {'success': False, 'error': 'Transaction not found'}
        
        # Update status from provider if still active
        if tx.status in [TransactionStatus.WAITING_DEPOSIT.value, 
                        TransactionStatus.CONFIRMING.value,
                        TransactionStatus.EXCHANGING.value,
                        TransactionStatus.SENDING.value]:
            self._update_transaction_status(tx)
            tx = self.tx_manager.get(tx_id)  # Reload after update
        
        return {
            'success': True,
            'id': tx.id,
            'status': tx.status,
            'from_currency': tx.from_currency,
            'to_currency': tx.to_currency,
            'deposit_address': tx.deposit_address,
            'deposit_amount': tx.deposit_amount,
            'expected_amount': tx.expected_amount,
            'actual_amount': tx.actual_amount,
            'created_at': self._format_time(tx.created_at),
            'updated_at': self._format_time(tx.updated_at),
            'expires_at': self._format_time(tx.expires_at),
            'provider': tx.provider,
            'to_address': tx.to_address
        }
    
    def _update_transaction_status(self, tx: Transaction):
        """Update transaction status from provider"""
        try:
            if tx.provider == Provider.EXOLIX.value:
                client = self.providers[Provider.EXOLIX]
                status = client.get_transaction(tx.provider_tx_id)
                
                # Map Exolix status to our status
                exolix_status = status.get('status', '').lower()
                if exolix_status == 'wait':
                    new_status = TransactionStatus.WAITING_DEPOSIT.value
                elif exolix_status in ['confirmation', 'confirmed']:
                    new_status = TransactionStatus.CONFIRMING.value
                elif exolix_status == 'exchanging':
                    new_status = TransactionStatus.EXCHANGING.value
                elif exolix_status == 'sending':
                    new_status = TransactionStatus.SENDING.value
                elif exolix_status == 'success':
                    new_status = TransactionStatus.COMPLETED.value
                elif exolix_status in ['overdue', 'failed']:
                    new_status = TransactionStatus.FAILED.value
                elif exolix_status == 'refunded':
                    new_status = TransactionStatus.REFUNDED.value
                else:
                    new_status = tx.status
                
                actual_amount = status.get('withdrawalAmount')
                
            elif tx.provider == Provider.FIXEDFLOAT.value:
                client = self.providers[Provider.FIXEDFLOAT]
                token = tx.metadata.get('token', '')
                status = client.get_order(tx.provider_tx_id, token)
                
                # Map FixedFloat status to our status
                ff_status = status.get('status', '').upper()
                if ff_status == 'NEW':
                    new_status = TransactionStatus.WAITING_DEPOSIT.value
                elif ff_status == 'PENDING':
                    new_status = TransactionStatus.CONFIRMING.value
                elif ff_status == 'EXCHANGING':
                    new_status = TransactionStatus.EXCHANGING.value
                elif ff_status == 'WITHDRAWING':
                    new_status = TransactionStatus.SENDING.value
                elif ff_status == 'COMPLETED':
                    new_status = TransactionStatus.COMPLETED.value
                elif ff_status == 'FAILED':
                    new_status = TransactionStatus.FAILED.value
                elif ff_status == 'EMERGENCY':
                    new_status = TransactionStatus.FAILED.value
                else:
                    new_status = tx.status
                
                actual_amount = status.get('to', {}).get('amount')
            
            elif tx.provider == Provider.ALFACASH.value:
                client = self.providers[Provider.ALFACASH]
                secret_key = tx.metadata.get('secret_key', '')
                status = client.get_transaction(secret_key)
                
                # Map Alfacash status to our status
                ac_status = status.get('status', '').lower()
                if ac_status == 'waiting':
                    new_status = TransactionStatus.WAITING_DEPOSIT.value
                elif ac_status in ['received', 'confirmed']:
                    new_status = TransactionStatus.CONFIRMING.value
                elif ac_status == 'exchanging':
                    new_status = TransactionStatus.EXCHANGING.value
                elif ac_status == 'withdrawal':
                    new_status = TransactionStatus.SENDING.value
                elif ac_status == 'completed':
                    new_status = TransactionStatus.COMPLETED.value
                elif ac_status in ['cancelled', 'expired']:
                    new_status = TransactionStatus.FAILED.value
                elif ac_status == 'refunded':
                    new_status = TransactionStatus.REFUNDED.value
                else:
                    new_status = tx.status
                
                actual_amount = status.get('withdrawal_amount')
            
            else:
                return
            
            # Update if status changed
            if new_status != tx.status:
                self.tx_manager.update_status(tx.id, new_status, actual_amount)
                logger.info(f"Transaction {tx.id} status updated: {tx.status} -> {new_status}")
                
        except Exception as e:
            logger.error(f"Failed to update status for {tx.id}: {e}")
    
    def _monitor_transactions(self):
        """Background thread to monitor active transactions"""
        logger.info("Transaction monitor started")
        
        while self.monitoring_active:
            try:
                active_txs = self.tx_manager.get_active_transactions()
                
                # Track futures for proper completion
                futures = []
                
                for tx in active_txs:
                    # Check if expired
                    if datetime.fromisoformat(tx.expires_at) < datetime.now(timezone.utc):
                        self.tx_manager.update_status(tx.id, TransactionStatus.EXPIRED.value)
                        logger.info(f"Transaction {tx.id} expired")
                        continue
                    
                    # Update status from provider
                    future = self.executor.submit(self._update_transaction_status, tx)
                    futures.append(future)
                
                # Wait for all updates to complete (with timeout)
                from concurrent.futures import wait, FIRST_COMPLETED
                if futures:
                    completed, pending = wait(futures, timeout=20)
                    # Cancel any pending futures that timed out
                    for future in pending:
                        future.cancel()
                
                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _load_cache(self) -> Optional[Dict[str, Any]]:
        """Load pairs cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # Check if cache is still valid
                cache_time = cache_data.get('timestamp', 0)
                if time.time() - cache_time < self.cache_duration:
                    return cache_data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_cache(self, data: Dict[str, Any]):
        """Save pairs cache to file"""
        try:
            cache_data = {
                'timestamp': time.time(),
                'data': data
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _invalidate_cache(self):
        """Remove cache file to force refresh"""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                logger.info("Pairs cache invalidated")
        except Exception as e:
            logger.warning(f"Failed to invalidate cache: {e}")
    
    def _get_provider_currencies(self, provider: Provider) -> List[str]:
        """Get currency list from specific provider"""
        try:
            if provider not in self.providers or not self.provider_status.get(provider, False):
                return []
                
            if provider == Provider.EXOLIX:
                client = self.providers[Provider.EXOLIX]
                currencies = client.get_currencies()
                
                # Handle different response formats
                if isinstance(currencies, list):
                    return [curr.get('code') if isinstance(curr, dict) else str(curr) for curr in currencies]
                elif isinstance(currencies, dict) and 'data' in currencies:
                    return [curr.get('code') if isinstance(curr, dict) else str(curr) for curr in currencies['data']]
                    
            elif provider == Provider.ALFACASH:
                client = self.providers[Provider.ALFACASH]
                return client.get_currencies()
                
            elif provider == Provider.FIXEDFLOAT:
                # FixedFloat doesn't have a currencies endpoint, use common ones
                return ['BTC', 'ETH', 'LTC', 'XMR', 'BCH', 'XRP', 'USDT', 'USDC', 'BNB', 'ADA']
                
        except Exception as e:
            logger.warning(f"Failed to get currencies from {provider.value}: {e}")
        
        return []
    
    def _get_all_currencies(self) -> List[str]:
        """Get merged currency list from all active providers"""
        all_currencies = set()
        
        for provider in Provider:
            currencies = self._get_provider_currencies(provider)
            all_currencies.update(currencies)
            
        return sorted(list(all_currencies))
    
    def _generate_pairs(self, currencies: List[str], include_networks: bool = False) -> List[str]:
        """Generate all possible pairs from currency list"""
        pairs = set()
        
        # Basic pairs
        for from_curr in currencies:
            for to_curr in currencies:
                if from_curr != to_curr and from_curr and to_curr:
                    pairs.add(f"{from_curr}2{to_curr}")
        
        # Network variants if requested
        if include_networks:
            multi_network_tokens = {
                'USDT': ['ETH', 'TRX', 'BSC', 'MATIC', 'AVAX'],
                'USDC': ['ETH', 'BSC', 'MATIC', 'AVAX'],
                'BTC': ['BTC', 'BSC'],  # Wrapped BTC
                'ETH': ['ETH', 'BSC', 'MATIC']  # Wrapped ETH
            }
            
            for token, networks in multi_network_tokens.items():
                if token in currencies:
                    for network in networks:
                        network_token = f"{token}{network}"
                        for from_curr in currencies:
                            if from_curr != token:
                                pairs.add(f"{from_curr}2{network_token}")
                                pairs.add(f"{network_token}2{from_curr}")
                        
                        # Cross-network pairs
                        for other_network in networks:
                            if other_network != network:
                                other_token = f"{token}{other_network}"
                                pairs.add(f"{network_token}2{other_token}")
        
        return sorted(list(pairs))
    
    def get_supported_pairs(self, include_networks: bool = False) -> Dict[str, Any]:
        """Get supported currency pairs with 24h caching"""
        try:
            # Try to load from cache first
            cache_key = f"pairs_{'with_networks' if include_networks else 'basic'}"
            cached = self._load_cache()
            
            if cached and cache_key in cached.get('data', {}):
                logger.info(f"Returning cached pairs ({cache_key})")
                cached_pairs = cached['data'][cache_key]
                return {
                    'success': True,
                    'pairs': cached_pairs['pairs'],
                    'total': cached_pairs['total'],
                    'cached': True,
                    'cache_age': time.time() - cached['timestamp']
                }
            
            # Cache miss - rebuild
            logger.info(f"Building fresh pairs list ({cache_key})")
            currencies = self._get_all_currencies()
            pairs = self._generate_pairs(currencies, include_networks)
            
            # Update cache
            cache_data = cached.get('data', {}) if cached else {}
            cache_data[cache_key] = {
                'pairs': pairs,
                'total': len(pairs),
                'currencies': currencies
            }
            self._save_cache(cache_data)
            
            return {
                'success': True,
                'pairs': pairs,
                'total': len(pairs),
                'cached': False,
                'providers_queried': [p.value for p in Provider if self.provider_status.get(p, False)]
            }
            
        except Exception as e:
            logger.error(f"Failed to get supported pairs: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_pair_info(self, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """Get rates and info from ALL active providers (no cache)"""
        try:
            # Parse currencies to extract base currency and network
            from_curr, from_net = self._parse_currency_network(from_currency)
            to_curr, to_net = self._parse_currency_network(to_currency)
            
            # Validate base currencies against ALL provider currencies
            all_currencies = set()
            for provider in self.providers.keys():
                if self.provider_status.get(provider, False):
                    provider_currencies = self._get_provider_currencies(provider)
                    all_currencies.update(provider_currencies)
            
            # Also add from crypto mapper
            from crypto_mapper import crypto_mapper
            all_currencies.update(crypto_mapper.get_all_symbols())
            
            if from_curr not in all_currencies:
                return {
                    'success': False,
                    'error': f'Invalid source currency: {from_curr}. Use /p to see supported currencies.'
                }
            
            if to_curr not in all_currencies:
                return {
                    'success': False,
                    'error': f'Invalid destination currency: {to_curr}. Use /p to see supported currencies.'
                }
            
            comparison = self.get_provider_comparison(from_currency, to_currency, 1.0)
            
            if not comparison['success']:
                return comparison
            
            # Format for single pair info display - use comparison data directly
            result = {
                'success': True,
                'from_currency': from_currency,
                'to_currency': to_currency,
                'providers': {},
                'best_provider': comparison.get('best_provider'),
                'best_rate': comparison.get('best_rate', 0),
                'market_rate': comparison.get('market_rate')
            }
            
            # Add detailed info for each provider - use comparison data and add min/max
            for provider_name, data in comparison['providers'].items():
                if not data.get('active', True):
                    result['providers'][provider_name] = {'status': 'deactivated'}
                elif 'error' in data:
                    result['providers'][provider_name] = {'error': data['error']}
                else:
                    # Start with comparison data (includes fees)
                    provider_data = data.copy()
                    provider_data['selected'] = provider_name == comparison.get('best_provider')
                    
                    # Try to get min/max info from provider
                    provider_enum = None
                    for p in Provider:
                        if p.value == provider_name:
                            provider_enum = p
                            break
                    
                    min_amount = max_amount = None
                    if provider_enum and provider_enum in self.providers:
                        try:
                            # Use parsed currencies with networks for rate info
                            if provider_enum == Provider.EXOLIX:
                                client = self.providers[Provider.EXOLIX]
                                rate_info = client.get_rate(from_curr, to_curr, 1.0, 
                                                          network_from=from_net, network_to=to_net)
                                min_amount = float(rate_info.get('minAmount', 0))
                                max_amount = float(rate_info.get('maxAmount', 0))
                            elif provider_enum == Provider.FIXEDFLOAT:
                                client = self.providers[Provider.FIXEDFLOAT]
                                price_info = client.get_price(from_curr, to_curr, 1.0, direction="from")
                                min_amount = float(price_info.get('min', 0))
                                max_amount = float(price_info.get('max', 0))
                            elif provider_enum == Provider.ALFACASH:
                                client = self.providers[Provider.ALFACASH]
                                # Convert back to Alfacash format for rate lookup
                                alfacash_from = from_currency
                                alfacash_to = to_currency
                                rate_info = client.get_rate(alfacash_from, alfacash_to)
                                min_amount = float(rate_info.get('min_deposit_amount', 0))
                                max_amount = float(rate_info.get('max_deposit_amount', 0))
                        except Exception as e:
                            logger.warning(f"Failed to get limits from {provider_name}: {e}")
                    
                    provider_data['min_amount'] = min_amount
                    provider_data['max_amount'] = max_amount
                    
                    result['providers'][provider_name] = provider_data
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get pair info: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _select_best_provider(self, from_currency: str, to_currency: str) -> Optional[Provider]:
        """Select provider with best rate from active providers"""
        active_providers = [p for p in self.providers.keys() if self.provider_status.get(p, False)]
        
        if not active_providers:
            return None
        
        best_provider = None
        best_fee = float('inf')  # Lower fees are better
        
        for provider in active_providers:
            try:
                rate = self._get_provider_rate(provider, from_currency, to_currency)
                if rate:
                    # Calculate effective fee percentage
                    market_rate = self._get_market_rate(from_currency, to_currency)
                    if market_rate and market_rate > 0:
                        effective_fee = ((market_rate - rate) / market_rate) * 100
                        if effective_fee < best_fee:
                            best_fee = effective_fee
                            best_provider = provider
                    else:
                        # Fallback to rate comparison if market rate unavailable
                        if rate > self._get_provider_rate(best_provider, from_currency, to_currency) if best_provider else 0:
                            best_provider = provider
            except Exception as e:
                logger.warning(f"Failed to get rate from {provider.value}: {e}")
                continue
        
        # Fallback to first active provider if rate comparison fails
        return best_provider or active_providers[0]
    
    def _parse_currency_network(self, currency_string: str) -> tuple[str, Optional[str]]:
        """Parse currency string to extract currency and network"""
        currency = currency_string.upper()
        network = None
        
        # Check for colon separator (new format: "usdt:eth")
        if ':' in currency:
            parts = currency.split(':')
            currency = parts[0]
            network = parts[1].upper()
        # Check for legacy network suffix (old format: "USDTETH", "USDTTRC" etc.)
        # Only allow specific known combinations, not arbitrary suffixes
        elif currency in ['USDTETH', 'USDTTRC', 'USDTTRON', 'USDTBSC']:
            network = currency[4:]  # Extract network part
            currency = 'USDT'
        elif currency in ['USDCETH', 'USDCBSC', 'USDCMATIC', 'USDCAVAX']:
            network = currency[4:]
            currency = 'USDC'
        elif currency in ['BTCBSC', 'BTCETH']:
            # Handle wrapped BTC variants
            network = currency[3:]
            currency = 'BTC'
        elif currency in ['ETHBSC', 'ETHMATIC']:
            # Handle wrapped ETH variants
            network = currency[3:]
            currency = 'ETH'
        # If no colon and not a known combination, treat as plain currency
        # This prevents "USDTDSFDSFDSF" from being parsed as USDT
        
        return currency, network

    def _validate_network(self, currency: str, network: str) -> bool:
        """Validate that a network is supported for a given currency"""
        if not network:
            return True  # No network specified is always valid
            
        currency = currency.upper()
        network = network.upper()
        
        # Define valid networks for each currency
        valid_networks = {
            'USDT': ['ETH', 'ERC20', 'ETHEREUM', 'TRC', 'TRC20', 'TRON', 'BSC', 'BEP20', 'BINANCE'],
            'USDC': ['ETH', 'ERC20', 'ETHEREUM', 'BSC', 'BEP20', 'BINANCE', 'MATIC', 'POLYGON', 'AVAX', 'AVALANCHE'],
            'BTC': ['BTC', 'BITCOIN', 'BSC', 'BEP20'],  # Native BTC and wrapped variants
            'ETH': ['ETH', 'ETHEREUM', 'BSC', 'BEP20', 'MATIC', 'POLYGON'],  # Native ETH and wrapped variants
        }
        
        return network in valid_networks.get(currency, [])

    def _get_market_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Get market rate from CoinGecko for fee calculation"""
        try:
            import requests
            
            # Map currency symbols to CoinGecko IDs
            from crypto_mapper import crypto_mapper
            from_id = crypto_mapper.symbol_to_id(from_currency)
            to_id = crypto_mapper.symbol_to_id(to_currency)
            
            if not from_id or not to_id:
                return None
            
            response = requests.get(
                f'https://api.coingecko.com/api/v3/simple/price?ids={from_id},{to_id}&vs_currencies=usd',
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if from_id in data and to_id in data:
                from_price = data[from_id]['usd']
                to_price = data[to_id]['usd']
                return from_price / to_price
            return None
        except Exception:
            return None

    def _get_provider_rate(self, provider: Provider, from_currency: str, to_currency: str) -> Optional[float]:
        """Get exchange rate from specific provider"""
        try:
            # Parse networks from currency strings
            from_curr, from_net = self._parse_currency_network(from_currency)
            to_curr, to_net = self._parse_currency_network(to_currency)
            
            if provider == Provider.EXOLIX:
                client = self.providers[Provider.EXOLIX]
                rate_info = client.get_rate(from_curr, to_curr, 1.0, 
                                          network_from=from_net, network_to=to_net)
                return float(rate_info.get('toAmount', 0))
            elif provider == Provider.FIXEDFLOAT:
                client = self.providers[Provider.FIXEDFLOAT]
                # FixedFloat doesn't support network specification in rate calls
                price_info = client.get_price(from_curr, to_curr, 1.0, direction="from")
                return float(price_info.get('rate', 0))
            elif provider == Provider.ALFACASH:
                client = self.providers[Provider.ALFACASH]
                # For Alfacash, convert network format back to their format
                alfacash_from = from_curr
                alfacash_to = to_curr
                
                if from_curr == 'USDT' and from_net:
                    if from_net.upper() in ['ETH', 'ERC20', 'ETHEREUM']:
                        alfacash_from = 'USDTETH'
                    elif from_net.upper() in ['TRC', 'TRC20', 'TRON']:
                        alfacash_from = 'USDTTRC'
                    elif from_net.upper() in ['BSC', 'BEP20', 'BINANCE']:
                        alfacash_from = 'USDTBSC'
                
                if to_curr == 'USDT' and to_net:
                    if to_net.upper() in ['ETH', 'ERC20', 'ETHEREUM']:
                        alfacash_to = 'USDTETH'
                    elif to_net.upper() in ['TRC', 'TRC20', 'TRON']:
                        alfacash_to = 'USDTTRC'
                    elif to_net.upper() in ['BSC', 'BEP20', 'BINANCE']:
                        alfacash_to = 'USDTBSC'
                
                rate_info = client.get_rate(alfacash_from, alfacash_to)
                return float(rate_info.get('rate', 0))
        except Exception:
            return None
        return None
    
    def _create_alfacash_exchange(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Create exchange using Alfacash"""
        client = self.providers[Provider.ALFACASH]
        
        # Determine amount
        amount = parsed.get('from_amount')
        if not amount:
            # Get minimum amount from rate
            try:
                rate_info = client.get_rate(parsed['from_currency'], parsed['to_currency'])
                amount = float(rate_info.get('min_deposit_amount', 0.001))
            except:
                amount = 0.001
        
        # Create transaction
        tx = client.create_transaction(
            gate_deposit=parsed['from_currency'],
            gate_withdrawal=parsed['to_currency'],
            withdrawal_address=parsed['to_address'],
            email=self.alfacash_email,
            deposit_amount=amount
        )
        
        return {
            'provider_tx_id': tx.get('order_id', tx.get('id', '')),
            'deposit_address': tx.get('deposit', {}).get('address', ''),
            'deposit_amount': float(tx.get('deposit_amount', amount)),
            'expected_amount': float(tx.get('withdrawal_amount', 0)),
            'metadata': {
                'secret_key': tx.get('secret_key', ''),
                'rate': tx.get('rate', 0),
                'raw_response': str(tx)
            }
        }
    
    def get_provider_comparison(self, from_currency: str, to_currency: str, amount: float = 1.0) -> Dict[str, Any]:
        """Compare rates across all active providers with fee transparency"""
        comparison = {
            'success': True,
            'from_currency': from_currency,
            'to_currency': to_currency,
            'amount': amount,
            'providers': {}
        }
        
        # Get market rate for fee calculation
        market_rate = self._get_market_rate(from_currency, to_currency)
        
        for provider in self.providers.keys():
            if not self.provider_status.get(provider, False):
                continue
                
            try:
                rate = self._get_provider_rate(provider, from_currency, to_currency)
                if rate:
                    provider_data = {
                        'rate': rate,
                        'output_amount': rate * amount,
                        'active': True
                    }
                    
                    # Calculate fee if market rate is available
                    if market_rate:
                        effective_fee = ((market_rate - rate) / market_rate) * 100
                        provider_data['effective_fee_percent'] = round(effective_fee, 2)
                        provider_data['market_rate'] = market_rate
                        
                        # Add fee category
                        if effective_fee < 0:
                            provider_data['fee_note'] = 'Better than market'
                        elif effective_fee < 1:
                            provider_data['fee_note'] = 'Low fee'
                        elif effective_fee < 2:
                            provider_data['fee_note'] = 'Standard fee'
                        else:
                            provider_data['fee_note'] = 'High fee'
                    
                    comparison['providers'][provider.value] = provider_data
                else:
                    comparison['providers'][provider.value] = {
                        'error': 'Rate not available',
                        'active': True
                    }
            except Exception as e:
                comparison['providers'][provider.value] = {
                    'error': str(e),
                    'active': True
                }
        
        # Add inactive providers
        for provider in self.providers.keys():
            if not self.provider_status.get(provider, False):
                comparison['providers'][provider.value] = {
                    'status': 'deactivated',
                    'active': False
                }
        
        # Find best rate
        best_rate = 0
        best_provider = None
        for provider, data in comparison['providers'].items():
            if data.get('active') and 'rate' in data and data['rate'] > best_rate:
                best_rate = data['rate']
                best_provider = provider
        
        comparison['best_provider'] = best_provider
        comparison['best_rate'] = best_rate
        comparison['market_rate'] = market_rate
        
        return comparison
    
    def set_provider_status(self, provider: Provider, active: bool):
        """Activate or deactivate a provider"""
        self.provider_status[provider] = active
        status = "activated" if active else "deactivated"
        logger.info(f"Provider {provider.value} {status}")
        
        # Invalidate pairs cache when provider status changes
        self._invalidate_cache()
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        return {
            'providers': {
                provider.value: {
                    'active': self.provider_status.get(provider, False),
                    'configured': provider in self.providers
                }
                for provider in Provider
            }
        }
    
    def _format_time(self, iso_time: str) -> str:
        """Format ISO time to human readable format"""
        try:
            dt = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
            # Convert to UTC and format nicely
            utc_dt = dt.replace(tzinfo=pytz.UTC)
            return utc_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except:
            return iso_time[:19] + ' UTC'
    
    def shutdown(self):
        """Graceful shutdown"""
        self.monitoring_active = False
        self.executor.shutdown(wait=True)


# Flask application
app = Flask(__name__)
CORS(app)

# Initialize service
service = ExchangeService()


@app.before_request
def before_request():
    """Apply rate limiting to all requests except static content"""
    # Skip rate limiting for favicon and other static content
    if request.path.startswith('/favicon') or request.path.startswith('/static'):
        return None
    
    return check_rate_limit()


@app.route('/')
@app.route('/h')
def index():
    """API documentation"""
    return """2ex.ch - Instant Cryptocurrency Exchange

EXCHANGE:
  {from}2{to}:{address}:{amount}     Create exchange (amount = what you send)
  {from}2{to}:{address}              Create with minimum amount
  {from}2{to}:{network}:{address}:{amount}  Specify destination network

STATUS:
  /s/{id}                            Check transaction status
  /d/{id}                            Delete transaction

RATES:
  /i/{from}2{to}                     Detailed rate info with fees
  /c/{from}2{to}                     Compare provider rates (alias: /compare/)

INFO:
  /p                                 Supported pairs
  /pn                                Pairs with network variants
  /pr                                Provider status
  /h                                 This help message (alias: /)

NETWORK SPECIFICATION:
  For destination only (after first colon):
  btc2usdt:eth:0xAddress:100        Send 100 BTC, receive USDT on Ethereum
  btc2usdt:trc:TAddress:50          Send 50 BTC, receive USDT on Tron
  btc2usdt:bsc:0xAddress:10         Send 10 BTC, receive USDT on BSC

EXAMPLES:
  curl 2ex.ch/btc2eth:0xAddress:0.1
  curl 2ex.ch/btc2usdt:trc:TAddress:100
  curl 2ex.ch/ltc2xmr:4Address:5
  curl 2ex.ch/s/abc123
  curl 2ex.ch/i/btc2ltc
""", 200, {'Content-Type': 'text/plain'}


@app.route('/<path:exchange_request>', methods=['GET'])
def create_exchange(exchange_request):
    """Create exchange from curl-style request"""
    result = service.create_exchange(exchange_request)
    
    if result['success']:
        # Extract network info if available
        from_net = f" ({result['from_network']})" if result.get('from_network') else ""
        to_net = f" ({result['to_network']})" if result.get('to_network') else ""
        
        # Calculate exchange rate
        rate = result['expected_amount'] / result['deposit_amount'] if result['deposit_amount'] > 0 else 0
        
        # Fee information
        fee_text = ""
        if result.get('fee_info') and 'effective_fee_percent' in result['fee_info']:
            fee_info = result['fee_info']
            fee_text = f"\n\nFEE INFO:\n  Exchange Rate: 1 {result['deposit_currency']} = {rate:.6f} {result['destination_currency']}\n  Effective Fee: {fee_info['effective_fee_percent']:.2f}% ({fee_info['fee_note']})"
            if fee_info.get('market_rate'):
                fee_text += f"\n  Market Rate: 1 {result['deposit_currency']} = {fee_info['market_rate']:.6f} {result['destination_currency']}"
        
        text = f"""Exchange Created: {result['id']}

SEND:
  Amount: {result['deposit_amount']} {result['deposit_currency']}{from_net}
  To Address: {result['deposit_address']}

RECEIVE:
  Amount: {result['expected_amount']} {result['destination_currency']}{to_net}
  Your Address: {result['destination_address']}

DETAILS:
  Provider: {result['provider']}
  Status Check: {result['status_url']}
  Expires: {result['expires_at']}{fee_text}

Send exactly {result['deposit_amount']} {result['deposit_currency']} to the deposit address above.
"""
        return text, 200, {'Content-Type': 'text/plain'}
    else:
        return f"Error: {result['error']}\n", 400, {'Content-Type': 'text/plain'}


@app.route('/s/<tx_id>', methods=['GET'])
def get_status(tx_id):
    """Get transaction status"""
    result = service.get_status(tx_id)
    
    if result['success']:
        # Status descriptions
        status_desc = {
            'waiting_deposit': 'Waiting for you to send crypto',
            'confirming': 'Confirming your transaction',
            'exchanging': 'Converting currencies',
            'sending': 'Sending to your address',
            'completed': 'Successfully completed!',
            'failed': 'Transaction failed',
            'expired': 'Transaction expired',
            'refunded': 'Funds refunded'
        }
        
        status_info = status_desc.get(result['status'], result['status'])
        actual = f"\nActual Received: {result['actual_amount']} {result['to_currency']}" if result['actual_amount'] else ""
        
        text = f"""Transaction: {result['id']}
Status: {result['status'].upper()} - {status_info}

EXCHANGE: {result['from_currency']} → {result['to_currency']}

AMOUNTS:
  Send: {result['deposit_amount']} {result['from_currency']}
  Expected: {result['expected_amount']} {result['to_currency']}{actual}

ADDRESSES:
  Deposit To: {result['deposit_address']}
  Your Address: {result.get('to_address', 'N/A')}

TIMELINE:
  Created: {result['created_at']}
  Updated: {result['updated_at']}
  Provider: {result['provider']}

{'✓ Transaction complete!' if result['status'] == 'completed' else '⏳ Please wait for completion...' if result['status'] in ['waiting_deposit', 'confirming', 'exchanging', 'sending'] else '⚠ Check status carefully'}
"""
        return text, 200, {'Content-Type': 'text/plain'}
    else:
        return f"Error: {result['error']}\n", 404, {'Content-Type': 'text/plain'}


@app.route('/p', methods=['GET'])
def get_pairs():
    """Get supported currency pairs (basic)"""
    result = service.get_supported_pairs(include_networks=False)
    
    if result['success']:
        pairs = result['pairs']
        
        # Cache status info
        if result.get('cached'):
            cache_hours = result.get('cache_age', 0) / 3600
            cache_info = f" (cached {cache_hours:.1f}h ago)"
        else:
            providers = result.get('providers_queried', [])
            cache_info = f" (fresh from: {', '.join(providers)})"
        
        text = f"Supported Pairs ({result['total']} total){cache_info}:\n\n"
        text += ", ".join(pairs)
        text += f"\n\nUse /pn for network variants\nUse /i/{{from}}2{{to}} for specific pair info\n"
        
        return text, 200, {'Content-Type': 'text/plain'}
    else:
        return f"Error: {result['error']}\n", 500, {'Content-Type': 'text/plain'}


@app.route('/pn', methods=['GET'])
def get_pairs_with_networks():
    """Get supported currency pairs including network variants"""
    result = service.get_supported_pairs(include_networks=True)
    
    if result['success']:
        pairs = result['pairs']
        
        # Cache status info
        if result.get('cached'):
            cache_hours = result.get('cache_age', 0) / 3600
            cache_info = f" (cached {cache_hours:.1f}h ago)"
        else:
            providers = result.get('providers_queried', [])
            cache_info = f" (fresh from: {', '.join(providers)})"
        
        text = f"Supported Pairs with Networks ({result['total']} total){cache_info}:\n\n"
        text += ", ".join(pairs)
        text += f"\n\nUse /i/{{from}}2{{to}} for specific pair info\n"
        
        return text, 200, {'Content-Type': 'text/plain'}
    else:
        return f"Error: {result['error']}\n", 500, {'Content-Type': 'text/plain'}


@app.route('/i/<from_currency>2<to_currency>', methods=['GET'])
def get_pair_info(from_currency, to_currency):
    """Get info for specific currency pair from ALL providers"""
    result = service.get_pair_info(from_currency.upper(), to_currency.upper())
    
    if result['success']:
        text = f"{result['from_currency']} → {result['to_currency']}\n\n"
        
        if result.get('best_provider'):
            text += f"🏆 BEST: {result['best_provider']} (rate: {result['best_rate']:.6f})\n\n"
        
        # Add market rate info if available
        market_rate = result.get('market_rate')
        if market_rate:
            text += f"Market Rate: {market_rate:.6f}\n\n"
        
        text += "All Providers:\n"
        for provider, data in result['providers'].items():
            if 'status' in data:
                text += f"  {provider}: {data['status'].upper()}\n"
            elif 'error' in data:
                text += f"  {provider}: {data['error']}\n"
            else:
                selected_mark = " ← SELECTED" if data.get('selected') else ""
                fee_info = ""
                if 'effective_fee_percent' in data:
                    fee_info = f" (fee: {data['effective_fee_percent']:.2f}% - {data['fee_note']})"
                text += f"  {provider}: {data['rate']:.6f}{fee_info}{selected_mark}\n"
                if data.get('min_amount') is not None:
                    text += f"    Min: {data['min_amount']} {result['from_currency']}"
                    if data.get('max_amount'):
                        text += f", Max: {data['max_amount']} {result['from_currency']}"
                    text += "\n"
        
        text += f"\nUse /{from_currency}2{to_currency}:address:amount to create exchange\n"
        return text, 200, {'Content-Type': 'text/plain'}
    else:
        return f"Error: {result['error']}\n", 400, {'Content-Type': 'text/plain'}


@app.route('/compare/<from_currency>2<to_currency>', methods=['GET'])
@app.route('/c/<from_currency>2<to_currency>', methods=['GET'])
def compare_providers(from_currency, to_currency):
    """Compare rates across all providers for a currency pair"""
    amount = request.args.get('amount', 1.0, type=float)
    result = service.get_provider_comparison(from_currency.upper(), to_currency.upper(), amount)
    
    if result['success']:
        text = f"Rate Comparison: {result['from_currency']} → {result['to_currency']}\n"
        text += f"Amount: {result['amount']} {result['from_currency']}\n\n"
        
        # Add market rate info if available
        market_rate = result.get('market_rate')
        if market_rate:
            market_output = market_rate * result['amount']
            text += f"Market Rate: {market_rate:.6f} → {market_output:.6f} {result['to_currency']}\n\n"
        
        if result['best_provider']:
            text += f"🏆 BEST: {result['best_provider']} (rate: {result['best_rate']:.6f})\n\n"
        
        text += "All Providers:\n"
        for provider, data in result['providers'].items():
            if not data.get('active', True):
                text += f"  {provider}: DEACTIVATED\n"
            elif 'error' in data:
                text += f"  {provider}: {data['error']}\n"
            else:
                output = data['output_amount']
                fee_info = ""
                if 'effective_fee_percent' in data:
                    fee_info = f" (fee: {data['effective_fee_percent']:.2f}% - {data['fee_note']})"
                text += f"  {provider}: {data['rate']:.6f} → {output:.6f} {result['to_currency']}{fee_info}\n"
        
        text += f"\nUse /{from_currency}2{to_currency}:address:amount to create exchange\n"
        return text, 200, {'Content-Type': 'text/plain'}
    else:
        return f"Error: {result.get('error', 'Comparison failed')}\n", 400, {'Content-Type': 'text/plain'}


@app.route('/pr', methods=['GET'])
def get_providers():
    """Get provider status"""
    result = service.get_provider_status()
    
    text = "Provider Status:\n\n"
    for provider, status in result['providers'].items():
        state = "ACTIVE" if status['active'] else "DEACTIVATED"
        configured = "✓" if status['configured'] else "✗"
        text += f"  {provider}: {state} {configured}\n"
    
    text += "\n✓ = configured, ✗ = not configured\n"
    text += "Use /c/{from}2{to} to compare rates\n"
    
    return text, 200, {'Content-Type': 'text/plain'}


@app.route('/d/<tx_id>', methods=['GET'])
def delete_transaction(tx_id):
    """Delete transaction from database"""
    if service.tx_manager.delete(tx_id):
        return f"Transaction {tx_id} deleted\n", 200, {'Content-Type': 'text/plain'}
    else:
        return f"Transaction {tx_id} not found\n", 404, {'Content-Type': 'text/plain'}


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return "Error: Endpoint not found\n\nUse curl 2ex.ch/ for help\n", 404, {'Content-Type': 'text/plain'}


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return "Error: Internal server error\n", 500, {'Content-Type': 'text/plain'}


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='2ex.ch - Cryptocurrency Exchange Service')
    parser.add_argument('-p', '--port', type=int, default=5000, 
                        help='Port to run the service on (default: 5000)')
    args = parser.parse_args()
    
    try:
        # Run with threading enabled
        print(f"Starting 2ex.ch service on port {args.port}...")
        app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)
    finally:
        service.shutdown()