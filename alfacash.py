import json
import requests
from typing import Dict, Any, Optional, List
from enum import Enum
from crypto_mapper import symbol_to_id


class AlfacashError(Exception):
    """Base exception for Alfacash API errors."""
    pass


class TransactionStatus(Enum):
    """Transaction statuses in Alfacash system."""
    WAITING = "waiting"
    RECEIVED = "received"
    CONFIRMED = "confirmed"
    EXCHANGING = "exchanging"
    WITHDRAWAL = "withdrawal"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REFUNDED = "refunded"


class AlfacashClient:
    """Alfacash API client for cryptocurrency exchange operations."""
    
    BASE_URL = "https://www.alfa.cash/api"
    
    def __init__(self):
        """
        Initialize Alfacash API client.
        
        Note: Alfacash doesn't require API keys for basic operations
        """
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "2ex.ch/1.0"
        })
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make API request.
        
        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint path
            params: URL parameters for GET requests
            data: Request data for POST requests
            
        Returns:
            API response data
            
        Raises:
            AlfacashError: On API errors
        """
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            # Handle JSON responses
            try:
                result = response.json()
                
                # Check for API errors
                if isinstance(result, dict) and "error" in result:
                    raise AlfacashError(f"API Error: {result['error']}")
                    
                return result
            except json.JSONDecodeError:
                # Some endpoints might return plain text
                return {"response": response.text}
                
        except requests.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get('error', str(e))
                    raise AlfacashError(f"API Error: {error_msg}")
                except json.JSONDecodeError:
                    raise AlfacashError(f"HTTP {e.response.status_code}: {e.response.text}")
            else:
                raise AlfacashError(f"Request failed: {str(e)}")
    
    def get_currencies(self) -> List[str]:
        """
        Get list of available currencies.
        
        Returns:
            List of currency codes
        """
        # Based on Alfacash documentation, they support these currencies
        # Including network-specific variants for USDT
        return [
            'BTC', 'ETH', 'LTC', 'XMR', 'BCH', 'XRP', 'ADA', 'DOT', 'LINK',
            'USDT', 'USDTETH', 'USDTTRC',  # USDT variants
            'USDC', 'DAI', 'BNB', 'MATIC', 'AVAX', 'SOL', 'DOGE',
            'TRX', 'UNI', 'AAVE', 'SUSHI', 'YFI', 'MKR', 'SNX', 'COMP'
        ]
    
    def _code_to_name(self, currency_code: str) -> str:
        """Convert currency code to Alfacash gate name"""
        # Parse currency and network if using colon format
        currency_upper = currency_code.upper()
        currency = currency_upper
        network = None
        
        # Check for colon separator (new format: "USDT:ETH")
        if ':' in currency_upper:
            parts = currency_upper.split(':')
            currency = parts[0]
            network = parts[1]
        
        # Special handling for Alfacash-specific naming
        alfacash_mapping = {
            'USDT': 'tethererc20',     # Default USDT to ERC-20
            'USDTETH': 'tethererc20',  # ERC-20 USDT (legacy format)
            'USDTTRC': 'trc20usdt',    # TRC-20 USDT (legacy format)
            'USDTTRON': 'trc20usdt',   # Alternative TRC-20 name
            'USDTBSC': 'tetherbsc',    # BSC USDT
        }
        
        # Handle new colon format with networks
        if currency == 'USDT' and network:
            if network in ['ETH', 'ERC20', 'ETHEREUM']:
                return 'tethererc20'
            elif network in ['TRC', 'TRC20', 'TRON']:
                return 'trc20usdt'
            elif network in ['BSC', 'BEP20', 'BINANCE']:
                return 'tetherbsc'
        
        # Check Alfacash-specific mappings first
        if currency_upper in alfacash_mapping:
            return alfacash_mapping[currency_upper]
        
        # For other currencies, try CoinGecko ID (works for most)
        coin_id = symbol_to_id(currency)
        return coin_id if coin_id else currency.lower()
    
    def get_rate(self, gate_deposit: str, gate_withdrawal: str, 
                 amount: Optional[float] = None) -> Dict[str, Any]:
        """
        Get exchange rate for currency pair.
        
        Args:
            gate_deposit: Source currency code (e.g., BTC)
            gate_withdrawal: Destination currency code (e.g., ETH)
            amount: Amount to exchange (optional)
            
        Returns:
            Rate information including min/max amounts
        """
        # Convert symbols to Alfacash names (CoinGecko IDs)
        deposit_name = self._code_to_name(gate_deposit)
        withdrawal_name = self._code_to_name(gate_withdrawal)
        
        data = {
            "gate_deposit": deposit_name,
            "gate_withdrawal": withdrawal_name
        }
        
        if amount is not None:
            data["deposit_amount"] = str(amount)
            
        return self._make_request("POST", "rate.json", data=data)
    
    def create_transaction(self, gate_deposit: str, gate_withdrawal: str,
                          withdrawal_address: str, email: str,
                          deposit_amount: Optional[float] = None,
                          withdrawal_amount: Optional[float] = None,
                          withdrawal_extra: Optional[str] = None,
                          refund_address: Optional[str] = None,
                          refund_extra: Optional[str] = None) -> Dict[str, Any]:
        """
        Create new exchange transaction.
        
        Args:
            gate_deposit: Source currency code
            gate_withdrawal: Destination currency code
            withdrawal_address: Destination wallet address
            email: Contact email (required by Alfacash)
            deposit_amount: Amount to send (optional)
            withdrawal_amount: Amount to receive (optional)
            withdrawal_extra: Destination extra ID/memo (optional)
            refund_address: Refund address if needed (optional)
            refund_extra: Refund extra ID/memo (optional)
            
        Returns:
            Transaction details including deposit address and secret key
        """
        # Convert symbols to Alfacash names (CoinGecko IDs)
        deposit_name = self._code_to_name(gate_deposit)
        withdrawal_name = self._code_to_name(gate_withdrawal)
        
        data = {
            "gate_deposit": deposit_name,
            "gate_withdrawal": withdrawal_name,
            "email": email,
            "options": {
                "address": withdrawal_address
            }
        }
        
        # Either deposit_amount or withdrawal_amount should be specified
        if deposit_amount is not None:
            data["deposit_amount"] = str(deposit_amount)
        elif withdrawal_amount is not None:
            data["withdrawal_amount"] = str(withdrawal_amount)
        else:
            # If no amount specified, get minimum from rate
            try:
                rate_info = self.get_rate(gate_deposit, gate_withdrawal)
                min_amount = rate_info.get('min_deposit_amount', 0.001)
                data["deposit_amount"] = str(min_amount)
            except:
                data["deposit_amount"] = "0.001"  # Fallback minimum
        
        if withdrawal_extra:
            data["options"]["extra"] = withdrawal_extra
        if refund_address:
            data["refund_address"] = refund_address
        if refund_extra:
            data["refund_extra"] = refund_extra
            
        return self._make_request("POST", "create.json", data=data)
    
    def get_transaction(self, secret_key: str) -> Dict[str, Any]:
        """
        Get specific transaction details.
        
        Args:
            secret_key: Transaction secret key from creation
            
        Returns:
            Transaction details including current status
        """
        return self._make_request("GET", f"status/{secret_key}")
    
    def get_transaction_status(self, secret_key: str) -> str:
        """
        Get transaction status.
        
        Args:
            secret_key: Transaction secret key
            
        Returns:
            Transaction status string
        """
        transaction = self.get_transaction(secret_key)
        return transaction.get("status", "unknown")
    
    def is_transaction_complete(self, secret_key: str) -> bool:
        """
        Check if transaction is complete.
        
        Args:
            secret_key: Transaction secret key
            
        Returns:
            True if transaction is completed, False otherwise
        """
        status = self.get_transaction_status(secret_key)
        return status == TransactionStatus.COMPLETED.value


# Convenience functions for common operations
def create_simple_exchange(gate_deposit: str, gate_withdrawal: str,
                          withdrawal_address: str, email: str,
                          amount: float) -> Dict[str, Any]:
    """
    Create a simple exchange transaction.
    
    Args:
        gate_deposit: Source currency code
        gate_withdrawal: Destination currency code
        withdrawal_address: Destination address
        email: Contact email
        amount: Amount to exchange
        
    Returns:
        Transaction details
    """
    client = AlfacashClient()
    return client.create_transaction(gate_deposit, gate_withdrawal, 
                                   withdrawal_address, email, 
                                   deposit_amount=amount)


def check_exchange_rate(gate_deposit: str, gate_withdrawal: str,
                       amount: float) -> Dict[str, Any]:
    """
    Check exchange rate for a currency pair.
    
    Args:
        gate_deposit: Source currency code
        gate_withdrawal: Destination currency code
        amount: Amount to exchange
        
    Returns:
        Rate information
    """
    client = AlfacashClient()
    return client.get_rate(gate_deposit, gate_withdrawal, amount)


def get_supported_currencies() -> List[str]:
    """
    Get list of supported currencies.
    
    Returns:
        List of supported currencies
    """
    client = AlfacashClient()
    return client.get_currencies()


class AlfacashTwoExService:
    """2ex.ch style service using Alfacash API."""
    
    def __init__(self, default_email: str = "noreply@2ex.ch"):
        """
        Initialize service with Alfacash client.
        
        Args:
            default_email: Default email for transactions
        """
        self.client = AlfacashClient()
        self.default_email = default_email
    
    def parse_exchange_request(self, request_string: str) -> dict:
        """
        Parse 2ex.ch style request: ltc2xmr:address
        
        Args:
            request_string: Format like "ltc2xmr:ADDRESS"
            
        Returns:
            Dictionary with parsed components
        """
        # Split by colons to get parts
        parts = request_string.split(':')
        
        if len(parts) < 2:
            raise ValueError("Invalid format - use: {from}2{to}:{address}")
        
        # First part is currency pair
        pair_part = parts[0]
        
        # Second part is address
        address = parts[1]
        if not address:
            raise ValueError("No destination address provided")
        
        # Parse currency pair
        currency_sep = pair_part.find('2')
        if currency_sep == -1:
            raise ValueError("Invalid format - use: {from}2{to}:{address}")
        
        from_currency = pair_part[:currency_sep].upper()
        to_currency = pair_part[currency_sep + 1:].upper()
        
        if not from_currency or not to_currency:
            raise ValueError("Invalid currency codes")
        
        return {
            'from_currency': from_currency,
            'to_currency': to_currency,
            'to_address': address
        }
    
    def create_exchange(self, request_string: str, amount: float = None,
                       email: str = None) -> dict:
        """
        Create exchange order from 2ex.ch style request.
        
        Args:
            request_string: Format like "ltc2xmr:ADDRESS"  
            amount: Amount to exchange (will use minimum if not specified)
            email: Email address (uses default if not specified)
            
        Returns:
            Order details with deposit address
        """
        try:
            # Parse the request
            parsed = self.parse_exchange_request(request_string)
            
            # Use default email if not provided
            if email is None:
                email = self.default_email
            
            # Get rate info to determine amount if not specified
            if amount is None:
                try:
                    rate_info = self.client.get_rate(
                        parsed['from_currency'], 
                        parsed['to_currency']
                    )
                    # Use minimum amount from rate info
                    amount = float(rate_info.get('min_deposit_amount', 0.001))
                except AlfacashError:
                    amount = 0.001  # Fallback minimum
            
            # Create the transaction
            transaction = self.client.create_transaction(
                gate_deposit=parsed['from_currency'],
                gate_withdrawal=parsed['to_currency'],
                withdrawal_address=parsed['to_address'],
                email=email,
                deposit_amount=amount
            )
            
            return {
                'success': True,
                'transaction_id': transaction.get('id'),
                'secret_key': transaction.get('secret_key'),
                'deposit_address': transaction.get('deposit_address'),
                'deposit_amount': transaction.get('deposit_amount'),
                'expected_amount': transaction.get('withdrawal_amount'),
                'from_currency': parsed['from_currency'],
                'to_currency': parsed['to_currency'],
                'to_address': parsed['to_address'],
                'status': transaction.get('status', 'created'),
                'expires_at': transaction.get('expires_at'),
                'rate': transaction.get('rate')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def check_transaction_status(self, secret_key: str) -> dict:
        """Check status of existing transaction."""
        try:
            transaction = self.client.get_transaction(secret_key)
            return {
                'success': True,
                'status': transaction.get('status'),
                'deposit_address': transaction.get('deposit_address'),
                'withdrawal_address': transaction.get('withdrawal_address'),
                'deposit_amount': transaction.get('deposit_amount'),
                'withdrawal_amount': transaction.get('withdrawal_amount'),
                'received_amount': transaction.get('received_amount'),
                'remaining_time': transaction.get('remaining_time'),
                'tx_from': transaction.get('tx_from'),
                'tx_to': transaction.get('tx_to')
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_supported_pairs(self) -> list:
        """Get list of supported currency pairs."""
        try:
            currencies = self.client.get_currencies()
            pairs = []
            
            # Create pairs from available currencies
            for from_curr in currencies:
                for to_curr in currencies:
                    if from_curr != to_curr:
                        pairs.append(f"{from_curr}2{to_curr}")
            
            return pairs
        except AlfacashError as e:
            raise Exception(f"Failed to get supported pairs: {e}")
    
    def get_current_rates(self) -> dict:
        """Get current exchange rates for common pairs."""
        common_pairs = [
            ('BTC', 'ETH'), ('BTC', 'LTC'), ('BTC', 'XMR'),
            ('ETH', 'BTC'), ('ETH', 'LTC'), ('LTC', 'BTC'),
            ('LTC', 'XMR'), ('XMR', 'BTC'), ('USDT', 'BTC'),
            ('BTC', 'USDT'), ('ETH', 'USDT'), ('USDT', 'ETH')
        ]
        
        rates = {}
        for from_curr, to_curr in common_pairs:
            try:
                rate_info = self.client.get_rate(from_curr, to_curr)
                rates[f"{from_curr}2{to_curr}"] = {
                    'rate': rate_info.get('rate', 0),
                    'min': rate_info.get('min_deposit_amount', 0),
                    'max': rate_info.get('max_deposit_amount', 0)
                }
            except:
                continue  # Skip pairs that aren't available
        
        return rates