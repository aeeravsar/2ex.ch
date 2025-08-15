import json
import requests
from typing import Dict, Any, Optional, List
from enum import Enum


class ExolixError(Exception):
    """Base exception for Exolix API errors."""
    pass


class RateType(Enum):
    """Rate types supported by Exolix."""
    FIXED = "fixed"
    FLOAT = "float"


class TransactionStatus(Enum):
    """Transaction statuses in Exolix system."""
    WAIT = "wait"
    CONFIRMATION = "confirmation"
    CONFIRMED = "confirmed"
    EXCHANGING = "exchanging"
    SENDING = "sending"
    SUCCESS = "success"
    OVERDUE = "overdue"
    REFUNDED = "refunded"


class ExolixClient:
    """Exolix API client for cryptocurrency exchange operations."""
    
    BASE_URL = "https://exolix.com/api/v2"
    
    def __init__(self, api_key: str):
        """
        Initialize Exolix API client.
        
        Args:
            api_key: Your Exolix API key
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": self.api_key
        })
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make authenticated API request.
        
        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint path
            params: URL parameters for GET requests
            data: Request data for POST requests
            
        Returns:
            API response data
            
        Raises:
            ExolixError: On API errors
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
            
            # Handle both JSON and plain responses
            try:
                result = response.json()
                return result
            except json.JSONDecodeError:
                # Some endpoints might return plain text or other formats
                return {"response": response.text}
                
        except requests.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get('message', error_data.get('error', str(e)))
                    raise ExolixError(f"API Error: {error_msg}")
                except json.JSONDecodeError:
                    raise ExolixError(f"HTTP {e.response.status_code}: {e.response.text}")
            else:
                raise ExolixError(f"Request failed: {str(e)}")
    
    def get_currencies(self) -> List[Dict[str, Any]]:
        """
        Get list of available currencies.
        
        Returns:
            List of currency dictionaries
        """
        return self._make_request("GET", "currencies")
    
    def get_currency_networks(self, currency_code: str) -> List[Dict[str, Any]]:
        """
        Get networks for a specific currency.
        
        Args:
            currency_code: Currency code (e.g., 'BTC', 'ETH')
            
        Returns:
            List of network dictionaries for the currency
        """
        return self._make_request("GET", f"currencies/{currency_code}/networks")
    
    def get_all_networks(self) -> List[Dict[str, Any]]:
        """
        Get list of all available networks.
        
        Returns:
            List of all network dictionaries
        """
        return self._make_request("GET", "currencies/networks")
    
    def get_rate(self, coin_from: str, coin_to: str, amount: float,
                 network_from: Optional[str] = None, network_to: Optional[str] = None,
                 rate_type: RateType = RateType.FIXED) -> Dict[str, Any]:
        """
        Get exchange rate for currency pair.
        
        Args:
            coin_from: Source currency code
            coin_to: Destination currency code
            amount: Amount to exchange
            network_from: Source network (optional)
            network_to: Destination network (optional)
            rate_type: Rate type (fixed or float)
            
        Returns:
            Rate information including exchange amounts and fees
        """
        params = {
            "coinFrom": coin_from,
            "coinTo": coin_to,
            "amount": str(amount),
            "rateType": rate_type.value
        }
        
        if network_from:
            params["networkFrom"] = network_from
        if network_to:
            params["networkTo"] = network_to
            
        return self._make_request("GET", "rate", params=params)
    
    def create_transaction(self, coin_from: str, coin_to: str, withdrawal_address: str,
                          amount: Optional[float] = None, network_from: str = None,
                          network_to: str = None,
                          withdrawal_amount: Optional[float] = None,
                          rate_type: RateType = RateType.FIXED,
                          withdrawal_extra_id: Optional[str] = None,
                          refund_address: Optional[str] = None,
                          refund_extra_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create new exchange transaction.
        
        Args:
            coin_from: Source currency code
            coin_to: Destination currency code
            withdrawal_address: Destination wallet address
            amount: Amount to exchange
            network_from: Source network (optional)
            network_to: Destination network (optional)
            rate_type: Rate type (fixed or float)
            withdrawal_extra_id: Destination extra ID/memo (optional)
            refund_address: Refund address if needed (optional)
            refund_extra_id: Refund extra ID/memo (optional)
            
        Returns:
            Transaction details including deposit address and transaction ID
        """
        # Networks are REQUIRED - use appropriate network for each coin
        if not network_from:
            if coin_from == 'USDT':
                network_from = 'ETH'  # USDT uses Ethereum network
            else:
                network_from = coin_from
        if not network_to:
            if coin_to == 'USDT':
                network_to = 'ETH'  # USDT uses Ethereum network  
            else:
                network_to = coin_to
            
        data = {
            "coinFrom": coin_from,
            "coinTo": coin_to,
            "networkFrom": network_from,
            "networkTo": network_to,
            "withdrawalAddress": withdrawal_address,
            "rateType": rate_type.value
        }
        
        # Either amount or withdrawalAmount must be specified
        if withdrawal_amount is not None:
            data["withdrawalAmount"] = str(withdrawal_amount)
        elif amount is not None:
            data["amount"] = str(amount)
        else:
            raise ValueError("Either amount or withdrawalAmount must be specified")
        
        if withdrawal_extra_id:
            data["withdrawalExtraId"] = withdrawal_extra_id
        if refund_address:
            data["refundAddress"] = refund_address
        if refund_extra_id:
            data["refundExtraId"] = refund_extra_id
            
        return self._make_request("POST", "transactions", data=data)
    
    def get_transactions(self, limit: Optional[int] = None, 
                        offset: Optional[int] = None) -> Dict[str, Any]:
        """
        Get transaction history.
        
        Args:
            limit: Number of transactions to return (optional)
            offset: Number of transactions to skip (optional)
            
        Returns:
            Paginated list of transactions
        """
        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
            
        return self._make_request("GET", "transactions", params=params)
    
    def get_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """
        Get specific transaction details.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            Transaction details including current status
        """
        return self._make_request("GET", f"transactions/{transaction_id}")
    
    def get_transaction_status(self, transaction_id: str) -> str:
        """
        Get transaction status.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            Transaction status string
        """
        transaction = self.get_transaction(transaction_id)
        return transaction.get("status", "unknown")
    
    def is_transaction_complete(self, transaction_id: str) -> bool:
        """
        Check if transaction is complete.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            True if transaction is successful, False otherwise
        """
        status = self.get_transaction_status(transaction_id)
        return status == TransactionStatus.SUCCESS.value


# Convenience functions for common operations
def create_simple_exchange(api_key: str, coin_from: str, coin_to: str,
                          withdrawal_address: str, amount: float) -> Dict[str, Any]:
    """
    Create a simple exchange transaction.
    
    Args:
        api_key: Exolix API key
        coin_from: Source currency code
        coin_to: Destination currency code
        withdrawal_address: Destination address
        amount: Amount to exchange
        
    Returns:
        Transaction details
    """
    client = ExolixClient(api_key)
    return client.create_transaction(coin_from, coin_to, withdrawal_address, amount)


def check_exchange_rate(api_key: str, coin_from: str, coin_to: str,
                       amount: float) -> Dict[str, Any]:
    """
    Check exchange rate for a currency pair.
    
    Args:
        api_key: Exolix API key
        coin_from: Source currency code
        coin_to: Destination currency code
        amount: Amount to exchange
        
    Returns:
        Rate information
    """
    client = ExolixClient(api_key)
    return client.get_rate(coin_from, coin_to, amount)


def get_supported_currencies(api_key: str) -> List[Dict[str, Any]]:
    """
    Get list of supported currencies.
    
    Args:
        api_key: Exolix API key
        
    Returns:
        List of supported currencies
    """
    client = ExolixClient(api_key)
    return client.get_currencies()


class ExolixTwoExService:
    """2ex.ch style service using Exolix API."""
    
    def __init__(self, api_key: str):
        """
        Initialize service with Exolix API key.
        
        Args:
            api_key: Exolix API key
        """
        self.client = ExolixClient(api_key)
    
    def parse_exchange_request(self, request_string: str) -> dict:
        """
        Parse 2ex.ch style request: ltc2xmr4{monero_address}
        
        Args:
            request_string: Format like "ltc2xmr4ADDRESS"
            
        Returns:
            Dictionary with parsed components
        """
        # Find the '4' separator for address
        addr_sep = request_string.find('4')
        if addr_sep == -1:
            raise ValueError("Invalid request format - missing address separator '4'")
        
        # Extract address
        address = request_string[addr_sep + 1:]
        if not address:
            raise ValueError("No destination address provided")
        
        # Parse currency pair from the part before '4'
        pair_part = request_string[:addr_sep]
        currency_sep = pair_part.find('2')
        if currency_sep == -1:
            raise ValueError("Invalid request format - missing currency separator '2'")
        
        from_currency = pair_part[:currency_sep].upper()
        to_currency = pair_part[currency_sep + 1:].upper()
        
        if not from_currency or not to_currency:
            raise ValueError("Invalid currency codes")
        
        return {
            'from_currency': from_currency,
            'to_currency': to_currency,
            'to_address': address
        }
    
    def create_exchange(self, request_string: str, amount: float = None) -> dict:
        """
        Create exchange order from 2ex.ch style request.
        
        Args:
            request_string: Format like "ltc2xmr4ADDRESS"  
            amount: Amount to exchange (will use minimum if not specified)
            
        Returns:
            Order details with deposit address
        """
        try:
            # Parse the request
            parsed = self.parse_exchange_request(request_string)
            
            # Get rate info to determine amount if not specified
            if amount is None:
                rate_info = self.client.get_rate(
                    parsed['from_currency'], 
                    parsed['to_currency'], 
                    1.0
                )
                # Use a default small amount if no minimum specified
                amount = 0.001
            
            # Create the transaction
            transaction = self.client.create_transaction(
                coin_from=parsed['from_currency'],
                coin_to=parsed['to_currency'],
                withdrawal_address=parsed['to_address'],
                amount=amount
            )
            
            return {
                'success': True,
                'transaction_id': transaction.get('id'),
                'deposit_address': transaction.get('depositAddress'),
                'deposit_amount': transaction.get('amount'),
                'expected_amount': transaction.get('withdrawalAmount'),
                'from_currency': parsed['from_currency'],
                'to_currency': parsed['to_currency'],
                'to_address': parsed['to_address'],
                'status': transaction.get('status')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def check_transaction_status(self, transaction_id: str) -> dict:
        """Check status of existing transaction."""
        try:
            transaction = self.client.get_transaction(transaction_id)
            return {
                'success': True,
                'status': transaction.get('status'),
                'deposit_address': transaction.get('depositAddress'),
                'withdrawal_address': transaction.get('withdrawalAddress'),
                'amount': transaction.get('amount'),
                'withdrawal_amount': transaction.get('withdrawalAmount'),
                'tx_from': transaction.get('hashIn'),
                'tx_to': transaction.get('hashOut')
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
                    if from_curr['code'] != to_curr['code']:
                        pairs.append(f"{from_curr['code']}2{to_curr['code']}")
            
            return pairs
        except ExolixError as e:
            raise Exception(f"Failed to get supported pairs: {e}")
    
    def get_current_rates(self) -> dict:
        """Get current exchange rates for common pairs."""
        common_pairs = [
            ('BTC', 'ETH'), ('BTC', 'LTC'), ('BTC', 'XMR'),
            ('ETH', 'BTC'), ('ETH', 'LTC'), ('LTC', 'BTC'),
            ('LTC', 'XMR'), ('XMR', 'BTC')
        ]
        
        rates = {}
        for from_curr, to_curr in common_pairs:
            try:
                rate_info = self.client.get_rate(from_curr, to_curr, 1.0)
                rates[f"{from_curr}2{to_curr}"] = {
                    'rate': rate_info.get('toAmount', 0),
                    'min': rate_info.get('minAmount', 0),
                    'max': rate_info.get('maxAmount', 0)
                }
            except:
                continue  # Skip pairs that aren't available
        
        return rates