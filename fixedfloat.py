import json
import hmac
import hashlib
import requests
from typing import Dict, Any, Optional, List
from enum import Enum


class FixedFloatError(Exception):
    """Base exception for FixedFloat API errors."""
    pass


class OrderType(Enum):
    """Order types supported by FixedFloat."""
    FIXED = "fixed"
    FLOAT = "float"


class OrderStatus(Enum):
    """Order statuses in FixedFloat system."""
    NEW = "NEW"
    PENDING = "PENDING" 
    EXCHANGING = "EXCHANGING"
    WITHDRAWING = "WITHDRAWING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EMERGENCY = "EMERGENCY"


class EmergencyAction(Enum):
    """Emergency actions available for orders."""
    EXCHANGE = "EXCHANGE"
    REFUND = "REFUND"


class FixedFloatClient:
    """FixedFloat API client for cryptocurrency exchange operations."""
    
    BASE_URL = "https://ff.io/api/v2"
    RATES_URL = "https://ff.io/rates"
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize FixedFloat API client.
        
        Args:
            api_key: Your FixedFloat API key
            api_secret: Your FixedFloat API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json; charset=UTF-8",
            "X-API-KEY": self.api_key
        })
    
    def _generate_signature(self, data: Dict[str, Any]) -> str:
        """
        Generate HMAC SHA256 signature for API request.
        
        Args:
            data: Request data dictionary
            
        Returns:
            HMAC SHA256 signature as hex string
        """
        json_data = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            json_data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make authenticated API request.
        
        Args:
            endpoint: API endpoint path
            data: Request data
            
        Returns:
            API response data
            
        Raises:
            FixedFloatError: On API errors
        """
        url = f"{self.BASE_URL}/{endpoint}"
        signature = self._generate_signature(data)
        
        headers = {"X-API-SIGN": signature}
        
        try:
            response = self.session.post(url, json=data, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0:
                return result.get("data", {})
            else:
                raise FixedFloatError(f"API Error {result.get('code')}: {result.get('msg', 'Unknown error')}")
                
        except requests.RequestException as e:
            raise FixedFloatError(f"Request failed: {str(e)}")
        except json.JSONDecodeError:
            raise FixedFloatError("Invalid JSON response from API")
    
    def get_currencies(self) -> List[Dict[str, Any]]:
        """
        Get list of available currencies.
        
        Returns:
            List of currency dictionaries with code, network, send/receive availability
        """
        return self._make_request("ccies", {})
    
    def get_price(self, from_currency: str, to_currency: str, 
                  amount: float, direction: str = "from",
                  order_type: OrderType = OrderType.FLOAT,
                  from_network: Optional[str] = None, 
                  to_network: Optional[str] = None) -> Dict[str, Any]:
        """
        Get exchange rate for currency pair.
        
        Args:
            from_currency: Source currency code
            to_currency: Destination currency code  
            amount: Amount to exchange
            direction: "from" to send amount in fromCcy, "to" to receive amount in toCcy
            order_type: Exchange type (fixed or float)
            from_network: Source currency network (optional)
            to_network: Destination currency network (optional)
            
        Returns:
            Price information including rates and amounts
        """
        if direction not in ["from", "to"]:
            raise ValueError("direction must be 'from' or 'to'")
            
        data = {
            "fromCcy": from_currency,
            "toCcy": to_currency,
            "amount": str(amount),
            "direction": direction,
            "type": order_type.value
        }
        
        if from_network:
            data["fromNetwork"] = from_network
        if to_network:
            data["toNetwork"] = to_network
            
        return self._make_request("price", data)
    
    def create_order(self, from_currency: str, to_currency: str,
                     to_address: str, amount: float,
                     order_type: OrderType = OrderType.FLOAT,
                     from_network: Optional[str] = None,
                     to_network: Optional[str] = None,
                     extra_id: Optional[str] = None,
                     refund_address: Optional[str] = None,
                     refund_extra_id: Optional[str] = None,
                     affiliate_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create new exchange order.
        
        Args:
            from_currency: Source currency code
            to_currency: Destination currency code
            to_address: Destination wallet address
            amount: Amount to exchange
            order_type: Exchange type (fixed or float)
            from_network: Source currency network (optional)
            to_network: Destination currency network (optional)
            extra_id: Destination extra ID/memo (optional)
            refund_address: Refund address if needed (optional)
            refund_extra_id: Refund extra ID/memo (optional)
            affiliate_id: Affiliate program ID (optional)
            
        Returns:
            Order details including deposit address and order ID
        """
        data = {
            "fromCcy": from_currency,
            "toCcy": to_currency,
            "toAddress": to_address,
            "amount": str(amount),
            "type": order_type.value
        }
        
        if from_network:
            data["fromNetwork"] = from_network
        if to_network:
            data["toNetwork"] = to_network
        if extra_id:
            data["toAddressExtraId"] = extra_id
        if refund_address:
            data["refundAddress"] = refund_address
        if refund_extra_id:
            data["refundAddressExtraId"] = refund_extra_id
        if affiliate_id:
            data["afId"] = affiliate_id
            
        return self._make_request("create", data)
    
    def get_order(self, order_id: str, token: str) -> Dict[str, Any]:
        """
        Get order details and status.
        
        Args:
            order_id: Order ID
            token: Order token
            
        Returns:
            Order details including current status and transaction info
        """
        data = {
            "id": order_id,
            "token": token
        }
        
        return self._make_request("order", data)
    
    def emergency_action(self, order_id: str, token: str, 
                        action: EmergencyAction, 
                        address: Optional[str] = None,
                        extra_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle order in emergency status.
        
        Args:
            order_id: Order ID
            token: Order token
            action: Emergency action (EXCHANGE or REFUND)
            address: Address for refund (required if action is REFUND)
            extra_id: Extra ID for refund address (optional)
            
        Returns:
            Emergency action result
        """
        data = {
            "id": order_id,
            "token": token,
            "choice": action.value
        }
        
        if action == EmergencyAction.REFUND:
            if not address:
                raise ValueError("Address is required for refund action")
            data["address"] = address
            if extra_id:
                data["extraId"] = extra_id
                
        return self._make_request("emergency", data)
    
    def set_email(self, order_id: str, token: str, email: str) -> Dict[str, Any]:
        """
        Subscribe to email notifications for order.
        
        Args:
            order_id: Order ID
            token: Order token
            email: Email address for notifications
            
        Returns:
            Subscription result
        """
        data = {
            "id": order_id,
            "token": token,
            "email": email
        }
        
        return self._make_request("setEmail", data)
    
    def get_qr_code(self, order_id: str, token: str) -> Dict[str, Any]:
        """
        Generate QR code for order.
        
        Args:
            order_id: Order ID
            token: Order token
            
        Returns:
            QR code data
        """
        data = {
            "id": order_id,
            "token": token
        }
        
        return self._make_request("qr", data)
    
    def get_rates_xml(self, order_type: OrderType = OrderType.FIXED) -> str:
        """
        Get exchange rates in XML format.
        
        Args:
            order_type: Type of rates (fixed or float)
            
        Returns:
            XML string with exchange rates
        """
        url = f"{self.RATES_URL}/{order_type.value}.xml"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise FixedFloatError(f"Failed to fetch rates XML: {str(e)}")


# Convenience functions for common operations
def create_simple_exchange(api_key: str, api_secret: str,
                          from_currency: str, to_currency: str,
                          to_address: str, amount: float) -> Dict[str, Any]:
    """
    Create a simple float-type exchange order.
    
    Args:
        api_key: FixedFloat API key
        api_secret: FixedFloat API secret
        from_currency: Source currency code
        to_currency: Destination currency code
        to_address: Destination address
        amount: Amount to exchange
        
    Returns:
        Order details
    """
    client = FixedFloatClient(api_key, api_secret)
    return client.create_order(from_currency, to_currency, to_address, amount)


def check_order_status(api_key: str, api_secret: str,
                      order_id: str, token: str) -> Dict[str, Any]:
    """
    Check the status of an existing order.
    
    Args:
        api_key: FixedFloat API key
        api_secret: FixedFloat API secret
        order_id: Order ID
        token: Order token
        
    Returns:
        Order status details
    """
    client = FixedFloatClient(api_key, api_secret)
    return client.get_order(order_id, token)