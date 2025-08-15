import json
import time
import requests
import os
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class CryptoMapper:
    """
    Cryptocurrency symbol/name mapper with 24-hour caching
    Uses CoinGecko API to get comprehensive symbol->name->id mappings
    """
    
    def __init__(self, cache_file: str = "crypto_mapping_cache.json"):
        self.cache_file = cache_file
        self.cache_duration = 24 * 60 * 60  # 24 hours in seconds
        self.coingecko_url = "https://api.coingecko.com/api/v3/coins/list"
        
        # Load mappings
        self._load_mappings()
    
    def _load_cache(self) -> Optional[Dict]:
        """Load mappings from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # Check if cache is still valid
                cache_time = cache_data.get('timestamp', 0)
                if time.time() - cache_time < self.cache_duration:
                    logger.info("Using cached crypto mappings")
                    return cache_data['mappings']
        except Exception as e:
            logger.warning(f"Failed to load crypto mapping cache: {e}")
        return None
    
    def _save_cache(self, mappings: Dict):
        """Save mappings to cache file"""
        try:
            cache_data = {
                'timestamp': time.time(),
                'mappings': mappings
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)
            logger.info("Crypto mappings cached successfully")
        except Exception as e:
            logger.warning(f"Failed to save crypto mapping cache: {e}")
    
    def _fetch_from_coingecko(self) -> Dict:
        """Fetch fresh mappings from CoinGecko API"""
        logger.info("Fetching fresh crypto mappings from CoinGecko...")
        
        try:
            response = requests.get(self.coingecko_url, timeout=30)
            response.raise_for_status()
            coins_data = response.json()
            
            # Build comprehensive mappings
            mappings = {
                'symbol_to_id': {},      # BTC -> bitcoin
                'symbol_to_name': {},    # BTC -> Bitcoin  
                'name_to_id': {},        # bitcoin -> bitcoin
                'id_to_symbol': {},      # bitcoin -> BTC
                'id_to_name': {},        # bitcoin -> Bitcoin
                'all_symbols': set(),    # All available symbols
                'all_names': set(),      # All available names
                'all_ids': set()         # All available IDs
            }
            
            # Priority coins to handle duplicates properly
            priority_ids = {
                'bitcoin', 'ethereum', 'litecoin', 'monero', 'bitcoin-cash',
                'ripple', 'cardano', 'polkadot', 'chainlink', 'tether',
                'usd-coin', 'dai', 'binancecoin', 'matic-network', 'avalanche-2',
                'solana', 'dogecoin', 'tron', 'uniswap', 'aave'
            }
            
            # First pass: Process priority coins
            for coin in coins_data:
                coin_id = coin.get('id', '').lower()
                symbol = coin.get('symbol', '').upper()
                name = coin.get('name', '')
                
                if not coin_id or not symbol or coin_id not in priority_ids:
                    continue
                
                mappings['symbol_to_id'][symbol] = coin_id
                mappings['symbol_to_name'][symbol] = name
                mappings['name_to_id'][name.lower()] = coin_id
                mappings['id_to_symbol'][coin_id] = symbol
                mappings['id_to_name'][coin_id] = name
                
                mappings['all_symbols'].add(symbol)
                mappings['all_names'].add(name.lower())
                mappings['all_ids'].add(coin_id)
            
            # Second pass: Process remaining coins, avoiding symbol conflicts
            for coin in coins_data:
                coin_id = coin.get('id', '').lower()
                symbol = coin.get('symbol', '').upper()
                name = coin.get('name', '')
                
                if not coin_id or not symbol or coin_id in priority_ids:
                    continue
                
                # Only add if symbol not already taken by priority coin
                if symbol not in mappings['symbol_to_id']:
                    mappings['symbol_to_id'][symbol] = coin_id
                    mappings['symbol_to_name'][symbol] = name
                
                mappings['name_to_id'][name.lower()] = coin_id
                mappings['id_to_symbol'][coin_id] = symbol
                mappings['id_to_name'][coin_id] = name
                
                mappings['all_symbols'].add(symbol)
                mappings['all_names'].add(name.lower())
                mappings['all_ids'].add(coin_id)
            
            # Convert sets to lists for JSON serialization
            mappings['all_symbols'] = sorted(list(mappings['all_symbols']))
            mappings['all_names'] = sorted(list(mappings['all_names']))
            mappings['all_ids'] = sorted(list(mappings['all_ids']))
            
            logger.info(f"Fetched mappings for {len(mappings['all_symbols'])} symbols")
            return mappings
            
        except Exception as e:
            logger.error(f"Failed to fetch crypto mappings from CoinGecko: {e}")
            return self._get_fallback_mappings()
    
    def _get_fallback_mappings(self) -> Dict:
        """Fallback mappings for common cryptocurrencies"""
        logger.info("Using fallback crypto mappings")
        
        fallback_data = {
            'BTC': {'id': 'bitcoin', 'name': 'Bitcoin'},
            'ETH': {'id': 'ethereum', 'name': 'Ethereum'},
            'LTC': {'id': 'litecoin', 'name': 'Litecoin'},
            'XMR': {'id': 'monero', 'name': 'Monero'},
            'BCH': {'id': 'bitcoin-cash', 'name': 'Bitcoin Cash'},
            'XRP': {'id': 'ripple', 'name': 'XRP'},
            'ADA': {'id': 'cardano', 'name': 'Cardano'},
            'DOT': {'id': 'polkadot', 'name': 'Polkadot'},
            'LINK': {'id': 'chainlink', 'name': 'Chainlink'},
            'USDT': {'id': 'tether', 'name': 'Tether'},
            'USDC': {'id': 'usd-coin', 'name': 'USD Coin'},
            'DAI': {'id': 'dai', 'name': 'Dai'},
            'BNB': {'id': 'binancecoin', 'name': 'BNB'},
            'MATIC': {'id': 'matic-network', 'name': 'Polygon'},
            'AVAX': {'id': 'avalanche-2', 'name': 'Avalanche'},
            'SOL': {'id': 'solana', 'name': 'Solana'},
            'DOGE': {'id': 'dogecoin', 'name': 'Dogecoin'},
            'TRX': {'id': 'tron', 'name': 'TRON'},
            'UNI': {'id': 'uniswap', 'name': 'Uniswap'},
            'AAVE': {'id': 'aave', 'name': 'Aave'},
            'SUSHI': {'id': 'sushi', 'name': 'SushiSwap'},
            'YFI': {'id': 'yearn-finance', 'name': 'yearn.finance'},
            'MKR': {'id': 'maker', 'name': 'Maker'},
            'SNX': {'id': 'havven', 'name': 'Synthetix'},
            'COMP': {'id': 'compound-governance-token', 'name': 'Compound'}
        }
        
        mappings = {
            'symbol_to_id': {},
            'symbol_to_name': {},
            'name_to_id': {},
            'id_to_symbol': {},
            'id_to_name': {},
            'all_symbols': [],
            'all_names': [],
            'all_ids': []
        }
        
        for symbol, data in fallback_data.items():
            coin_id = data['id']
            name = data['name']
            
            mappings['symbol_to_id'][symbol] = coin_id
            mappings['symbol_to_name'][symbol] = name
            mappings['name_to_id'][name.lower()] = coin_id
            mappings['id_to_symbol'][coin_id] = symbol
            mappings['id_to_name'][coin_id] = name
        
        mappings['all_symbols'] = sorted(fallback_data.keys())
        mappings['all_names'] = sorted([data['name'].lower() for data in fallback_data.values()])
        mappings['all_ids'] = sorted([data['id'] for data in fallback_data.values()])
        
        return mappings
    
    def _load_mappings(self):
        """Load or fetch cryptocurrency mappings"""
        # Try cache first
        cached_mappings = self._load_cache()
        if cached_mappings:
            self.mappings = cached_mappings
            return
        
        # Fetch fresh data
        self.mappings = self._fetch_from_coingecko()
        self._save_cache(self.mappings)
    
    def symbol_to_id(self, symbol: str) -> Optional[str]:
        """Convert symbol (BTC) to CoinGecko ID (bitcoin)"""
        return self.mappings['symbol_to_id'].get(symbol.upper())
    
    def symbol_to_name(self, symbol: str) -> Optional[str]:
        """Convert symbol (BTC) to full name (Bitcoin)"""
        return self.mappings['symbol_to_name'].get(symbol.upper())
    
    def name_to_id(self, name: str) -> Optional[str]:
        """Convert name (Bitcoin) to CoinGecko ID (bitcoin)"""
        return self.mappings['name_to_id'].get(name.lower())
    
    def id_to_symbol(self, coin_id: str) -> Optional[str]:
        """Convert CoinGecko ID (bitcoin) to symbol (BTC)"""
        return self.mappings['id_to_symbol'].get(coin_id.lower())
    
    def id_to_name(self, coin_id: str) -> Optional[str]:
        """Convert CoinGecko ID (bitcoin) to name (Bitcoin)"""
        return self.mappings['id_to_name'].get(coin_id.lower())
    
    def get_all_symbols(self) -> List[str]:
        """Get all available cryptocurrency symbols"""
        return self.mappings['all_symbols']
    
    def get_all_names(self) -> List[str]:
        """Get all available cryptocurrency names"""
        return self.mappings['all_names']
    
    def get_all_ids(self) -> List[str]:
        """Get all available CoinGecko IDs"""
        return self.mappings['all_ids']
    
    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid"""
        return symbol.upper() in self.mappings['symbol_to_id']
    
    def is_valid_name(self, name: str) -> bool:
        """Check if name is valid"""
        return name.lower() in self.mappings['name_to_id']
    
    def is_valid_id(self, coin_id: str) -> bool:
        """Check if CoinGecko ID is valid"""
        return coin_id.lower() in self.mappings['id_to_symbol']
    
    def get_info(self, identifier: str) -> Dict[str, str]:
        """
        Get all info for a cryptocurrency by any identifier
        Returns dict with symbol, name, id
        """
        identifier = identifier.strip()
        
        # Try as symbol first
        coin_id = self.symbol_to_id(identifier)
        if coin_id:
            return {
                'symbol': identifier.upper(),
                'name': self.id_to_name(coin_id),
                'id': coin_id
            }
        
        # Try as name
        coin_id = self.name_to_id(identifier)
        if coin_id:
            return {
                'symbol': self.id_to_symbol(coin_id),
                'name': identifier.title(),
                'id': coin_id
            }
        
        # Try as ID
        if self.is_valid_id(identifier):
            return {
                'symbol': self.id_to_symbol(identifier),
                'name': self.id_to_name(identifier),
                'id': identifier.lower()
            }
        
        return {}
    
    def invalidate_cache(self):
        """Force refresh of mappings on next use"""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                logger.info("Crypto mapping cache invalidated")
        except Exception as e:
            logger.warning(f"Failed to invalidate crypto mapping cache: {e}")
    
    def get_cache_info(self) -> Dict:
        """Get information about current cache"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    cache_age = time.time() - cache_data.get('timestamp', 0)
                    return {
                        'cached': True,
                        'age_hours': cache_age / 3600,
                        'total_symbols': len(self.mappings.get('all_symbols', [])),
                        'expires_in_hours': (self.cache_duration - cache_age) / 3600
                    }
        except:
            pass
        
        return {
            'cached': False,
            'total_symbols': len(self.mappings.get('all_symbols', [])),
        }


# Global instance for easy importing
crypto_mapper = CryptoMapper()

# Convenience functions
def symbol_to_name(symbol: str) -> str:
    """Convert BTC -> Bitcoin"""
    name = crypto_mapper.symbol_to_name(symbol)
    return name if name else symbol.lower()

def symbol_to_id(symbol: str) -> str:
    """Convert BTC -> bitcoin"""
    coin_id = crypto_mapper.symbol_to_id(symbol)
    return coin_id if coin_id else symbol.lower()

def name_to_symbol(name: str) -> str:
    """Convert Bitcoin -> BTC"""
    # First try to find by name
    coin_id = crypto_mapper.name_to_id(name)
    if coin_id:
        symbol = crypto_mapper.id_to_symbol(coin_id)
        return symbol if symbol else name.upper()
    return name.upper()

def get_crypto_info(identifier: str) -> Dict[str, str]:
    """Get comprehensive info for any crypto identifier"""
    return crypto_mapper.get_info(identifier)