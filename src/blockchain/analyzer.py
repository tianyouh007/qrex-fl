"""
QREX-FL Blockchain Analyzer
Multi-blockchain transaction analysis based on research findings
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx

class BlockchainAnalyzer:
    """
    Multi-blockchain analyzer for cryptocurrency risk assessment
    Based on research findings for cross-chain analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_chains = ['bitcoin', 'ethereum', 'binance_smart_chain']
        
    def analyze_transaction_patterns(self, transactions: pd.DataFrame, 
                                   blockchain: str = 'bitcoin') -> Dict[str, Any]:
        """
        Analyze transaction patterns for suspicious activity detection
        
        Args:
            transactions: DataFrame with transaction data
            blockchain: Blockchain type
            
        Returns:
            Analysis results with risk indicators
        """
        try:
            self.logger.info(f"Analyzing {len(transactions)} transactions on {blockchain}")
            
            # Basic statistical analysis
            stats = self._compute_basic_statistics(transactions)
            
            # Temporal pattern analysis
            temporal_patterns = self._analyze_temporal_patterns(transactions)
            
            # Amount pattern analysis
            amount_patterns = self._analyze_amount_patterns(transactions)
            
            # Address reuse analysis
            address_patterns = self._analyze_address_patterns(transactions)
            
            analysis_result = {
                'blockchain': blockchain,
                'transaction_count': len(transactions),
                'analysis_timestamp': datetime.now().isoformat(),
                'basic_statistics': stats,
                'temporal_patterns': temporal_patterns,
                'amount_patterns': amount_patterns,
                'address_patterns': address_patterns,
                'risk_score': self._calculate_overall_risk_score(
                    temporal_patterns, amount_patterns, address_patterns
                )
            }
            
            self.logger.info(f"Analysis completed. Risk score: {analysis_result['risk_score']:.3f}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Transaction analysis failed: {e}")
            raise
    
    def _compute_basic_statistics(self, transactions: pd.DataFrame) -> Dict[str, Any]:
        """Compute basic transaction statistics"""
        amount_col = 'amount' if 'amount' in transactions.columns else 'value'
        
        if amount_col not in transactions.columns:
            return {"error": "No amount/value column found"}
        
        amounts = transactions[amount_col].astype(float)
        
        return {
            'total_volume': float(amounts.sum()),
            'average_amount': float(amounts.mean()),
            'median_amount': float(amounts.median()),
            'std_amount': float(amounts.std()),
            'min_amount': float(amounts.min()),
            'max_amount': float(amounts.max()),
            'unique_senders': transactions['from_address'].nunique() if 'from_address' in transactions.columns else 0,
            'unique_receivers': transactions['to_address'].nunique() if 'to_address' in transactions.columns else 0
        }
    
    def _analyze_temporal_patterns(self, transactions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal transaction patterns"""
        if 'timestamp' not in transactions.columns:
            return {"error": "No timestamp column found"}
        
        # Convert timestamps
        timestamps = pd.to_datetime(transactions['timestamp'])
        transactions_copy = transactions.copy()
        transactions_copy['datetime'] = timestamps
        transactions_copy['hour'] = timestamps.dt.hour
        transactions_copy['day_of_week'] = timestamps.dt.dayofweek
        
        # Hourly distribution
        hourly_dist = transactions_copy['hour'].value_counts().sort_index()
        
        # Daily distribution
        daily_dist = transactions_copy['day_of_week'].value_counts().sort_index()
        
        # Transaction frequency analysis
        transactions_copy['date'] = timestamps.dt.date
        daily_counts = transactions_copy['date'].value_counts()
        
        # Detect burst patterns (high frequency in short time)
        burst_threshold = daily_counts.mean() + 2 * daily_counts.std()
        burst_days = daily_counts[daily_counts > burst_threshold]
        
        return {
            'time_span_days': (timestamps.max() - timestamps.min()).days,
            'avg_daily_transactions': float(daily_counts.mean()),
            'max_daily_transactions': int(daily_counts.max()),
            'burst_days_count': len(burst_days),
            'peak_hour': int(hourly_dist.idxmax()),
            'peak_day_of_week': int(daily_dist.idxmax()),
            'hourly_variance': float(hourly_dist.var()),
            'burst_risk_score': min(len(burst_days) / 10, 1.0)  # Normalize to 0-1
        }
    
    def _analyze_amount_patterns(self, transactions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transaction amount patterns for structuring detection"""
        amount_col = 'amount' if 'amount' in transactions.columns else 'value'
        
        if amount_col not in transactions.columns:
            return {"error": "No amount/value column found"}
        
        amounts = transactions[amount_col].astype(float)
        
        # Round amount detection (potential structuring)
        round_amounts = amounts[amounts % 1000 == 0]
        round_amount_ratio = len(round_amounts) / len(amounts)
        
        # Amount clustering analysis
        amount_bins = np.histogram(amounts, bins=50)[0]
        amount_variance = np.var(amount_bins)
        
        # Detect repeated amounts (suspicious pattern)
        amount_counts = amounts.value_counts()
        repeated_amounts = amount_counts[amount_counts > 1]
        repeated_amount_ratio = len(repeated_amounts) / len(amount_counts)
        
        # Threshold analysis (amounts just below common reporting thresholds)
        thresholds = [3000, 5000, 10000, 15000]  # Common AML thresholds
        near_threshold_counts = {}
        
        for threshold in thresholds:
            # Count transactions 95-99% of threshold (potential avoidance)
            near_threshold = amounts[(amounts >= threshold * 0.95) & (amounts < threshold)]
            near_threshold_counts[f"near_{threshold}"] = len(near_threshold)
        
        return {
            'round_amount_ratio': float(round_amount_ratio),
            'repeated_amount_ratio': float(repeated_amount_ratio),
            'amount_variance': float(amount_variance),
            'near_threshold_counts': near_threshold_counts,
            'structuring_risk_score': min(round_amount_ratio * 2 + repeated_amount_ratio, 1.0)
        }
    
    def _analyze_address_patterns(self, transactions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze address usage patterns"""
        if 'from_address' not in transactions.columns or 'to_address' not in transactions.columns:
            return {"error": "Missing address columns"}
        
        # Address reuse analysis
        from_counts = transactions['from_address'].value_counts()
        to_counts = transactions['to_address'].value_counts()
        
        # Detect high-frequency addresses
        high_freq_senders = from_counts[from_counts > from_counts.quantile(0.95)]
        high_freq_receivers = to_counts[to_counts > to_counts.quantile(0.95)]
        
        # One-time use addresses (privacy-conscious behavior)
        one_time_senders = from_counts[from_counts == 1]
        one_time_receivers = to_counts[to_counts == 1]
        
        # Address overlap (addresses that both send and receive)
        overlapping_addresses = set(transactions['from_address']) & set(transactions['to_address'])
        
        return {
            'unique_senders': len(from_counts),
            'unique_receivers': len(to_counts),
            'high_freq_senders': len(high_freq_senders),
            'high_freq_receivers': len(high_freq_receivers),
            'one_time_sender_ratio': len(one_time_senders) / len(from_counts),
            'one_time_receiver_ratio': len(one_time_receivers) / len(to_counts),
            'overlapping_addresses': len(overlapping_addresses),
            'address_reuse_risk_score': min(len(high_freq_senders) / 100, 1.0)
        }
    
    def _calculate_overall_risk_score(self, temporal: Dict, amounts: Dict, addresses: Dict) -> float:
        """Calculate overall risk score from pattern analysis"""
        risk_factors = []
        
        # Temporal risk factors
        if 'burst_risk_score' in temporal:
            risk_factors.append(temporal['burst_risk_score'])
        
        # Amount pattern risk factors
        if 'structuring_risk_score' in amounts:
            risk_factors.append(amounts['structuring_risk_score'])
        
        # Address pattern risk factors
        if 'address_reuse_risk_score' in addresses:
            risk_factors.append(addresses['address_reuse_risk_score'])
        
        # Weighted average of risk factors
        if risk_factors:
            return sum(risk_factors) / len(risk_factors)
        else:
            return 0.0

    def cross_chain_analysis(self, bitcoin_txs: pd.DataFrame, 
                           ethereum_txs: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform cross-chain analysis to detect laundering patterns
        Based on research findings about $7B+ cross-chain laundering
        """
        try:
            self.logger.info("Performing cross-chain analysis...")
            
            # Analyze each chain separately
            btc_analysis = self.analyze_transaction_patterns(bitcoin_txs, 'bitcoin')
            eth_analysis = self.analyze_transaction_patterns(ethereum_txs, 'ethereum')
            
            # Cross-chain correlation analysis
            correlation_results = self._analyze_cross_chain_correlations(bitcoin_txs, ethereum_txs)
            
            return {
                'bitcoin_analysis': btc_analysis,
                'ethereum_analysis': eth_analysis,
                'cross_chain_correlations': correlation_results,
                'combined_risk_score': (btc_analysis.get('risk_score', 0) + 
                                      eth_analysis.get('risk_score', 0) + 
                                      correlation_results.get('correlation_risk', 0)) / 3
            }
            
        except Exception as e:
            self.logger.error(f"Cross-chain analysis failed: {e}")
            raise
    
    def _analyze_cross_chain_correlations(self, btc_txs: pd.DataFrame, 
                                        eth_txs: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between blockchain transactions"""
        
        # Temporal correlation analysis
        btc_daily = btc_txs.groupby(pd.to_datetime(btc_txs['timestamp']).dt.date).size()
        eth_daily = eth_txs.groupby(pd.to_datetime(eth_txs['timestamp']).dt.date).size()
        
        # Find common dates
        common_dates = set(btc_daily.index) & set(eth_daily.index)
        
        if len(common_dates) > 1:
            btc_common = btc_daily.reindex(common_dates, fill_value=0)
            eth_common = eth_daily.reindex(common_dates, fill_value=0)
            
            # Calculate correlation
            temporal_correlation = np.corrcoef(btc_common.values, eth_common.values)[0, 1]
        else:
            temporal_correlation = 0.0
        
        # Amount correlation analysis (if both have amount data)
        amount_correlation = 0.0
        if 'amount' in btc_txs.columns and 'amount' in eth_txs.columns:
            # Sample equal amounts from both datasets for comparison
            sample_size = min(1000, len(btc_txs), len(eth_txs))
            btc_amounts = btc_txs['amount'].sample(sample_size).astype(float)
            eth_amounts = eth_txs['amount'].sample(sample_size).astype(float)
            
            amount_correlation = np.corrcoef(btc_amounts.values, eth_amounts.values)[0, 1]
        
        # Calculate correlation risk score
        correlation_risk = (abs(temporal_correlation) + abs(amount_correlation)) / 2
        
        return {
            'temporal_correlation': float(temporal_correlation) if not np.isnan(temporal_correlation) else 0.0,
            'amount_correlation': float(amount_correlation) if not np.isnan(amount_correlation) else 0.0,
            'common_analysis_days': len(common_dates),
            'correlation_risk': float(correlation_risk) if not np.isnan(correlation_risk) else 0.0
        }
